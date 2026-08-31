"""Typed, mTLS-capable HTTP client for the PII engine adapter endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import ssl
import time

import httpx

from agentgateway_extproc.config.settings import EngineSettings
from agentgateway_extproc.metrics import engine_request_latency_seconds, engine_requests_total
from agentgateway_extproc.models.engine import EngineErrorReply, EngineReply, EngineRequest
from agentgateway_extproc.models.exceptions import (
    ENGINE_ERROR_CONTRACT,
    EnginePolicyError,
    EngineUnavailableError,
    InvalidEngineReplyError,
)

_logger = logging.getLogger(__name__)


class EngineClient:
    """Call and strictly validate the PII engine adapter contract."""

    def __init__(self, settings: EngineSettings, client: httpx.AsyncClient | None = None) -> None:
        """Configure the client; an injected client is useful for tests."""
        tls_context: ssl.SSLContext | bool = True
        if client is None:
            ca_cert = settings.ca_cert
            client_cert = settings.client_cert
            client_key = settings.client_key
            if (
                not settings.base_url.startswith("https://")
                or not ca_cert
                or not client_cert
                or not client_key
            ):
                raise ValueError(  # noqa: TRY003
                    "production engine client requires HTTPS and client mTLS"
                )
            tls_context = ssl.create_default_context(cafile=ca_cert)
            tls_context.load_cert_chain(certfile=client_cert, keyfile=client_key)
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.timeout),
            verify=tls_context,
            headers={"content-type": "application/json"},
        )
        self._owns_client = client is None

    async def close(self) -> None:
        """Close the HTTP client when this instance owns it."""
        if self._owns_client:
            await self._client.aclose()

    async def check_ready(self) -> None:
        """Verify the ready PII Service endpoint through the adapter mTLS path."""
        try:
            async with asyncio.timeout(self._settings.readiness_timeout):
                response = await self._client.get("/v1/adapter/ready")
                response.raise_for_status()
        except (TimeoutError, httpx.HTTPError) as exc:
            _logger.debug("PII engine readiness failed error=%s", type(exc).__name__)
            raise EngineUnavailableError from exc

    async def analyze_request(self, request: EngineRequest, session_key: str) -> EngineReply:
        """Send a complete request to the engine and validate its complete reply."""
        started = time.monotonic()
        outcome = "error"
        try:
            async with self._client.stream(
                "POST",
                f"{self._settings.base_url.rstrip('/')}/v1/adapter/analyze-request",
                json=request.model_dump(by_alias=True, exclude_none=True),
                headers={"x-pii-session-key": session_key},
            ) as response:
                content = await _read_bounded(response, self._settings.max_response_bytes)
            if not response.is_success:
                raise _parse_engine_error(response.status_code, content)
            try:
                payload = json.loads(
                    content,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                    parse_float=_finite_float,
                )
                reply = EngineReply.model_validate(payload, strict=True)
            except InvalidEngineReplyError:
                raise
            except (TypeError, ValueError) as exc:
                raise InvalidEngineReplyError from exc
            else:
                outcome = "success"
                return reply
        except EnginePolicyError as exc:
            outcome = f"rejected_{exc.code}"
            raise
        except InvalidEngineReplyError:
            outcome = "invalid_reply"
            raise
        except (httpx.HTTPError, ValueError) as exc:
            _logger.warning("PII engine request failed error=%s", type(exc).__name__)
            raise EngineUnavailableError from exc
        finally:
            engine_requests_total.labels(outcome=outcome).inc()
            engine_request_latency_seconds.labels(outcome=outcome).observe(
                time.monotonic() - started
            )


async def _read_bounded(response: httpx.Response, limit: int) -> bytes:
    """Read decoded response chunks without allocating beyond the engine limit."""
    content_lengths = response.headers.get_list("content-length")
    if len(content_lengths) == 1:
        raw_length = content_lengths[0]
        if raw_length.isascii() and raw_length.isdecimal() and int(raw_length) > limit:
            raise InvalidEngineReplyError
    content = bytearray()
    async for chunk in response.aiter_bytes():
        if len(content) + len(chunk) > limit:
            raise InvalidEngineReplyError
        content.extend(chunk)
    return bytes(content)


def _parse_engine_error(status_code: int, content: bytes) -> EnginePolicyError:
    """Validate a non-success envelope and bind its code to the HTTP status."""
    try:
        payload = json.loads(
            content,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
        reply = EngineErrorReply.model_validate(payload, strict=True)
    except InvalidEngineReplyError:
        raise
    except (TypeError, ValueError) as exc:
        raise InvalidEngineReplyError from exc
    expected = ENGINE_ERROR_CONTRACT[reply.error.code]
    if (status_code, reply.error.message, reply.error.retryable) != expected:
        raise InvalidEngineReplyError
    return EnginePolicyError(reply.error.code)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys before they can collapse trusted reply fields."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidEngineReplyError
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    """Reject NaN and infinity in trusted engine JSON."""
    raise InvalidEngineReplyError


def _finite_float(value: str) -> float:
    """Reject JSON exponent overflow before typed engine validation."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise InvalidEngineReplyError
    return parsed
