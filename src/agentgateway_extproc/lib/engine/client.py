"""Typed, mTLS-capable HTTP client for the PII engine adapter endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import ssl
import time
from uuid import uuid4

import httpx
from neurwerk_request_segments import ExtractedRequest, TextSegment

from agentgateway_extproc.config.settings import EngineSettings
from agentgateway_extproc.lib.json_limits import JsonBudgetError, bounded_json_text
from agentgateway_extproc.lib.request_segments import extract_request
from agentgateway_extproc.metrics import engine_request_latency_seconds, engine_requests_total
from agentgateway_extproc.models.engine import (
    EngineErrorReply,
    EngineLimitErrorReply,
    EngineReply,
    EngineRequest,
    VisualFindings,
)
from agentgateway_extproc.models.exceptions import (
    ENGINE_ERROR_CONTRACT,
    EnginePolicyError,
    EngineUnavailableError,
    InvalidEngineReplyError,
    LimitDetail,
)

_logger = logging.getLogger(__name__)


def _correlation_id(value: str | None) -> str:
    """Reuse a bounded transport identifier or generate a content-independent one."""
    return value if value and re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value) else uuid4().hex


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
                response = await self._client.get("/v2/adapter/ready")
                response.raise_for_status()
        except (TimeoutError, httpx.HTTPError) as exc:
            _logger.debug("PII engine readiness failed error=%s", type(exc).__name__)
            raise EngineUnavailableError from exc

    async def analyze_request(
        self,
        request: EngineRequest,
        session_key: str,
        *,
        document: bool = False,
        text_pii_enabled: bool = True,
        visual_findings: VisualFindings | None = None,
        correlation_id: str | None = None,
    ) -> EngineReply:
        """Send only opaque segments and rebuild the validated reply locally."""
        started = time.monotonic()
        outcome = "error"
        correlation_id = _correlation_id(correlation_id)
        extracted = extract_request(request)
        wire: dict[str, object] = {
            "api_version": "v2",
            "request_kind": extracted.request_kind,
            "scope": "request" if document else "session",
            "segments": [segment.model_dump() for segment in extracted.segments],
            "text_pii_enabled": text_pii_enabled,
            "attachments_present": extracted.attachments_present,
        }
        if (visual_findings is not None and not document) or (
            visual_findings is None and not text_pii_enabled
        ):
            raise ValueError("visual controls require the document envelope")  # noqa: TRY003
        if visual_findings is not None:
            wire["visual_findings"] = visual_findings.model_dump()
        try:
            async with (
                asyncio.timeout(self._settings.timeout),
                self._client.stream(
                    "POST",
                    f"{self._settings.base_url.rstrip('/')}/v2/adapter/analyze-segments",
                    json=wire,
                    headers={"x-pii-session-key": session_key, "x-correlation-id": correlation_id},
                ) as response,
            ):
                content = await _read_bounded(response, self._settings.max_response_bytes)
            if not response.is_success:
                raise _parse_engine_error(
                    response.status_code, content, correlation_id=correlation_id
                )
            try:
                payload = json.loads(
                    _bounded_reply_json(content),
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                    parse_float=_finite_float,
                )
                reply = _rebuild_reply(payload, extracted)
                _validate_visual_reply(reply, request, visual_findings, text_pii_enabled)
            except InvalidEngineReplyError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidEngineReplyError from exc
            else:
                outcome = "success"
                return reply
        except EnginePolicyError as exc:
            outcome = f"rejected_{exc.code}"
            raise
        except InvalidEngineReplyError as exc:
            outcome = "invalid_reply"
            exc.correlation_id = correlation_id
            raise
        except (TimeoutError, httpx.HTTPError, ValueError) as exc:
            _logger.warning("PII engine request failed error=%s", type(exc).__name__)
            raise EngineUnavailableError from exc
        finally:
            engine_requests_total.labels(outcome=outcome).inc()
            engine_request_latency_seconds.labels(outcome=outcome).observe(
                time.monotonic() - started
            )


def _rebuild_reply(payload: object, extracted: ExtractedRequest) -> EngineReply:
    """Validate the v2 shape and preserve every local provider control."""
    if not isinstance(payload, dict) or payload.get("api_version") != "v2" or "request" in payload:
        raise InvalidEngineReplyError
    segments = payload.pop("segments")
    if payload.get("decision") == "block":
        if segments is not None:
            raise InvalidEngineReplyError
        rebuilt = None
    else:
        if not isinstance(segments, list):
            raise InvalidEngineReplyError
        rebuilt = extracted.rebuild(
            [TextSegment.model_validate(segment, strict=True) for segment in segments]
        )
    # The request-bearing object is an internal pipeline convenience only.
    reply = EngineReply.model_validate(
        {**payload, "api_version": "v1", "request": rebuilt}, strict=True
    )
    if (
        rebuilt is not None
        and (
            reply.decision == "pass"
            or reply.analysis.source == "cached_decision"
            or not reply.analysis.scan_performed
            or not any(row.transformed_count for row in reply.report.rows)
        )
        and extracted.segments != extract_request(rebuilt).segments
    ):
        raise InvalidEngineReplyError
    return reply


def _validate_visual_reply(
    reply: EngineReply,
    request: EngineRequest,
    findings: VisualFindings | None,
    text_pii_enabled: bool,
) -> None:
    """Require an exact findings echo and respect the trusted text-analysis switch."""
    if findings is None:
        if "visual_findings" in reply.model_fields_set:
            raise InvalidEngineReplyError
        return
    if reply.visual_findings != findings:
        raise InvalidEngineReplyError
    if text_pii_enabled:
        if reply.decision != "block" and (
            not reply.analysis.scan_performed or reply.analysis.text_leaf_count < 1
        ):
            raise InvalidEngineReplyError
    elif (
        reply.analysis.scan_performed
        or set(reply.entities) - {"FACE"}
        or reply.reversal
        or (reply.request is not None and reply.request != request)
        or set(reply.applied_actions) - {"pass", "block", "text-only", "reroute"}
    ):
        raise InvalidEngineReplyError


async def _read_bounded(response: httpx.Response, limit: int) -> bytes:
    """Read decoded response chunks without allocating beyond the engine limit."""
    content_lengths = response.headers.get_list("content-length")
    if len(content_lengths) == 1:
        raw_length = content_lengths[0]
        if raw_length.isascii() and raw_length.isdecimal() and int(raw_length) > limit:
            raise InvalidEngineReplyError(
                limit=LimitDetail(
                    stage="engine_response",
                    reason="declared_bytes",
                    measured=int(raw_length),
                    maximum=limit,
                    unit="bytes",
                    exact=True,
                )
            )
    content = bytearray()
    async for chunk in response.aiter_bytes():
        if len(content) + len(chunk) > limit:
            raise InvalidEngineReplyError(
                limit=LimitDetail(
                    stage="engine_response",
                    reason="decoded_bytes",
                    measured=len(content) + len(chunk),
                    maximum=limit,
                    unit="bytes",
                    exact=False,
                )
            )
        content.extend(chunk)
    return bytes(content)


def _bounded_reply_json(content: bytes) -> str:
    """Preserve stopped structural counts when translating an invalid engine reply."""
    try:
        return bounded_json_text(content)
    except JsonBudgetError as exc:
        raise InvalidEngineReplyError(limit=exc.limit) from exc


def _parse_engine_error(
    status_code: int,
    content: bytes,
    *,
    correlation_id: str | None = None,
) -> EnginePolicyError:
    """Validate a non-success envelope and bind its code to the HTTP status."""
    try:
        payload = json.loads(
            _bounded_reply_json(content),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
        reply = (
            EngineLimitErrorReply.model_validate(payload, strict=True)
            if isinstance(payload, dict) and payload.get("api_version") == "v2"
            else EngineErrorReply.model_validate(payload, strict=True)
        )
    except InvalidEngineReplyError:
        raise
    except (TypeError, ValueError) as exc:
        raise InvalidEngineReplyError from exc
    expected = ENGINE_ERROR_CONTRACT[reply.error.code]
    if (status_code, reply.error.message, reply.error.retryable) != expected:
        raise InvalidEngineReplyError
    return EnginePolicyError(
        reply.error.code,
        limit=reply.error.limit if isinstance(reply, EngineLimitErrorReply) else None,
        correlation_id=correlation_id,
    )


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
