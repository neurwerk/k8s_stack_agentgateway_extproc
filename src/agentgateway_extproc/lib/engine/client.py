"""Typed, mTLS-capable HTTP client for the PII engine adapter endpoint."""

from __future__ import annotations

import asyncio
import logging
import re
import ssl
import time
from uuid import uuid4

import httpx
from neurwerk_request_segments import ExtractedRequest, TextSegment

from agentgateway_extproc.config.settings import EngineSettings
from agentgateway_extproc.lib.json_limits import JsonBudgetError, strict_json_loads
from agentgateway_extproc.lib.pipeline.mcp_sources import SOURCE_PATTERN, source_union
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
    GatewayPolicyError,
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

    async def reasoning_receipts(
        self,
        scope: str,
        digests: list[str],
        *,
        record: bool,
        correlation_id: str,
        policy_version: str | None = None,
    ) -> None:
        """Exchange only fingerprints over the existing workload-authenticated channel."""
        try:
            async with (
                asyncio.timeout(self._settings.timeout),
                self._client.stream(
                    "PUT" if record else "POST",
                    f"{self._settings.base_url.rstrip('/')}/v2/adapter/reasoning-receipts",
                    json={"scope": scope, "digests": digests, "policy_version": policy_version},
                    headers={"x-correlation-id": _correlation_id(correlation_id)},
                ) as response,
            ):
                if response.status_code == 204:
                    return
                if not record and response.status_code == 403:
                    raise GatewayPolicyError("unverified_reasoning")
                if response.status_code == 413:
                    error = _parse_engine_error(
                        response.status_code,
                        await _read_bounded(response, self._settings.max_response_bytes),
                        correlation_id=correlation_id,
                    )
                    if error.limit and error.limit.reason == "reasoning_fingerprints":
                        raise GatewayPolicyError("reasoning_history_limit", limit=error.limit)
                    raise error
                _logger.warning(
                    "reasoning receipts failed request_id=%s operation=%s http.status=%d",
                    _correlation_id(correlation_id),
                    "record" if record else "verify",
                    response.status_code,
                )
                raise GatewayPolicyError("reasoning_service_unavailable")
        except (TimeoutError, httpx.HTTPError) as exc:
            _logger.warning(
                "reasoning receipts failed request_id=%s operation=%s error=%s",
                _correlation_id(correlation_id),
                "record" if record else "verify",
                type(exc).__name__,
            )
            raise GatewayPolicyError("reasoning_service_unavailable") from exc

    async def analyze_request(
        self,
        request: EngineRequest,
        session_key: str,
        *,
        document: bool = False,
        text_pii_enabled: bool = True,
        visual_findings: VisualFindings | None = None,
        mcp_sources: list[str] | None = None,
        destination_model: str | None = None,
        correlation_id: str | None = None,
    ) -> EngineReply:
        """Send only opaque segments and rebuild the validated reply locally."""
        started = time.monotonic()
        outcome = "error"
        correlation_id = _correlation_id(correlation_id)
        extracted = extract_request(request)
        sources = _request_sources(mcp_sources, extracted.request_kind)
        wire: dict[str, object] = {
            "api_version": "v2",
            "request_kind": extracted.request_kind,
            "scope": "request" if document else "session",
            "segments": [segment.model_dump() for segment in extracted.segments],
            "text_pii_enabled": text_pii_enabled,
            "attachments_present": extracted.attachments_present,
        }
        wire.update(_analysis_controls(document, text_pii_enabled, visual_findings, sources))
        if destination_model is not None:
            wire["destination_model"] = destination_model
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
                payload = _parse_reply_json(content)
                reply = _rebuild_reply(payload, extracted, sources, text_pii_enabled)
                _validate_visual_reply(reply, request, visual_findings, text_pii_enabled)
                _validate_source_reply(reply, request, sources, text_pii_enabled)
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


def _analysis_controls(
    document: bool, text_pii_enabled: bool, findings: VisualFindings | None, sources: list[str]
) -> dict[str, object]:
    if (findings is not None and not document) or (
        findings is None and not text_pii_enabled and not sources
    ):
        raise ValueError("visual controls require the document envelope")  # noqa: TRY003
    controls: dict[str, object] = {}
    if sources:
        controls["mcp_sources"] = sources
    if findings is not None:
        controls["visual_findings"] = findings.model_dump()
    return controls


def _request_sources(sources: list[str] | None, kind: str) -> list[str]:
    sources = source_union(sources or [])
    if any(len(source) > 64 or re.fullmatch(SOURCE_PATTERN, source) is None for source in sources):
        raise ValueError("invalid MCP source classification")  # noqa: TRY003
    if sources and kind == "mcp":
        raise ValueError("MCP sources require a model request")  # noqa: TRY003
    return sources


def _rebuild_reply(
    payload: object, extracted: ExtractedRequest, sources: list[str], text_pii_enabled: bool
) -> EngineReply:
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
        {**payload, "api_version": "v1", "request": rebuilt},
        strict=True,
        context={"mcp_sources": sources, "text_pii_enabled": text_pii_enabled},
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
        or any(entity != "FACE" and not entity.startswith("MCP_TOOL_") for entity in reply.entities)
        or reply.reversal
        or (reply.request is not None and reply.request != request)
        or set(reply.applied_actions) - {"pass", "block", "text-only", "reroute"}
    ):
        raise InvalidEngineReplyError


def _validate_source_reply(
    reply: EngineReply, request: EngineRequest, sources: list[str], text_pii_enabled: bool
) -> None:
    """Keep source policy facts separate from text scans and transformations."""
    if reply.decision != "block" and not set(sources) <= set(reply.entities):
        raise InvalidEngineReplyError
    for row in reply.report.rows:
        if row.entity_type.startswith("MCP_TOOL_") and (
            row.action not in {"pass", "block", "reroute"}
            or row.detected_count != 1
            or row.transformed_count
            or row.unique_transformed_count
            or (not reply.analysis.cached_decision_applied and row.entity_type not in sources)
        ):
            raise InvalidEngineReplyError
    current_entities = set(reply.entities)
    if reply.analysis.cached_decision_applied:
        # Cached conflict blocks also merge historical report rows, but forward no content.
        current_entities = (
            {row.entity_type for row in reply.report.rows}
            if reply.analysis.source == "current_request" and reply.decision != "block"
            else set()
        )
    if not text_pii_enabled and (
        reply.analysis.scan_performed
        or reply.reversal
        or any(
            entity != "FACE" and not entity.startswith("MCP_TOOL_") for entity in current_entities
        )
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


def _parse_reply_json(content: bytes) -> object:
    """Preserve stopped structural counts when translating an invalid engine reply."""
    try:
        return strict_json_loads(content)
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
        payload = _parse_reply_json(content)
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
