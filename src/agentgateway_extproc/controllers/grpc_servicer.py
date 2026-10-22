"""Envoy ExternalProcessor gRPC controller with fail-closed handling."""

# ruff: noqa: BLE001

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import NoReturn, Protocol, cast, override
from uuid import uuid4

import grpc

from agentgateway_extproc.config.settings import Settings
from agentgateway_extproc.gen import ext_proc_pb2, ext_proc_pb2_grpc
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.notice.preferences import NoticePreferencesClient
from agentgateway_extproc.lib.pipeline.diagnostics import (
    diagnostic_json,
    exception_diagnostic,
    exception_summary,
    safe_diagnostic,
    validation_details,
)
from agentgateway_extproc.lib.pipeline.mcp import McpProtocolError, McpUnsupportedFeatureError
from agentgateway_extproc.lib.pipeline.request import immediate_response
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.lib.rejections import annotate_rejection, log_limit_rejection
from agentgateway_extproc.metrics import active_streams, errors_total, response_failures_total
from agentgateway_extproc.models.exceptions import (
    ContextForgeAccountRequiredError,
    EnginePolicyError,
    EngineUnavailableError,
    GatewayPolicyError,
    InvalidEngineReplyError,
    InvalidReversalError,
    McpHttpError,
    TrustedMetadataError,
)

_logger = logging.getLogger(__name__)


class AbortContext(Protocol):
    """Describe the aio context operation needed for a committed stream failure."""

    async def abort(self, code: grpc.StatusCode, details: str = "") -> NoReturn:
        """Terminate the gRPC stream with a fixed safe status."""


@dataclass
class StreamPhase:
    """Track whether an upstream response can still become an ImmediateResponse."""

    response_headers_committed: bool = False

    def observe(self, response: ext_proc_pb2.ProcessingResponse) -> None:
        """Mark the downstream response committed when its header acknowledgement is sent."""
        if response.HasField("response_headers"):
            self.response_headers_committed = True


class ExtProcServicer(ext_proc_pb2_grpc.ExternalProcessorServicer):
    """Create isolated stream handlers and never forward failed processing."""

    def __init__(
        self,
        client: EngineClient,
        settings: Settings | None = None,
        docling: DoclingClient | None = None,
        preferences_client: NoticePreferencesClient | None = None,
    ) -> None:
        """Store the shared engine client used by all streams."""
        self._client = client
        self._settings = settings or Settings()
        self._docling = docling
        self._preferences_client = preferences_client
        self._active = 0
        self._buffered_bytes = 0

    @override
    async def Process(
        self, request_iterator: AsyncIterator[ext_proc_pb2.ProcessingRequest], context: object
    ) -> AsyncIterator[ext_proc_pb2.ProcessingResponse]:
        """Process one Envoy bidirectional stream."""
        correlation_id = _initial_request_id(context)
        if self._active >= self._settings.grpc_maximum_concurrent_rpcs:
            yield _overloaded(correlation_id, debug=self._settings.error_log_level == "debug")
            return
        self._active += 1
        try:
            async with aclosing(self._process(request_iterator, context, correlation_id)) as stream:
                async for response in stream:
                    yield response
        finally:
            self._active -= 1

    def _reserve_body(self, request: ext_proc_pb2.ProcessingRequest) -> int | None:
        size = len(request.request_body.body) if request.HasField("request_body") else 0
        # Bound retained input before JSON/base64 decoding; Docling separately
        # admits one native conversion per Pod.
        if self._buffered_bytes + size > 4 * self._settings.max_request_bytes:
            return None
        self._buffered_bytes += size
        return size

    async def _process(
        self,
        request_iterator: AsyncIterator[ext_proc_pb2.ProcessingRequest],
        context: object,
        correlation_id: str | None,
    ) -> AsyncGenerator[ext_proc_pb2.ProcessingResponse, None]:
        active_streams.inc()
        handler = StreamHandler(
            self._client,
            self._settings,
            self._docling,
            self._preferences_client,
            correlation_id=correlation_id,
        )
        phase = StreamPhase()
        buffered_bytes = 0
        try:
            async for request in request_iterator:
                kind = request.WhichOneof("request") or "unknown"
                size = self._reserve_body(request)
                if size is None:
                    yield _overloaded(handler.correlation_id, debug=handler.debug_errors)
                    return
                buffered_bytes += size
                try:
                    outputs = await _handle_message(handler, request, kind)
                    if handler.request_processed:
                        self._buffered_bytes -= buffered_bytes
                        buffered_bytes = 0
                    for output in outputs:
                        self._annotate_rejection(handler, output)
                        phase.observe(output)
                        yield output
                except asyncio.CancelledError:
                    # The peer can no longer receive an immediate response.
                    # Teardown still clears the stream-local reversal state.
                    raise
                except Exception as exc:
                    _record_dispatch_failure(handler, exc)
                    _log_committed_limit(handler, phase, exc)
                    # Current header values are available only for this failure, never retained.
                    input_data = (
                        {
                            item.key: [value, *re.split(r"[\s;=]+", value)]
                            for item in request.request_headers.headers.headers
                            if re.search(
                                r"authorization|api.?key|token|password|secret|cookie",
                                item.key,
                                re.I,
                            )
                            for value in [
                                item.raw_value.decode("utf-8", errors="replace") or item.value
                            ]
                        }
                        if request.HasField("request_headers")
                        else None
                    )
                    output = await _failure_for_phase(
                        context, phase, kind, exc, handler, input_data=input_data
                    )
                    self._annotate_rejection(handler, output, exc, input_data=input_data)
                    yield output
                    return
                if outputs and outputs[-1].HasField("immediate_response"):
                    return
            try:
                outputs = await _finish_stream(handler)
                for output in outputs:
                    self._annotate_rejection(handler, output)
                    phase.observe(output)
                    yield output
            except Exception as exc:
                _record_dispatch_failure(handler, exc)
                _log_committed_limit(handler, phase, exc)
                output = await _failure_for_phase(context, phase, "response_eof", exc, handler)
                self._annotate_rejection(handler, output, exc)
                yield output
        finally:
            handler.clear_sensitive_state()
            self._buffered_bytes -= buffered_bytes
            active_streams.dec()

    def _annotate_rejection(
        self,
        handler: StreamHandler,
        response: ext_proc_pb2.ProcessingResponse,
        error: Exception | None = None,
        *,
        input_data: object = None,
    ) -> None:
        if not response.HasField("immediate_response"):
            return
        limit = (
            error.limit
            if isinstance(error, (EnginePolicyError, InvalidEngineReplyError, GatewayPolicyError))
            else None
        )
        code = annotate_rejection(
            response,
            handler.correlation_id,
            limit=limit,
            model=handler.destination_policy is not None
            and handler.destination_policy.destination_kind == "model",
        )
        status = response.immediate_response.status.code
        parsed = json.loads(response.immediate_response.body)
        parsed["error"]["message"] = json.loads(
            safe_diagnostic(parsed["error"]["message"], handler, error, input_data=input_data)
        )
        if handler.debug_errors:
            detail = (
                exception_diagnostic(error, handler, input_data=input_data)
                if error is not None
                else f"rejection code={code}; http.status={status}"
            )
            parsed["error"]["message"] += (
                f". Diagnostics: {detail}; request_id={handler.correlation_id}"
            )
        response.immediate_response.body = json.dumps(parsed, ensure_ascii=True)
        _logger.error(
            "gateway rejection request_id=%s http.status=%d reason=%s",
            handler.correlation_id,
            status,
            code,
        )
        if handler.admission is not None:
            error_body = json.loads(response.immediate_response.body).get("error", {})
            admission = handler.admission.request_context
            _logger.warning(
                "gateway rejection request_id=%s binding_id=%s provider_id=%s profile=%s "
                "model=%s phase=%s http.status=%d reason=%s param=%s",
                handler.correlation_id,
                admission.binding_id,
                admission.provider_id or "none",
                admission.protocol_profile,
                handler.request_context.resolved_model if handler.request_context else "unresolved",
                "provider_response" if handler.response_started else "request_admission",
                status,
                code,
                error_body.get("param", "none")
                if code in {"provider_unsupported_parameter", "provider_stream_required"}
                else "none",
            )


def _initial_request_id(context: object) -> str | None:
    """Read only a single bounded gateway correlation value from gRPC stream metadata."""
    metadata = getattr(context, "invocation_metadata", None)
    if not callable(metadata):
        return None
    values = [value for key, value in metadata() if key == "x-request-id"]
    if (
        len(values) == 1
        and isinstance(values[0], str)
        and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", values[0])
    ):
        return values[0]
    return None


def _log_committed_limit(handler: StreamHandler, phase: StreamPhase, exc: Exception) -> None:
    if (
        phase.response_headers_committed
        and isinstance(exc, (EnginePolicyError, InvalidEngineReplyError))
        and exc.limit is not None
    ):
        log_limit_rejection(handler.correlation_id, exc.limit)


def _overloaded(
    correlation_id: str | None = None, *, debug: bool = False
) -> ext_proc_pb2.ProcessingResponse:
    """Reject before forwarding any request or unprocessed response content."""
    errors_total.labels(type="overloaded").inc()
    response = immediate_response(
        503,
        '{"error":{"message":"processor busy","code":"capacity_unavailable","retryable":true}}',
    )
    correlation_id = correlation_id or uuid4().hex
    annotate_rejection(response, correlation_id)
    if debug:
        parsed = json.loads(response.immediate_response.body)
        parsed["error"]["message"] += (
            f": capacity_unavailable; concurrency or buffer capacity exhausted; "
            f"retry after 1 second; request_id={correlation_id}"
        )
        response.immediate_response.body = json.dumps(parsed)
    _logger.error(
        "gateway rejection request_id=%s http.status=503 reason=capacity_unavailable",
        correlation_id,
    )
    response.immediate_response.headers.set_headers.add(
        header={"key": "retry-after", "value": "1"}, append_action=2
    )
    return response


def _ordered_response(
    handler: StreamHandler, response: ext_proc_pb2.ProcessingResponse
) -> tuple[ext_proc_pb2.ProcessingResponse, ...]:
    """Emit a deferred JSON header acknowledgement before its validated body."""
    if response.HasField("response_body"):
        headers = handler.pop_pending_response_headers()
        if headers is not None:
            return headers, response
    return (response,)


async def _finish_stream(handler: StreamHandler) -> tuple[ext_proc_pb2.ProcessingResponse, ...]:
    """Finalize an incomplete request or flush a normal response iterator EOF."""
    incomplete_request = handler.finish_request()
    if incomplete_request is not None:
        return (incomplete_request,)
    final_body = await handler.finish_response()
    if final_body is None:
        return ()
    return _ordered_response(handler, final_body)


async def _handle_message(
    handler: StreamHandler,
    request: ext_proc_pb2.ProcessingRequest,
    kind: str,
) -> tuple[ext_proc_pb2.ProcessingResponse, ...]:
    """Handle one input while preserving full-duplex headers-body-trailers order."""
    expected_response = {
        "request_headers": "request_headers",
        "request_body": "request_body",
        "request_trailers": "request_trailers",
    }.get(kind)
    if expected_response is not None:
        response = await handler.handle(request)
        if response is None or response.WhichOneof("response") not in {
            expected_response,
            "immediate_response",
        }:
            raise ValueError("request phase produced a mismatched response")  # noqa: TRY003
        return (response,)

    outputs: list[ext_proc_pb2.ProcessingResponse] = []
    # handle() validates admission before any trailer-triggered output is flushed.
    response = await handler.handle(request)
    if kind == "response_trailers":
        final_body = await handler.finish_response()
        if final_body is not None:
            outputs.extend(_ordered_response(handler, final_body))
            if final_body.HasField("immediate_response"):
                return tuple(outputs)
    if response is not None:
        outputs.extend(_ordered_response(handler, response))
    return tuple(outputs)


def _failure_response(
    phase: str, exc: Exception, handler: StreamHandler, *, input_data: object = None
) -> ext_proc_pb2.ProcessingResponse:
    """Record only bounded response failure context and fail the stream closed."""
    _record_failure(phase, exc, handler, input_data=input_data)
    if isinstance(exc, GatewayPolicyError):
        return immediate_response(
            exc.status_code,
            json.dumps(
                {"error": {"code": exc.code, "message": exc.message, "retryable": exc.retryable}},
                separators=(",", ":"),
            ),
        )
    if isinstance(exc, ContextForgeAccountRequiredError):
        return immediate_response(
            403,
            '{"error":{"code":"contextforge_account_required",'
            '"message":"A verified account email is required. '
            'Connect your ContextForge account."}}',
        )
    if isinstance(exc, McpUnsupportedFeatureError):
        return immediate_response(400, exc.body)
    if isinstance(exc, McpHttpError):
        response = immediate_response(exc.status_code, '{"error":"MCP HTTP request failed"}')
        for key, value in exc.headers.items():
            response.immediate_response.headers.set_headers.add(
                header={"key": key, "value": value}, append_action=2
            )
        return response
    if isinstance(exc, EnginePolicyError):
        body = json.dumps(
            {
                "error": {
                    "message": exc.message,
                    "type": "pii_engine_error",
                    "param": None,
                    "code": exc.code,
                    "retryable": exc.retryable,
                }
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return immediate_response(exc.status_code, body)
    return immediate_response(
        503,
        json.dumps(
            {
                "error": f"Processing failed: {_failure_reason(exc)} ({exception_summary(exc)}); "
                f"request_id={handler.correlation_id}"
            }
        ),
    )


async def _failure_for_phase(
    context: object,
    state: StreamPhase,
    phase: str,
    exc: Exception,
    handler: StreamHandler,
    *,
    input_data: object = None,
) -> ext_proc_pb2.ProcessingResponse:
    """Return a precommit local reply or abort a response already committed downstream."""
    if state.response_headers_committed:
        await _abort_stream(context, phase, exc, handler, input_data=input_data)
    return _failure_response(phase, exc, handler, input_data=input_data)


async def _abort_stream(
    context: object,
    phase: str,
    exc: Exception,
    handler: StreamHandler,
    *,
    input_data: object = None,
) -> NoReturn:
    """Abort a response whose headers have already been committed downstream."""
    _record_failure(phase, exc, handler, input_data=input_data)
    grpc_context = cast(AbortContext, context)
    detail = f"response processing failed closed; request_id={handler.correlation_id}"
    if handler.debug_errors:
        detail += "; " + exception_diagnostic(exc, handler, input_data=input_data)
    await grpc_context.abort(grpc.StatusCode.INTERNAL, detail)


def _record_failure(
    phase: str, exc: Exception, handler: StreamHandler, *, input_data: object = None
) -> None:
    """Keep normal logs concise; DEBUG adds full request and failure context."""
    reason = _failure_reason(exc)
    errors_total.labels(type="processing").inc()
    if phase in {"response_headers", "response_body", "response_trailers", "response_eof"}:
        response_failures_total.labels(phase=phase, reason=reason).inc()
    _logger.error(
        "ext_proc processing failed closed request_id=%s binding_id=%s phase=%s reason=%s "
        "error=%s diagnostics=%s",
        handler.correlation_id,
        handler.admission.request_context.binding_id if handler.admission else "unresolved",
        phase,
        reason,
        type(exc).__name__,
        exception_diagnostic(exc, handler, stack=True, input_data=input_data),
    )
    if handler.debug_logging and _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(
            "ext_proc failure diagnostics request_id=%s phase=%s details=%s request=%s",
            handler.correlation_id,
            phase,
            diagnostic_json(
                {
                    "validation": validation_details(exc),
                    "response": handler.failed_response,
                    "headers": handler.request_headers,
                },
            ),
            handler.request_diagnostic,
        )


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, GatewayPolicyError):
        return exc.code
    if isinstance(exc, EnginePolicyError):
        return f"engine_{exc.code}"
    if isinstance(exc, InvalidReversalError):
        return "invalid_reversal"
    if isinstance(exc, InvalidEngineReplyError):
        return "invalid_engine_reply"
    if isinstance(exc, EngineUnavailableError):
        return "engine_unavailable"
    if isinstance(exc, TrustedMetadataError):
        return "invalid_metadata"
    if isinstance(exc, (UnicodeError, ValueError)):
        return "invalid_data"
    return "internal"


def _record_dispatch_failure(handler: StreamHandler, exc: Exception) -> None:
    """Classify failures without request-derived metric labels."""
    if isinstance(exc, McpHttpError):
        return
    if isinstance(exc, McpProtocolError):
        handler.record_dispatch("protocol_failure")
    elif (
        isinstance(exc, (EngineUnavailableError, InvalidEngineReplyError))
        or handler.response_api_kind == "mcp"
    ):
        handler.record_dispatch("transport_failure")
