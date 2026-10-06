"""Envoy ExternalProcessor gRPC controller with fail-closed handling."""

# ruff: noqa: BLE001

from __future__ import annotations

import asyncio
import json
import logging
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
from agentgateway_extproc.lib.pipeline.mcp import McpProtocolError, McpUnsupportedFeatureError
from agentgateway_extproc.lib.pipeline.request import immediate_response
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.lib.rejection_capture import RejectionCapture, RejectionCaptureConfig
from agentgateway_extproc.lib.rejections import annotate_rejection, log_limit_rejection
from agentgateway_extproc.metrics import active_streams, errors_total, response_failures_total
from agentgateway_extproc.models.exceptions import (
    EnginePolicyError,
    EngineUnavailableError,
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
        self._capture = RejectionCapture(
            RejectionCaptureConfig(**self._settings.rejection_capture.model_dump())
        )
        self._active = 0
        self._buffered_bytes = 0

    @override
    async def Process(
        self, request_iterator: AsyncIterator[ext_proc_pb2.ProcessingRequest], context: object
    ) -> AsyncIterator[ext_proc_pb2.ProcessingResponse]:
        """Process one Envoy bidirectional stream."""
        if self._active >= self._settings.grpc_maximum_concurrent_rpcs:
            yield _overloaded()
            return
        self._active += 1
        try:
            async with aclosing(self._process(request_iterator, context)) as stream:
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

    def _capture_input(self, request: ext_proc_pb2.ProcessingRequest) -> tuple[bytes, int, bool]:
        if not request.HasField("request_body") or not self._capture.config.enabled:
            return b"", 0, False
        inbound = request.request_body.body
        original = inbound[: self._capture.config.max_file_bytes]
        complete = request.request_body.end_of_stream and len(original) == len(inbound)
        return original, len(inbound), complete

    async def _process(
        self, request_iterator: AsyncIterator[ext_proc_pb2.ProcessingRequest], context: object
    ) -> AsyncGenerator[ext_proc_pb2.ProcessingResponse, None]:
        active_streams.inc()
        handler = StreamHandler(
            self._client, self._settings, self._docling, self._preferences_client
        )
        phase = StreamPhase()
        buffered_bytes = 0
        original_body = b""
        observed_bytes = 0
        body_complete = False
        try:
            async for request in request_iterator:
                kind = request.WhichOneof("request") or "unknown"
                original_body, observed_bytes, body_complete = self._capture_input(request)
                size = self._reserve_body(request)
                if size is None:
                    yield _overloaded(handler.correlation_id)
                    return
                buffered_bytes += size
                try:
                    outputs = await _handle_message(handler, request, kind)
                    if handler.request_processed:
                        self._buffered_bytes -= buffered_bytes
                        buffered_bytes = 0
                    for output in outputs:
                        await self._annotate_rejection(
                            handler, output, original_body, observed_bytes, body_complete
                        )
                        phase.observe(output)
                        yield output
                    original_body = b""
                except asyncio.CancelledError:
                    # The peer can no longer receive an immediate response.
                    # Teardown still clears the stream-local reversal state.
                    raise
                except Exception as exc:
                    _record_dispatch_failure(handler, exc)
                    _log_committed_limit(handler, phase, exc)
                    output = await _failure_for_phase(context, phase, kind, exc)
                    await self._annotate_rejection(
                        handler, output, original_body, observed_bytes, body_complete, exc
                    )
                    yield output
                    return
                if outputs and outputs[-1].HasField("immediate_response"):
                    return
            try:
                outputs = _finish_stream(handler)
                for output in outputs:
                    await self._annotate_rejection(handler, output, b"", 0, False)
                    phase.observe(output)
                    yield output
            except Exception as exc:
                _record_dispatch_failure(handler, exc)
                _log_committed_limit(handler, phase, exc)
                output = await _failure_for_phase(context, phase, "response_eof", exc)
                await self._annotate_rejection(handler, output, b"", 0, False, exc)
                yield output
        finally:
            handler.clear_sensitive_state()
            self._buffered_bytes -= buffered_bytes
            active_streams.dec()

    async def _annotate_rejection(
        self,
        handler: StreamHandler,
        response: ext_proc_pb2.ProcessingResponse,
        original_body: bytes,
        observed_bytes: int,
        complete: bool,
        error: Exception | None = None,
    ) -> None:
        if not response.HasField("immediate_response"):
            return
        limit = (
            error.limit if isinstance(error, (EnginePolicyError, InvalidEngineReplyError)) else None
        )
        if (
            handler.destination_policy is not None
            and handler.destination_policy.destination_kind == "model"
        ):
            annotate_rejection(response, handler.correlation_id, limit=limit)
        status = response.immediate_response.status.code
        if status not in {400, 413} or not self._capture.config.enabled:
            return
        declared = handler.request_headers.get("content-length", "")
        declared_bytes = (
            int(declared)
            if declared.isascii() and declared.isdecimal() and len(declared) <= 20
            else None
        )
        reference = await self._capture.capture_rejection(
            original_body,
            correlation_id=handler.correlation_id,
            reason_code=limit.reason if limit is not None else "invalid_request",
            complete=complete,
            declared_bytes=declared_bytes,
            observed_bytes=observed_bytes,
        )
        _logger.warning(
            "rejection capture request_id=%s reference=%s observed_bytes=%d input_complete=%s",
            handler.correlation_id,
            reference or "unavailable",
            observed_bytes,
            complete,
        )


def _log_committed_limit(handler: StreamHandler, phase: StreamPhase, exc: Exception) -> None:
    if (
        phase.response_headers_committed
        and isinstance(exc, (EnginePolicyError, InvalidEngineReplyError))
        and exc.limit is not None
    ):
        log_limit_rejection(handler.correlation_id, exc.limit)


def _overloaded(correlation_id: str | None = None) -> ext_proc_pb2.ProcessingResponse:
    """Reject before forwarding any request or unprocessed response content."""
    errors_total.labels(type="overloaded").inc()
    response = immediate_response(
        503,
        '{"error":{"message":"processor busy","code":"capacity_unavailable","retryable":true}}',
    )
    annotate_rejection(response, correlation_id or uuid4().hex)
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


def _finish_stream(handler: StreamHandler) -> tuple[ext_proc_pb2.ProcessingResponse, ...]:
    """Finalize an incomplete request or flush a normal response iterator EOF."""
    incomplete_request = handler.finish_request()
    if incomplete_request is not None:
        return (incomplete_request,)
    final_body = handler.finish_response()
    if final_body is None:
        return ()
    return _ordered_response(handler, final_body)


async def _handle_message(
    handler: StreamHandler,
    request: ext_proc_pb2.ProcessingRequest,
    kind: str,
) -> tuple[ext_proc_pb2.ProcessingResponse, ...]:
    """Handle one input while preserving full-duplex headers-body-trailers order."""
    handler.validate_destination_policy(request)
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
    if kind == "response_trailers":
        final_body = handler.finish_response()
        if final_body is not None:
            outputs.extend(_ordered_response(handler, final_body))
    response = await handler.handle(request)
    if response is not None:
        outputs.extend(_ordered_response(handler, response))
    return tuple(outputs)


def _failure_response(phase: str, exc: Exception) -> ext_proc_pb2.ProcessingResponse:
    """Record only bounded response failure context and fail the stream closed."""
    if isinstance(exc, McpUnsupportedFeatureError):
        _record_failure(phase, exc)
        return immediate_response(400, exc.body)
    if isinstance(exc, McpHttpError):
        response = immediate_response(exc.status_code, '{"error":"MCP HTTP request failed"}')
        for key, value in exc.headers.items():
            response.immediate_response.headers.set_headers.add(
                header={"key": key, "value": value}, append_action=2
            )
        return response
    _record_failure(phase, exc)
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
    return immediate_response(503, '{"error":"internal processing error"}')


async def _failure_for_phase(
    context: object,
    state: StreamPhase,
    phase: str,
    exc: Exception,
) -> ext_proc_pb2.ProcessingResponse:
    """Return a precommit local reply or abort a response already committed downstream."""
    if state.response_headers_committed:
        await _abort_stream(context, phase, exc)
    return _failure_response(phase, exc)


async def _abort_stream(context: object, phase: str, exc: Exception) -> NoReturn:
    """Abort a response whose headers have already been committed downstream."""
    _record_failure(phase, exc)
    grpc_context = cast(AbortContext, context)
    await grpc_context.abort(grpc.StatusCode.INTERNAL, "response processing failed closed")


def _record_failure(phase: str, exc: Exception) -> None:
    """Record bounded failure metadata without response payloads or exception text."""
    reason = _failure_reason(exc)
    errors_total.labels(type="processing").inc()
    if phase in {"response_headers", "response_body", "response_trailers", "response_eof"}:
        response_failures_total.labels(phase=phase, reason=reason).inc()
    _logger.warning(
        "ext_proc processing failed closed phase=%s reason=%s error=%s",
        phase,
        reason,
        type(exc).__name__,
    )


def _failure_reason(exc: Exception) -> str:
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
