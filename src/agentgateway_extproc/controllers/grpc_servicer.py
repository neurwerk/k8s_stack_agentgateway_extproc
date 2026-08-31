"""Envoy ExternalProcessor gRPC controller with fail-closed handling."""

# ruff: noqa: BLE001

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import NoReturn, Protocol, cast, override

import grpc

from agentgateway_extproc.config.settings import Settings
from agentgateway_extproc.gen import ext_proc_pb2, ext_proc_pb2_grpc
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.pipeline.mcp import McpProtocolError
from agentgateway_extproc.lib.pipeline.request import immediate_response
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.metrics import active_streams, errors_total, response_failures_total
from agentgateway_extproc.models.exceptions import (
    EnginePolicyError,
    EngineUnavailableError,
    InvalidEngineReplyError,
    InvalidReversalError,
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
    ) -> None:
        """Store the shared engine client used by all streams."""
        self._client = client
        self._settings = settings or Settings()

    @override
    async def Process(
        self, request_iterator: AsyncIterator[ext_proc_pb2.ProcessingRequest], context: object
    ) -> AsyncIterator[ext_proc_pb2.ProcessingResponse]:
        """Process one Envoy bidirectional stream."""
        active_streams.inc()
        handler = StreamHandler(self._client, self._settings)
        phase = StreamPhase()
        try:
            async for request in request_iterator:
                kind = request.WhichOneof("request") or "unknown"
                try:
                    outputs = await _handle_message(handler, request, kind)
                    for output in outputs:
                        phase.observe(output)
                        yield output
                except asyncio.CancelledError:
                    # The peer can no longer receive an immediate response.
                    # Teardown still clears the stream-local reversal state.
                    raise
                except Exception as exc:
                    _record_dispatch_failure(handler, exc)
                    yield await _failure_for_phase(context, phase, kind, exc)
                    return
                if outputs and outputs[-1].HasField("immediate_response"):
                    return
            try:
                outputs = _finish_stream(handler)
                for output in outputs:
                    phase.observe(output)
                    yield output
            except Exception as exc:
                _record_dispatch_failure(handler, exc)
                yield await _failure_for_phase(context, phase, "response_eof", exc)
        finally:
            handler.clear_sensitive_state()
            active_streams.dec()


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
    if isinstance(exc, McpProtocolError):
        handler.record_dispatch("protocol_failure")
    elif (
        isinstance(exc, (EngineUnavailableError, InvalidEngineReplyError))
        or handler.response_api_kind == "mcp"
    ):
        handler.record_dispatch("transport_failure")
