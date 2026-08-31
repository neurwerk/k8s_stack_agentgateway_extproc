"""Per-stream Envoy ext_proc state and transport dispatch."""

# ruff: noqa: TRY003

from __future__ import annotations

import codecs
import zlib
from typing import TYPE_CHECKING, Protocol

from agentgateway_extproc.config.settings import Settings
from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.pipeline.mcp import (
    McpHeaderContext,
    McpMessageContext,
    McpProtocolError,
    validate_mcp_headers,
)
from agentgateway_extproc.lib.pipeline.request import immediate_response, process_request
from agentgateway_extproc.lib.pipeline.response import process_response_chunk
from agentgateway_extproc.metrics import dispatcher_total, response_transport_total
from agentgateway_extproc.models.destination import (
    DESTINATION_POLICY_NAMESPACE,
    DestinationPolicy,
    McpDestinationPolicy,
    ModelDestinationPolicy,
    destination_policy_from_request,
)
from agentgateway_extproc.models.exceptions import TrustedMetadataError
from agentgateway_extproc.models.types import (
    PRESIDIO_RESPONSE_HEADER,
    REQUEST_HEADERS,
    RequestStats,
)

if TYPE_CHECKING:
    from agentgateway_extproc.lib.masking.reversal import PlaceholderStreamRewriter
    from agentgateway_extproc.lib.pipeline.guard import GuardStreamStripper
    from agentgateway_extproc.lib.pipeline.mcp import McpSseResponseProcessor
    from agentgateway_extproc.lib.pipeline.response import SseResponseProcessor


class GzipDecompressor(Protocol):
    """Describe the incremental gzip decompressor methods used by the adapter."""

    @property
    def unconsumed_tail(self) -> bytes:
        """Return compressed input not consumed because the output limit was reached."""

    @property
    def unused_data(self) -> bytes:
        """Return bytes trailing the completed gzip member."""

    @property
    def eof(self) -> bool:
        """Return whether the gzip member reached its end marker."""

    def decompress(self, data: bytes, max_length: int, /) -> bytes:
        """Decompress one chunk."""

    def flush(self, length: int, /) -> bytes:
        """Flush buffered decompressed data."""


class StreamHandler:
    """Keep reversal and chunk state isolated to one ext_proc stream."""

    def __init__(
        self,
        client: EngineClient,
        settings: Settings | None = None,
    ) -> None:
        """Initialize state for one HTTP request/response stream."""
        limits = settings or Settings()
        self.client = client
        self.reversal_map: dict[str, str] = {}
        self.reversal_entity_prefixes: tuple[tuple[str, str], ...] = ()
        self.request_headers: dict[str, str] = {}
        self.request_body_chunks: list[bytes] = []
        self.response_buffer = ""
        self.response_content_type = ""
        self.response_format = "text"
        self.response_status = 200
        self.response_api_kind = "chat"
        self.response_is_gzip = False
        self.gzip_decompressor: GzipDecompressor | None = None
        self.response_body_chunks: list[bytes] = []
        self.response_passthrough_chunks: list[bytes] = []
        self.notice_messages: list[str] = []
        self.response_notice_allowed = True
        self.response_structured_json = False
        self.presidio_code: str | None = None
        self.request_stats: RequestStats | None = None
        self.restored_counts: dict[str, int] = {}
        self.reversal_misses = 0
        self.sse_processor: SseResponseProcessor | McpSseResponseProcessor | None = None
        self.plain_rewriter: PlaceholderStreamRewriter | None = None
        self.plain_guard_stripper: GuardStreamStripper | None = None
        self.guard_injected = False
        self.request_processed = False
        self.max_request_bytes = limits.max_request_bytes
        self.max_response_bytes = limits.max_response_bytes
        self.max_transformed_request_bytes = limits.max_transformed_request_bytes
        self.response_encoded_bytes = 0
        self.response_decoded_bytes = 0
        self.response_emitted_bytes = 0
        self.response_processed = False
        self.response_started = False
        self.response_headers_accepted = False
        self.request_started = False
        self.request_trailers_expected = False
        self.response_decoder = _utf8_decoder()
        self.pending_response_headers: ext_proc_pb2.ProcessingResponse | None = None
        self.destination_policy: DestinationPolicy | None = None
        self.mcp_headers: McpHeaderContext | None = None
        self.mcp_context: McpMessageContext | None = None
        self.request_nonce: bytes | None = None
        self.response_processing_enabled = True
        self._dispatch_outcomes: set[str] = set()

    def clear_sensitive_state(self) -> None:
        """Discard request-scoped plaintext reversal material."""
        self.reversal_map.clear()
        self.reversal_entity_prefixes = ()
        self.request_headers.clear()
        self.request_body_chunks.clear()
        self.notice_messages.clear()
        self.request_stats = None
        self.restored_counts.clear()
        self.reversal_misses = 0
        self.response_buffer = ""
        self.response_api_kind = "chat"
        self.response_body_chunks.clear()
        self.response_passthrough_chunks.clear()
        self.sse_processor = None
        self.plain_rewriter = None
        self.plain_guard_stripper = None
        self.guard_injected = False
        self.gzip_decompressor = None
        self.response_decoder = _utf8_decoder()
        self.destination_policy = None
        self.mcp_headers = None
        self.mcp_context = None
        self.request_nonce = None
        self.request_trailers_expected = False
        self.response_headers_accepted = False

    def record_dispatch(self, outcome: str) -> None:
        """Record each bounded dispatcher outcome at most once per stream."""
        if outcome not in self._dispatch_outcomes:
            dispatcher_total.labels(outcome=outcome).inc()
            self._dispatch_outcomes.add(outcome)

    async def handle(
        self, request: ext_proc_pb2.ProcessingRequest
    ) -> ext_proc_pb2.ProcessingResponse | None:
        """Dispatch one Envoy processing message."""
        self.validate_destination_policy(request)
        kind = request.WhichOneof("request")
        if kind == "request_headers":
            return self._request_headers(request)
        if kind == "request_body":
            return await self._request_body(request)
        if kind == "response_headers":
            return self._response_headers(request)
        if kind == "response_body":
            return self._response_body(request)
        if kind == "request_trailers":
            if not self.request_processed:
                self.request_processed = True
                return immediate_response(400, '{"error":"request body required"}')
            if not self.request_trailers_expected:
                return immediate_response(400, '{"error":"unexpected request trailers"}')
            self.request_trailers_expected = False
            return ext_proc_pb2.ProcessingResponse(request_trailers=ext_proc_pb2.TrailersResponse())
        if kind == "response_trailers":
            return self._response_trailers(request)
        return ext_proc_pb2.ProcessingResponse(
            immediate_response=ext_proc_pb2.ImmediateResponse(
                status={"code": 500}, body='{"error":"unsupported ext_proc message"}'
            )
        )

    def validate_destination_policy(self, request: ext_proc_pb2.ProcessingRequest) -> None:
        """Require one complete identical trusted policy on every stream message."""
        if self._is_metadata_eos_compat(request):
            self.record_dispatch("metadata_eos_compat")
            return
        try:
            policy = destination_policy_from_request(request)
        except TrustedMetadataError:
            self.record_dispatch("metadata_failure")
            raise
        if self.destination_policy is not None and policy != self.destination_policy:
            self.record_dispatch("metadata_failure")
            raise TrustedMetadataError
        self.destination_policy = policy

    def _is_metadata_eos_compat(self, request: ext_proc_pb2.ProcessingRequest) -> bool:
        """Accept AgentGateway's metadata-less empty final response callback."""
        return bool(
            self.destination_policy is not None
            and self.response_headers_accepted
            and not self.response_processed
            and request.WhichOneof("request") == "response_body"
            and request.response_body.end_of_stream
            and not request.response_body.body
            and DESTINATION_POLICY_NAMESPACE not in request.metadata_context.filter_metadata
        )

    def _request_headers(
        self, request: ext_proc_pb2.ProcessingRequest
    ) -> ext_proc_pb2.ProcessingResponse:
        """Remove adapter-owned headers before forwarding the request upstream."""
        header_items = request.request_headers.headers.headers
        headers = {
            item.key.lower(): (
                item.raw_value.decode(errors="replace") if item.raw_value else item.value
            )
            for item in header_items
        }
        removed = [key for key in headers if key.lower() in REQUEST_HEADERS]
        self.request_headers = {
            key.lower(): value
            for key, value in headers.items()
            if key.lower() not in REQUEST_HEADERS
        }
        self.request_started = True
        policy = self.destination_policy
        if policy is None:
            raise TrustedMetadataError
        if policy.destination_kind == "mcp":
            critical = {
                ":method",
                ":path",
                "accept",
                "content-type",
                "content-encoding",
                "mcp-protocol-version",
                "mcp-session-id",
                "last-event-id",
            }
            names = [item.key.lower() for item in header_items]
            if any(names.count(name) > 1 for name in critical):
                self.record_dispatch("protocol_failure")
                return immediate_response(400, '{"error":"invalid MCP request headers"}')
            try:
                self.mcp_headers = validate_mcp_headers(self.request_headers, policy)
            except McpProtocolError:
                self.record_dispatch("protocol_failure")
                return immediate_response(400, '{"error":"invalid MCP request headers"}')
            if self.mcp_headers.method in {"GET", "DELETE"}:
                if not request.request_headers.end_of_stream:
                    self.record_dispatch("protocol_failure")
                    return immediate_response(400, '{"error":"MCP request body is not allowed"}')
                self.request_processed = True
                self.mcp_context = McpMessageContext(
                    method=f"http/{self.mcp_headers.method.casefold()}",
                    request_id=None,
                    notification=True,
                    engine_request=None,
                    has_text_arguments=False,
                )
                self.response_api_kind = "mcp"
                self.record_dispatch("mcp_lifecycle_pass")
                return _request_headers_response(removed)
            if request.request_headers.end_of_stream:
                self.request_processed = True
                self.record_dispatch("protocol_failure")
                return immediate_response(400, '{"error":"MCP request body required"}')
            return _request_headers_response(removed)
        if request.request_headers.end_of_stream:
            self.request_processed = True
            return immediate_response(400, '{"error":"request body required"}')
        disable_response = isinstance(policy, ModelDestinationPolicy) and not any(
            policy.models.values()
        )
        return _request_headers_response(removed, disable_response=disable_response)

    async def _request_body(
        self, request: ext_proc_pb2.ProcessingRequest
    ) -> ext_proc_pb2.ProcessingResponse:
        """Analyze the one complete body supplied by buffered request-body mode."""
        if not self.request_started:
            self.request_processed = True
            return immediate_response(400, '{"error":"request headers required"}')
        if self.request_processed:
            return immediate_response(400, '{"error":"unexpected request body"}')
        size = sum(map(len, self.request_body_chunks)) + len(request.request_body.body)
        if size > self.max_request_bytes:
            return ext_proc_pb2.ProcessingResponse(
                immediate_response=ext_proc_pb2.ImmediateResponse(
                    status={"code": 413}, body='{"error":"request body too large"}'
                )
            )
        self.request_body_chunks.append(request.request_body.body)
        self.request_trailers_expected = not request.request_body.end_of_stream
        self.request_processed = True
        return await process_request(self, self.client)

    def _response_headers(
        self, request: ext_proc_pb2.ProcessingRequest
    ) -> ext_proc_pb2.ProcessingResponse | None:
        """Capture response format and remove encoding headers when decoding gzip."""
        if not self.response_processing_enabled:
            self.response_started = True
            self.response_headers_accepted = True
            self.response_processed = request.response_headers.end_of_stream
            return ext_proc_pb2.ProcessingResponse(
                response_headers=ext_proc_pb2.HeadersResponse(
                    response=ext_proc_pb2.CommonResponse()
                )
            )
        header_items = request.response_headers.headers.headers
        _validate_response_header_cardinality(
            self.response_api_kind,
            [item.key.lower() for item in header_items],
        )
        headers = {
            item.key.lower(): (
                item.raw_value.decode(errors="replace") if item.raw_value else item.value
            )
            for item in header_items
        }
        self.response_content_type = headers.get("content-type", "")
        self.response_started = True
        try:
            self.response_status = int(headers.get(":status", "200"))
        except ValueError as exc:
            raise ValueError("invalid response status") from exc
        if not 200 <= self.response_status < 300:
            self.notice_messages.clear()
            self.response_notice_allowed = False
        media_type = self.response_content_type.partition(";")[0].strip().casefold()
        self.response_format = (
            "sse"
            if media_type == "text/event-stream"
            else "json"
            if media_type == "application/json" or media_type.endswith("+json")
            else "text"
        )
        self.response_decoder = _utf8_decoder()
        encoding = headers.get("content-encoding", "").strip().casefold()
        if encoding not in {"", "identity", "gzip"}:
            raise ValueError("unsupported response content encoding")
        self.response_is_gzip = encoding == "gzip"
        if self.response_api_kind == "mcp":
            self._validate_mcp_response_headers(
                media_type,
                headers,
                end_of_stream=request.response_headers.end_of_stream,
            )
        response_transport_total.labels(
            format=self.response_format if self.response_format in {"sse", "json"} else "other",
            encoding="gzip" if self.response_is_gzip else "identity",
        ).inc()
        preserve_mcp_bytes = self._preserve_mcp_response_bytes()
        removed = (
            [] if preserve_mcp_bytes else ["content-length", "etag", "content-md5", "digest"]
        ) + [PRESIDIO_RESPONSE_HEADER]
        if self.response_is_gzip and not preserve_mcp_bytes:
            removed.append("content-encoding")
        if self.response_is_gzip:
            self.gzip_decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        set_headers = []
        if 200 <= self.response_status < 300 and self.presidio_code is not None:
            set_headers.append(
                {
                    "header": {
                        "key": PRESIDIO_RESPONSE_HEADER,
                        "value": self.presidio_code,
                    },
                    "append_action": 2,
                }
            )
        response = ext_proc_pb2.ProcessingResponse(
            response_headers=ext_proc_pb2.HeadersResponse(
                response=ext_proc_pb2.CommonResponse(
                    header_mutation={
                        "remove_headers": removed,
                        "set_headers": set_headers,
                    }
                )
            )
        )
        if self.response_format == "json" and not request.response_headers.end_of_stream:
            self.response_headers_accepted = True
            self.pending_response_headers = response
            return None
        self.response_headers_accepted = True
        return response

    def _validate_mcp_response_headers(
        self,
        media_type: str,
        headers: dict[str, str],
        *,
        end_of_stream: bool,
    ) -> None:
        """Validate MCP status/body semantics before response headers are committed."""
        from agentgateway_extproc.lib.pipeline.mcp import (
            McpProtocolError,
            validate_mcp_empty_response,
        )

        context = self.mcp_context
        if context is None:
            raise McpProtocolError("MCP response has no request context")
        self._validate_mcp_response_status(context, media_type)
        self._validate_mcp_bodyless_transport(headers, end_of_stream=end_of_stream)
        if not end_of_stream and media_type not in {"application/json", "text/event-stream"}:
            raise McpProtocolError("unsupported MCP response media type")
        if end_of_stream:
            if self.response_is_gzip:
                raise McpProtocolError("empty MCP response cannot be gzip encoded")
            validate_mcp_empty_response(self)
            self.response_processed = True

    def _validate_mcp_response_status(
        self,
        context: McpMessageContext,
        media_type: str,
    ) -> None:
        if context.method is None:
            if self.response_status != 202:
                raise McpProtocolError("MCP client response requires empty HTTP 202")
        elif context.method == "http/get":
            if self.response_status != 200 or media_type != "text/event-stream":
                raise McpProtocolError("invalid MCP GET response")
        elif context.method == "http/delete":
            if self.response_status != 204:
                raise McpProtocolError("MCP DELETE requires empty HTTP 204")
        elif context.notification and self.response_status != 202:
            raise McpProtocolError("MCP notification requires empty HTTP 202")
        elif not context.notification and self.response_status != 200:
            raise McpProtocolError("MCP request requires HTTP 200")

    def _validate_mcp_bodyless_transport(
        self,
        headers: dict[str, str],
        *,
        end_of_stream: bool,
    ) -> None:
        if self.response_status not in {202, 204}:
            return
        content_length = headers.get("content-length")
        if not end_of_stream:
            raise McpProtocolError("MCP empty response status cannot contain a body")
        if content_length is not None:
            content_length = content_length.strip()
            if (
                not content_length.isascii()
                or not content_length.isdecimal()
                or int(content_length) != 0
            ):
                raise McpProtocolError("MCP empty response status cannot contain a body")
        if "transfer-encoding" in headers:
            raise McpProtocolError("MCP empty response status cannot contain a body")

    def pop_pending_response_headers(self) -> ext_proc_pb2.ProcessingResponse | None:
        """Return a deferred JSON header acknowledgement exactly once."""
        response = self.pending_response_headers
        self.pending_response_headers = None
        return response

    def _response_body(
        self, request: ext_proc_pb2.ProcessingRequest
    ) -> ext_proc_pb2.ProcessingResponse | None:
        """Decode gzip as needed and process arbitrary response chunks."""
        if self.response_api_kind == "mcp" and self.response_processed:
            raise McpProtocolError("MCP empty response cannot contain a body")
        if not self.response_processing_enabled:
            self.response_processed = request.response_body.end_of_stream
            return ext_proc_pb2.ProcessingResponse(
                response_body=ext_proc_pb2.BodyResponse(response=ext_proc_pb2.CommonResponse())
            )
        return self._process_response_bytes(
            request.response_body.body, request.response_body.end_of_stream
        )

    def finish_response(self) -> ext_proc_pb2.ProcessingResponse | None:
        """Flush a full-duplex body before acknowledging response trailers."""
        if not self.response_started or self.response_processed:
            return None
        return self._process_response_bytes(b"", True)

    def _process_response_bytes(
        self, chunk: bytes, end: bool
    ) -> ext_proc_pb2.ProcessingResponse | None:
        """Apply bounded decoding and buffer only formats requiring whole-body mutation."""
        preserve_encoded = self._retain_encoded_mcp_response(chunk)
        self.response_encoded_bytes += len(chunk)
        if self.response_encoded_bytes > self.max_response_bytes:
            raise ValueError("response body too large")
        if self.response_is_gzip:
            chunk = self._decompress_bounded(chunk, end)
        else:
            self._record_decoded_size(len(chunk))
        buffer_complete_body = self.response_format == "json"
        if buffer_complete_body:
            self.response_body_chunks.append(chunk)
            if not end:
                return None
            chunk = b"".join(self.response_body_chunks)
            self.response_body_chunks.clear()
        try:
            decoded = self.response_decoder.decode(chunk, final=end)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid response encoding") from exc
        response = process_response_chunk(self, decoded, end)
        self._restore_encoded_mcp_response(response, preserve=preserve_encoded, end=end)
        emitted = response.response_body.response.body_mutation.streamed_response.body
        self.response_emitted_bytes += len(emitted)
        if self.response_emitted_bytes > self.max_response_bytes:
            raise ValueError("transformed response body too large")
        if end:
            self.response_processed = True
        return response

    def _preserve_mcp_response_bytes(self) -> bool:
        policy = self.destination_policy
        return isinstance(policy, McpDestinationPolicy) and not policy.pii_enabled

    def _retain_encoded_mcp_response(self, chunk: bytes) -> bool:
        preserve = bool(self.response_is_gzip) and self._preserve_mcp_response_bytes()
        if preserve:
            self.response_passthrough_chunks.append(chunk)
        return preserve

    def _restore_encoded_mcp_response(
        self,
        response: ext_proc_pb2.ProcessingResponse,
        *,
        preserve: bool,
        end: bool,
    ) -> None:
        if not preserve:
            return
        emitted = b"" if not end else b"".join(self.response_passthrough_chunks)
        response.response_body.response.body_mutation.streamed_response.body = emitted
        if end:
            self.response_passthrough_chunks.clear()

    def _decompress_bounded(self, chunk: bytes, end: bool) -> bytes:
        """Incrementally decompress without allowing a gzip payload to allocate past its limit."""
        decompressor = self.gzip_decompressor
        if decompressor is None:
            raise ValueError("gzip response decoder is unavailable")
        remaining = self.max_response_bytes - self.response_decoded_bytes
        decoded = decompressor.decompress(chunk, remaining + 1)
        if len(decoded) > remaining or decompressor.unconsumed_tail:
            raise ValueError("response body too large")
        if decompressor.unused_data:
            raise ValueError("gzip response contains trailing data")
        if end:
            if not decompressor.eof:
                raise ValueError("gzip response is truncated")
            tail = decompressor.flush(remaining - len(decoded) + 1)
            decoded += tail
            if len(decoded) > remaining or decompressor.unused_data:
                raise ValueError("response body too large")
            self.gzip_decompressor = None
        self._record_decoded_size(len(decoded))
        return decoded

    def _record_decoded_size(self, size: int) -> None:
        self.response_decoded_bytes += size
        if self.response_decoded_bytes > self.max_response_bytes:
            raise ValueError("response body too large")

    def finish_request(self) -> ext_proc_pb2.ProcessingResponse | None:
        """Fail closed when an opened request stream ends before analysis."""
        if not self.request_started:
            return None
        if self.request_trailers_expected:
            self.request_trailers_expected = False
            return immediate_response(400, '{"error":"incomplete request trailers"}')
        if self.request_processed:
            return None
        self.request_processed = True
        return immediate_response(400, '{"error":"incomplete request body"}')

    def _response_trailers(
        self, request: ext_proc_pb2.ProcessingRequest
    ) -> ext_proc_pb2.ProcessingResponse:
        names = {item.key.casefold() for item in request.response_trailers.trailers.headers}
        removed = (
            sorted(names & {"etag", "content-md5", "digest"})
            if self.response_processing_enabled and not self._preserve_mcp_response_bytes()
            else []
        )
        return ext_proc_pb2.ProcessingResponse(
            response_trailers=ext_proc_pb2.TrailersResponse(
                header_mutation={"remove_headers": removed}
            )
        )


def _utf8_decoder() -> codecs.IncrementalDecoder:
    """Create a strict decoder that retains split multibyte code points."""
    return codecs.getincrementaldecoder("utf-8")(errors="strict")


def _request_headers_response(
    removed: list[str], *, disable_response: bool = False
) -> ext_proc_pb2.ProcessingResponse:
    response = ext_proc_pb2.ProcessingResponse(
        request_headers=ext_proc_pb2.HeadersResponse(
            response=ext_proc_pb2.CommonResponse(header_mutation={"remove_headers": removed})
        )
    )
    if disable_response:
        response.mode_override.CopyFrom(
            ext_proc_pb2.ProcessingMode(
                response_header_mode=ext_proc_pb2.ProcessingMode.SKIP,
                response_body_mode=ext_proc_pb2.ProcessingMode.NONE,
                response_trailer_mode=ext_proc_pb2.ProcessingMode.SKIP,
            )
        )
    return response


def _validate_response_header_cardinality(api_kind: str, names: list[str]) -> None:
    """Reject ambiguous MCP response controls before collapsing header values."""
    if api_kind == "mcp" and any(
        names.count(name) > 1
        for name in {
            ":status",
            "content-type",
            "content-encoding",
            "content-length",
            "transfer-encoding",
        }
    ):
        raise McpProtocolError("ambiguous MCP response headers")
