"""Per-stream Envoy ext_proc state and transport dispatch."""

# ruff: noqa: TRY003

from __future__ import annotations

import codecs
import zlib
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import uuid4

from agentgateway_extproc.config.settings import Settings
from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.notice.preferences import (
    DEFAULT_PREFERENCES,
    NoticePreferences,
    NoticePreferencesClient,
)
from agentgateway_extproc.lib.pipeline.mcp import (
    McpHeaderContext,
    McpMessageContext,
    McpProtocolError,
    McpUnsupportedFeatureError,
    validate_mcp_headers,
)
from agentgateway_extproc.lib.pipeline.profiles import header_value
from agentgateway_extproc.lib.pipeline.profiles import request_headers as profile_request_headers
from agentgateway_extproc.lib.pipeline.provider_errors import (
    MAX_PROVIDER_ERROR_BYTES,
    provider_error_response,
)
from agentgateway_extproc.lib.pipeline.reasoning import reasoning_scope
from agentgateway_extproc.lib.pipeline.request import immediate_response, process_request
from agentgateway_extproc.lib.pipeline.response import process_response_chunk
from agentgateway_extproc.metrics import dispatcher_total, response_transport_total
from agentgateway_extproc.models.admission import (
    AdmissionEnvelope,
    RequestContext,
    RequestTextInspection,
    admission_snapshot,
    metadata_from_request,
)
from agentgateway_extproc.models.destination import (
    DESTINATION_POLICY_NAMESPACE,
    DestinationPolicy,
    McpDestinationPolicy,
)
from agentgateway_extproc.models.exceptions import (
    ContextForgeAccountRequiredError,
    EnginePolicyError,
    GatewayPolicyError,
    InvalidEngineReplyError,
    LimitDetail,
    McpHttpError,
    TrustedMetadataError,
)
from agentgateway_extproc.models.types import (
    CONTEXTFORGE_ACCOUNT_HEADER,
    PRESIDIO_RESPONSE_HEADER,
    SENSITIVE_RESPONSE_HEADERS,
    RequestStats,
)

if TYPE_CHECKING:
    from agentgateway_extproc.lib.pipeline.mcp import McpSseResponseProcessor
    from agentgateway_extproc.lib.pipeline.native_response import NativeSseProcessor
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
        docling: DoclingClient | None = None,
        preferences_client: NoticePreferencesClient | None = None,
    ) -> None:
        """Initialize state for one HTTP request/response stream."""
        limits = settings or Settings()
        self.correlation_id = uuid4().hex
        self.compatibility = limits.compatibility
        self.client = client
        self.docling = docling
        self.preferences_client = preferences_client
        self.notice_preferences: NoticePreferences = DEFAULT_PREFERENCES
        self.reversal_map: dict[str, str] = {}
        self.reversal_entity_prefixes: tuple[tuple[str, str], ...] = ()
        self.request_headers: dict[str, str] = {}
        self.request_body_chunks: list[bytes] = []
        self.response_content_type = ""
        self.response_format = "text"
        self.expected_response_format: Literal["json", "sse"] | None = None
        self.response_status = 200
        self.provider_error_body: bytearray | None = None
        self.response_api_kind = "chat"
        self.response_is_gzip = False
        self.gzip_decompressor: GzipDecompressor | None = None
        self.response_body_chunks: list[bytes] = []
        self.response_passthrough_chunks: list[bytes] = []
        self.notice_messages: list[str] = []
        self.safety_notice_messages: list[str] = []
        self.response_notice_allowed = True
        self.response_structured_json = False
        self.presidio_code: str | None = None
        self.request_stats: RequestStats | None = None
        self.restored_counts: dict[str, int] = {}
        self.reversal_misses = 0
        self.sse_processor: SseResponseProcessor | McpSseResponseProcessor | None = None
        self.native_sse_processor: NativeSseProcessor | None = None
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
        self.mcp_empty_eos_pending = False
        self.request_started = False
        self.response_decoder = _utf8_decoder()
        self.pending_response_headers: ext_proc_pb2.ProcessingResponse | None = None
        self.destination_policy: DestinationPolicy | None = None
        self.admission: AdmissionEnvelope | None = None
        self.request_context: RequestContext | None = None
        self._destination_metadata: AdmissionEnvelope | None = None
        self._admission_snapshot: str | None = None
        self.mcp_headers: McpHeaderContext | None = None
        self.mcp_context: McpMessageContext | None = None
        self.request_nonce: bytes | None = None
        self.response_processing_enabled = True
        self.text_pii_enabled = True
        self.reasoning_digests: set[str] = set()
        self._dispatch_outcomes: set[str] = set()

    def clear_sensitive_state(self) -> None:
        """Discard request-scoped plaintext reversal material."""
        self.reversal_map.clear()
        self.reversal_entity_prefixes = ()
        self.request_headers.clear()
        self.request_body_chunks.clear()
        self.notice_messages.clear()
        self.safety_notice_messages.clear()
        self.request_stats = None
        self.restored_counts.clear()
        self.reversal_misses = 0
        self.response_api_kind = "chat"
        self.response_body_chunks.clear()
        if self.provider_error_body is not None:
            self.provider_error_body.clear()
            self.provider_error_body = None
        self.expected_response_format = None
        self.response_passthrough_chunks.clear()
        self.sse_processor = None
        self.native_sse_processor = None
        self.guard_injected = False
        self.gzip_decompressor = None
        self.response_decoder = _utf8_decoder()
        self.destination_policy = None
        self.admission = None
        self.request_context = None
        self._destination_metadata = None
        self._admission_snapshot = None
        self.mcp_headers = None
        self.mcp_context = None
        self.request_nonce = None
        self.reasoning_digests.clear()
        self.response_headers_accepted = False
        self.mcp_empty_eos_pending = False

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
            return await self._request_headers(request)
        if kind == "request_body":
            return await self._request_body(request)
        if kind == "response_headers":
            return self._response_headers(request)
        if kind == "response_body":
            response = self._response_body(request)
            await self._record_reasoning()
            return response
        if kind == "request_trailers":
            if not self.request_processed:
                self.request_processed = True
                return immediate_response(400, '{"error":"request body required"}')
            return immediate_response(400, '{"error":"unexpected request trailers"}')
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
            metadata = metadata_from_request(request)
        except TrustedMetadataError:
            self.record_dispatch("metadata_failure")
            raise
        if self._destination_metadata is not None and metadata != self._destination_metadata:
            self.record_dispatch("metadata_failure")
            raise TrustedMetadataError
        snapshot = admission_snapshot(request)
        if self._admission_snapshot is not None and snapshot != self._admission_snapshot:
            self.record_dispatch("metadata_failure")
            raise TrustedMetadataError
        self._admission_snapshot = snapshot
        if self._destination_metadata is not None:
            return
        self._destination_metadata = metadata
        self.admission = metadata
        self.destination_policy = metadata.destination_policy
        self.correlation_id = metadata.request_context.request_id

    def _is_metadata_eos_compat(self, request: ext_proc_pb2.ProcessingRequest) -> bool:
        """Accept AgentGateway's metadata-less empty final response callback."""
        return bool(
            self.destination_policy is not None
            and self.response_headers_accepted
            and (not self.response_processed or self.mcp_empty_eos_pending)
            and request.WhichOneof("request") == "response_body"
            and request.response_body.end_of_stream
            and not request.response_body.body
            and DESTINATION_POLICY_NAMESPACE not in request.metadata_context.filter_metadata
            and not request.metadata_context.filter_metadata
        )

    async def _request_headers(  # noqa: C901
        self, request: ext_proc_pb2.ProcessingRequest
    ) -> ext_proc_pb2.ProcessingResponse:
        """Remove adapter-owned headers before forwarding the request upstream."""
        header_items = request.request_headers.headers.headers
        if self.admission is None:
            raise TrustedMetadataError
        headers, removed = profile_request_headers(request, self.admission)
        self.request_headers = headers
        self.request_started = True
        policy = self.destination_policy
        if policy is None:
            raise TrustedMetadataError
        if policy.destination_kind == "mcp":
            if "mcp-session-id" in headers:
                self.record_dispatch("protocol_failure")
                return immediate_response(
                    404,
                    '{"error":"Stateful MCP sessions are currently unsupported pending an '
                    'AgentGateway session-ownership fix. Reinitialize without Mcp-Session-Id."}',
                )
            critical = {
                ":method",
                ":path",
                "accept",
                "content-type",
                "content-encoding",
                "mcp-protocol-version",
                "mcp-method",
                "mcp-name",
                "last-event-id",
            }
            names = [item.key.lower() for item in header_items]
            if any(names.count(name) > 1 for name in critical):
                self.record_dispatch("protocol_failure")
                return immediate_response(400, '{"error":"invalid MCP request headers"}')
            try:
                self.mcp_headers = validate_mcp_headers(self.request_headers, policy)
            except McpUnsupportedFeatureError as exc:
                self.record_dispatch("protocol_failure")
                return immediate_response(400, exc.body)
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
                return _request_headers_response(removed, policy=policy)
            if request.request_headers.end_of_stream:
                self.request_processed = True
                self.record_dispatch("protocol_failure")
                return immediate_response(400, '{"error":"MCP request body required"}')
            return _request_headers_response(removed, policy=policy)
        if self.preferences_client is not None:
            self.notice_preferences = await self.preferences_client.get(
                self.admission.request_context
            )
        if request.request_headers.end_of_stream:
            self.request_processed = True
            return immediate_response(400, '{"error":"request body required"}')
        return _request_headers_response(removed)

    async def _request_body(
        self, request: ext_proc_pb2.ProcessingRequest
    ) -> ext_proc_pb2.ProcessingResponse:
        """Analyze the one complete body supplied by buffered request-body mode."""
        if not self.request_started:
            self.request_processed = True
            return immediate_response(400, '{"error":"request headers required"}')
        if self.request_processed:
            return immediate_response(400, '{"error":"unexpected request body"}')
        if not request.request_body.end_of_stream or (
            "content-length" in self.request_headers
            and int(self.request_headers["content-length"]) != len(request.request_body.body)
        ):
            raise GatewayPolicyError("invalid_request")
        size = sum(map(len, self.request_body_chunks)) + len(request.request_body.body)
        if size > self.max_request_bytes:
            raise EnginePolicyError(
                "request_too_large",
                limit=LimitDetail(
                    stage="admission",
                    reason="bytes",
                    measured=size,
                    maximum=self.max_request_bytes,
                    unit="bytes",
                    exact=request.request_body.end_of_stream,
                ),
                correlation_id=self.correlation_id,
            )
        self.request_body_chunks.append(request.request_body.body)
        self.request_processed = True
        response = await process_request(self, self.client)
        if self.request_context is not None and response.HasField("request_body"):
            self.request_context = self.request_context.model_copy(
                update={
                    "request_text_inspection": self._request_text_inspection(
                        self.request_context.pii_policy.enabled
                    )
                }
            )
            response.dynamic_metadata.update(
                {"neurwerk.request_context": self.request_context.model_dump(mode="json")}
            )
        elif (
            self.admission is not None
            and self.admission.request_context.destination_kind == "mcp"
            and response.HasField("request_body")
        ):
            context = self.admission.resolve_mcp()
            response.dynamic_metadata.update(
                {
                    "neurwerk.request_context": {
                        **context.model_dump(mode="json"),
                        "request_text_inspection": self._request_text_inspection(
                            context.pii_policy.enabled
                        ),
                    }
                }
            )
        return response

    def _request_text_inspection(self, enabled: bool) -> RequestTextInspection:
        """Describe validated request analysis, never configured policy alone as a scan."""
        if not enabled:
            return "disabled"
        if self.request_stats is None:
            return "unknown"
        analysis = self.request_stats.analysis
        if analysis.scan_performed:
            return "scanned"
        if analysis.cached_decision_applied:
            return "cached_decision"
        return "not_scanned"

    def _response_headers(  # noqa: C901
        self, request: ext_proc_pb2.ProcessingRequest
    ) -> ext_proc_pb2.ProcessingResponse | None:
        """Capture response format and remove encoding headers when decoding gzip."""
        header_items = request.response_headers.headers.headers
        _validate_response_header_cardinality(
            self.response_api_kind,
            [item.key.lower() for item in header_items],
            strict=self.admission is not None,
        )
        headers = {
            item.key.lower(): header_value(item)
            for item in header_items
            if item.key.lower() not in SENSITIVE_RESPONSE_HEADERS
        }
        self.response_content_type = headers.get("content-type", "")
        self.response_started = True
        try:
            self.response_status = int(headers.get(":status", "200"))
        except ValueError as exc:
            raise ValueError("invalid response status") from exc
        if (
            self.admission is not None
            and self.response_api_kind != "mcp"
            and 400 <= self.response_status <= 599
        ):
            # Only bounded, uncompressed JSON 400s are inspected for fixed diagnostics.
            # Authentication errors are rejected at headers without consuming their bodies.
            if (
                self.response_status == 400
                and not request.response_headers.end_of_stream
                and self.response_content_type.partition(";")[0].strip().lower()
                == "application/json"
                and headers.get("content-encoding", "").strip().lower() in {"", "identity"}
            ):
                self.provider_error_body = bytearray()
                self.response_headers_accepted = True
                return None
            return provider_error_response(self.response_status)
        if not 200 <= self.response_status < 300:
            self.notice_messages.clear()
            self.response_notice_allowed = False
        media_type = self.response_content_type.partition(";")[0].strip().casefold()
        if (
            self.admission is not None
            and self.response_api_kind != "mcp"
            and (
                not 200 <= self.response_status < 300
                or media_type not in {"application/json", "text/event-stream"}
                or (
                    self.expected_response_format is not None
                    and media_type
                    != (
                        "text/event-stream"
                        if self.expected_response_format == "sse"
                        else "application/json"
                    )
                )
            )
        ):
            raise ValueError("unsupported profile response")
        self.response_format = (
            "sse"
            if media_type == "text/event-stream"
            else "json"
            if media_type == "application/json" or media_type.endswith("+json")
            else "text"
        )
        self.response_decoder = _utf8_decoder()
        encoding = headers.get("content-encoding", "").strip().casefold()
        self.response_is_gzip = encoding == "gzip"
        if self.response_api_kind == "mcp":
            self._validate_mcp_response_headers(
                media_type,
                headers,
                end_of_stream=request.response_headers.end_of_stream,
            )
        if encoding not in {"", "identity", "gzip"}:
            raise ValueError("unsupported response content encoding")
        response_transport_total.labels(
            format=self.response_format if self.response_format in {"sse", "json"} else "other",
            encoding="gzip" if self.response_is_gzip else "identity",
        ).inc()
        preserve_mcp_bytes = self._preserve_mcp_response_bytes()
        removed = self._response_removed_headers(headers, preserve=preserve_mcp_bytes)
        if self.admission is not None or self._modern_protocol_error():
            removed.extend(
                item.key.lower()
                for item in header_items
                if item.key.lower() in SENSITIVE_RESPONSE_HEADERS
            )
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
        modern_protocol_error = (
            context.protocol_version == "2026-07-28"
            and self.response_status in {400, 404}
            and media_type == "application/json"
            and not end_of_stream
        )
        if 400 <= self.response_status <= 599 and not modern_protocol_error:
            raise McpHttpError(self.response_status, headers)
        if not modern_protocol_error:
            self._validate_mcp_response_status(context, media_type)
        self._validate_mcp_bodyless_transport(headers)
        bodyless = self.response_status in {202, 204}
        if bodyless:
            # Defer the header acknowledgement until an empty final callback proves no body.
            self.response_format = "json"
        if (
            not bodyless
            and not end_of_stream
            and media_type not in {"application/json", "text/event-stream"}
        ):
            raise McpProtocolError("unsupported MCP response media type")
        if end_of_stream:
            if self.response_is_gzip:
                raise McpProtocolError("empty MCP response cannot be gzip encoded")
            validate_mcp_empty_response(self)
            self.response_processed = True
            self.mcp_empty_eos_pending = bodyless

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
            if self.response_status not in {202, 204}:
                raise McpProtocolError("MCP DELETE requires empty HTTP 202 or 204")
        elif context.notification and self.response_status != 202:
            raise McpProtocolError("MCP notification requires empty HTTP 202")
        elif not context.notification and self.response_status != 200:
            raise McpProtocolError("MCP request requires HTTP 200")

    def _validate_mcp_bodyless_transport(
        self,
        headers: dict[str, str],
    ) -> None:
        if self.response_status not in {202, 204}:
            return
        content_length = headers.get("content-length")
        if self.response_is_gzip:
            raise McpProtocolError("empty MCP response cannot be gzip encoded")
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
            if not (
                self.mcp_empty_eos_pending
                and request.response_body.end_of_stream
                and not request.response_body.body
            ):
                raise McpProtocolError("MCP empty response cannot contain a body")
            self.mcp_empty_eos_pending = False
        if self.response_api_kind == "mcp" and self.response_status in {202, 204}:
            if request.response_body.body:
                raise McpProtocolError("MCP empty response status cannot contain a body")
            if not request.response_body.end_of_stream:
                return None
        return self._process_response_bytes(
            request.response_body.body, request.response_body.end_of_stream
        )

    async def finish_response(self) -> ext_proc_pb2.ProcessingResponse | None:
        """Flush a full-duplex body before acknowledging response trailers."""
        if not self.response_started or self.response_processed:
            return None
        response = self._process_response_bytes(b"", True)
        await self._record_reasoning()
        return response

    async def _record_reasoning(self) -> None:
        if self.response_processed and self.reasoning_digests and self.request_context is not None:
            await self.client.reasoning_receipts(
                reasoning_scope(self.request_context),
                sorted(self.reasoning_digests),
                record=True,
                correlation_id=self.correlation_id,
                policy_version=self.request_stats.analysis.policy_version
                if self.request_stats
                else None,
            )
            self.reasoning_digests.clear()

    def _process_response_bytes(
        self, chunk: bytes, end: bool
    ) -> ext_proc_pb2.ProcessingResponse | None:
        """Apply bounded decoding and buffer only formats requiring whole-body mutation."""
        if self.provider_error_body is not None:
            return self._provider_error_chunk(self.provider_error_body, chunk, end)
        preserve_encoded = self._retain_encoded_mcp_response(chunk)
        self.response_encoded_bytes += len(chunk)
        if self.response_encoded_bytes > self.max_response_bytes:
            raise self._response_limit(
                self.response_encoded_bytes, exact=end, reason="encoded_bytes"
            )
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
            raise self._response_limit(
                self.response_emitted_bytes, exact=end, reason="transformed_bytes"
            )
        if end:
            self.response_processed = True
        return response

    def _provider_error_chunk(
        self, body: bytearray, chunk: bytes, end: bool
    ) -> ext_proc_pb2.ProcessingResponse | None:
        if len(body) + len(chunk) > min(MAX_PROVIDER_ERROR_BYTES, self.max_response_bytes):
            body.clear()
            end = True
        else:
            body.extend(chunk)
        if not end:
            return None
        response = provider_error_response(self.response_status, bytes(body))
        body.clear()
        self.provider_error_body = None
        self.response_processed = True
        return response

    def _response_removed_headers(self, headers: dict[str, str], *, preserve: bool) -> list[str]:
        """Remove stale content controls and strip diagnostics from modern error replies."""
        removed = ([] if preserve else ["content-length", "etag", "content-md5", "digest"]) + [
            PRESIDIO_RESPONSE_HEADER
        ]
        if self._modern_protocol_error():
            removed.extend(key for key in headers if key not in {":status", "content-type"})
        if self.response_is_gzip and not preserve:
            removed.append("content-encoding")
        return removed

    def _modern_protocol_error(self) -> bool:
        """Identify modern negotiation replies that must never retain upstream bytes."""
        return (
            self.mcp_context is not None
            and self.mcp_context.protocol_version == "2026-07-28"
            and self.response_status in {400, 404}
        )

    def _preserve_mcp_response_bytes(self) -> bool:
        policy = self.destination_policy
        return (
            isinstance(policy, McpDestinationPolicy)
            and not policy.pii_enabled
            and not self._modern_protocol_error()
        )

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
            raise self._response_limit(self.response_decoded_bytes + len(decoded), exact=False)
        if decompressor.unused_data:
            raise ValueError("gzip response contains trailing data")
        if end:
            if not decompressor.eof:
                raise ValueError("gzip response is truncated")
            tail = decompressor.flush(remaining - len(decoded) + 1)
            decoded += tail
            if len(decoded) > remaining:
                raise self._response_limit(self.response_decoded_bytes + len(decoded), exact=False)
            if decompressor.unused_data:
                raise ValueError("gzip response contains trailing data")
            self.gzip_decompressor = None
        self._record_decoded_size(len(decoded))
        return decoded

    def _record_decoded_size(self, size: int) -> None:
        self.response_decoded_bytes += size
        if self.response_decoded_bytes > self.max_response_bytes:
            raise self._response_limit(self.response_decoded_bytes, exact=False)

    def _response_limit(
        self,
        measured: int,
        *,
        exact: bool,
        reason: Literal["encoded_bytes", "decoded_bytes", "transformed_bytes"] = "decoded_bytes",
    ) -> InvalidEngineReplyError:
        return InvalidEngineReplyError(
            limit=LimitDetail(
                stage="output" if reason == "transformed_bytes" else "provider_response",
                reason=reason,
                measured=measured,
                maximum=self.max_response_bytes,
                unit="bytes",
                exact=exact,
            )
        )

    def finish_request(self) -> ext_proc_pb2.ProcessingResponse | None:
        """Fail closed when an opened request stream ends before analysis."""
        if not self.request_started:
            return None
        if self.request_processed:
            return None
        self.request_processed = True
        return immediate_response(400, '{"error":"incomplete request body"}')

    def _response_trailers(
        self, request: ext_proc_pb2.ProcessingRequest
    ) -> ext_proc_pb2.ProcessingResponse:
        names = {item.key.casefold() for item in request.response_trailers.trailers.headers}
        if self._modern_protocol_error():
            return ext_proc_pb2.ProcessingResponse(
                response_trailers=ext_proc_pb2.TrailersResponse(
                    header_mutation={"remove_headers": sorted(names)}
                )
            )
        removed = (
            sorted(names & {"etag", "content-md5", "digest"})
            if self.response_processing_enabled and not self._preserve_mcp_response_bytes()
            else []
        )
        if self.admission is not None:
            removed.extend(sorted(names & SENSITIVE_RESPONSE_HEADERS))
        return ext_proc_pb2.ProcessingResponse(
            response_trailers=ext_proc_pb2.TrailersResponse(
                header_mutation={"remove_headers": removed}
            )
        )


def _utf8_decoder() -> codecs.IncrementalDecoder:
    """Create a strict decoder that retains split multibyte code points."""
    return codecs.getincrementaldecoder("utf-8")(errors="strict")


def _request_headers_response(
    removed: list[str],
    *,
    policy: McpDestinationPolicy | None = None,
) -> ext_proc_pb2.ProcessingResponse:
    response = ext_proc_pb2.ProcessingResponse(
        request_headers=ext_proc_pb2.HeadersResponse(
            response=ext_proc_pb2.CommonResponse(header_mutation={"remove_headers": removed})
        )
    )
    if policy is not None and policy.contextforge:
        if policy.account_email is None:
            raise ContextForgeAccountRequiredError
        response.request_headers.response.header_mutation.set_headers.add(
            header={"key": CONTEXTFORGE_ACCOUNT_HEADER, "value": policy.account_email},
            append_action=2,
        )
    return response


def _validate_response_header_cardinality(
    api_kind: str, names: list[str], *, strict: bool = False
) -> None:
    """Reject ambiguous MCP response controls before collapsing header values."""
    if (api_kind == "mcp" or strict) and any(
        names.count(name) > 1
        for name in {
            ":status",
            "content-type",
            "content-encoding",
            "content-length",
            "transfer-encoding",
            "allow",
            "retry-after",
        }
    ):
        raise McpProtocolError("ambiguous MCP response headers")
