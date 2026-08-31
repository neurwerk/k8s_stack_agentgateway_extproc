"""Strict MCP Streamable HTTP and JSON-RPC request contracts."""

# ruff: noqa: ANN401, C901, TRY003

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentgateway_extproc.lib.masking.reversal import reverse_placeholders
from agentgateway_extproc.lib.pipeline.sse import SseDecoder
from agentgateway_extproc.models.destination import McpDestinationPolicy
from agentgateway_extproc.models.engine import EngineMcpRequest, JsonValue
from agentgateway_extproc.models.exceptions import InvalidReversalError
from agentgateway_extproc.models.types import RESERVED_PLACEHOLDER_PREFIX_RE

if TYPE_CHECKING:
    from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler

MCP_PROTOCOL_VERSION = "2025-11-25"
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_METHOD_RE = re.compile(r"^[A-Za-z0-9_./:-]{1,256}$")
_SESSION_RE = re.compile(r"^[\x21-\x7e]{1,256}$")
_QUALITY_RE = re.compile(r"^(?:0(?:\.\d{0,3})?|1(?:\.0{0,3})?)$")
_MAX_SAFE_INTEGER = 2**53 - 1
_MAX_MCP_COLLECTION_ENTRIES = 256


class McpProtocolError(ValueError):
    """Indicate a bounded caller or upstream MCP protocol violation."""


@dataclass(frozen=True)
class McpHeaderContext:
    """Retain validated HTTP controls needed after body parsing."""

    method: str
    protocol_version: str | None
    session_id: str | None


@dataclass(frozen=True)
class McpMessageContext:
    """Retain one validated MCP request's response correlation controls."""

    method: str | None
    request_id: str | int | None
    notification: bool
    engine_request: EngineMcpRequest | None
    has_text_arguments: bool


def validate_mcp_headers(headers: dict[str, str], policy: McpDestinationPolicy) -> McpHeaderContext:
    """Validate exact route, method, version, media, and session controls."""
    method = headers.get(":method", "").upper()
    if method not in {"POST", "GET", "DELETE"}:
        raise McpProtocolError("unsupported MCP HTTP method")
    if headers.get(":path") != f"/mcp/{policy.destination_id}":
        raise McpProtocolError("MCP route does not match trusted destination")
    version = headers.get("mcp-protocol-version")
    if version is not None and version != MCP_PROTOCOL_VERSION:
        raise McpProtocolError("unsupported MCP protocol version")
    session_id = headers.get("mcp-session-id")
    if session_id is not None and _SESSION_RE.fullmatch(session_id) is None:
        raise McpProtocolError("invalid MCP session identity")
    if "last-event-id" in headers:
        raise McpProtocolError("resumable MCP streams are unsupported")
    if method == "POST":
        _validate_json_content_type(headers.get("content-type", ""))
        if headers.get("content-encoding", "identity").strip().casefold() not in {
            "",
            "identity",
        }:
            raise McpProtocolError("compressed MCP requests are unsupported")
        accept = headers.get("accept", "")
        if not all(
            _accepts_media_type(accept, media_type)
            for media_type in ("application/json", "text/event-stream")
        ):
            raise McpProtocolError("MCP POST must accept JSON and SSE")
    elif method == "GET":
        if version != MCP_PROTOCOL_VERSION:
            raise McpProtocolError("MCP protocol version is required")
        if not _accepts_media_type(headers.get("accept", ""), "text/event-stream"):
            raise McpProtocolError("MCP GET must accept SSE")
    elif version != MCP_PROTOCOL_VERSION:
        raise McpProtocolError("MCP protocol version is required")
    return McpHeaderContext(method=method, protocol_version=version, session_id=session_id)


def parse_mcp_message(body: bytes, headers: McpHeaderContext) -> McpMessageContext:
    """Parse one strict, bounded MCP JSON-RPC request without coercion."""
    if headers.method != "POST":
        raise McpProtocolError("MCP request body is not allowed for this method")
    try:
        payload = strict_json_loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise McpProtocolError("invalid MCP request JSON") from exc
    if not isinstance(payload, dict):
        raise McpProtocolError("MCP batches are unsupported")
    _validate_json_bounds(payload)
    if "method" not in payload:
        return _parse_client_response(payload)
    allowed = {"jsonrpc", "id", "method", "params"}
    if set(payload) - allowed or payload.get("jsonrpc") != "2.0":
        raise McpProtocolError("invalid MCP JSON-RPC envelope")
    method = payload.get("method")
    if not isinstance(method, str) or _METHOD_RE.fullmatch(method) is None:
        raise McpProtocolError("invalid MCP method")
    if method.startswith("tasks/") or method.startswith("notifications/tasks/"):
        raise McpProtocolError("MCP task extensions are unsupported")
    request_id = _request_id(payload)
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise McpProtocolError("MCP params must be an object")
    _reject_task_metadata(params)
    if method == "initialize":
        if request_id is None or params.get("protocolVersion") != MCP_PROTOCOL_VERSION:
            raise McpProtocolError("unsupported MCP initialization")
    elif headers.protocol_version != MCP_PROTOCOL_VERSION:
        raise McpProtocolError("MCP protocol version is required")
    if method != "tools/call":
        return McpMessageContext(
            method=method,
            request_id=request_id,
            notification="id" not in payload,
            engine_request=None,
            has_text_arguments=False,
        )
    if request_id is None or set(params) - {"name", "arguments", "_meta"}:
        raise McpProtocolError("invalid MCP tools/call controls")
    name = params.get("name")
    if not isinstance(name, str) or _TOOL_NAME_RE.fullmatch(name) is None:
        raise McpProtocolError("invalid MCP tool name")
    arguments = params.get("arguments")
    if "arguments" in params and not isinstance(arguments, dict):
        raise McpProtocolError("MCP tool arguments must be an object")
    engine_payload: dict[str, object] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name},
    }
    if arguments is not None:
        engine_payload["params"]["arguments"] = arguments  # type: ignore[index]
    if "_meta" in params:
        meta = params["_meta"]
        if not isinstance(meta, dict):
            raise McpProtocolError("MCP _meta must be an object")
        engine_payload["params"]["_meta"] = meta  # type: ignore[index]
    try:
        engine_request = EngineMcpRequest.model_validate(engine_payload, strict=True)
    except (TypeError, ValueError) as exc:
        raise McpProtocolError("invalid MCP tools/call request") from exc
    return McpMessageContext(
        method=method,
        request_id=request_id,
        notification=False,
        engine_request=engine_request,
        has_text_arguments=_contains_string(arguments),
    )


def _parse_client_response(payload: dict[str, Any]) -> McpMessageContext:
    """Validate one response to a server-initiated request without retaining its content."""
    if "id" not in payload:
        raise McpProtocolError("MCP client response requires an ID")
    request_id = _request_id(payload)
    _validate_response_envelope(payload, request_id)
    return McpMessageContext(
        method=None,
        request_id=request_id,
        notification=False,
        engine_request=None,
        has_text_arguments=False,
    )


def strict_json_loads(value: str | bytes) -> Any:
    """Decode strict JSON while rejecting duplicate names and non-finite numbers."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("JSON object contains duplicate keys")
            result[key] = item
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("JSON contains a non-finite number")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("JSON contains a non-finite number")
        return parsed

    return json.loads(
        value,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )


def _request_id(payload: dict[str, Any]) -> str | int | None:
    if "id" not in payload:
        return None
    value = payload["id"]
    if isinstance(value, bool):
        raise McpProtocolError("invalid MCP request ID")
    if isinstance(value, str) and 0 < len(value) <= 256:
        return value
    if isinstance(value, int) and -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
        return value
    raise McpProtocolError("invalid MCP request ID")


def _validate_json_content_type(value: str) -> None:
    parts = [part.strip().casefold() for part in value.split(";")]
    if not parts or parts[0] != "application/json":
        raise McpProtocolError("MCP POST requires application/json")
    parameters = parts[1:]
    if any(part != "charset=utf-8" for part in parameters) or len(parameters) > 1:
        raise McpProtocolError("MCP JSON must use UTF-8")


def _accepts_media_type(value: str, target: str) -> bool:
    target_type, target_subtype = target.split("/", 1)
    matches: list[tuple[int, int]] = []
    for item in value.split(","):
        if not item.strip():
            continue
        parts = [part.strip() for part in item.split(";")]
        media_range = parts[0].casefold()
        try:
            media_type, media_subtype = media_range.split("/", 1)
        except ValueError as exc:
            raise McpProtocolError("invalid MCP Accept header") from exc
        if not media_type or not media_subtype:
            raise McpProtocolError("invalid MCP Accept header")
        quality = 1_000
        quality_seen = False
        for parameter in parts[1:]:
            name, separator, parameter_value = parameter.partition("=")
            if not separator or not name.strip() or not parameter_value.strip():
                raise McpProtocolError("invalid MCP Accept header")
            if name.strip().casefold() != "q":
                continue
            if quality_seen:
                raise McpProtocolError("invalid MCP Accept header")
            quality_seen = True
            quality = _parse_quality(parameter_value.strip())
        specificity = (
            2
            if (media_type, media_subtype) == (target_type, target_subtype)
            else 1
            if media_type == target_type and media_subtype == "*"
            else 0
            if (media_type, media_subtype) == ("*", "*")
            else -1
        )
        if specificity >= 0:
            matches.append((specificity, quality))
    if not matches:
        return False
    specificity = max(item[0] for item in matches)
    return all(
        quality > 0 for item_specificity, quality in matches if item_specificity == specificity
    )


def _parse_quality(value: str) -> int:
    if _QUALITY_RE.fullmatch(value) is None:
        raise McpProtocolError("invalid MCP Accept quality")
    whole, separator, fraction = value.partition(".")
    if whole == "1":
        return 1_000
    return int((fraction if separator else "").ljust(3, "0") or "0")


def _validate_json_bounds(value: JsonValue) -> None:
    pending: list[JsonValue] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if len(item) > _MAX_MCP_COLLECTION_ENTRIES:
                raise McpProtocolError("MCP JSON collection exceeds entry limit")
            pending.extend(item.values())
        elif isinstance(item, list):
            if len(item) > _MAX_MCP_COLLECTION_ENTRIES:
                raise McpProtocolError("MCP JSON collection exceeds entry limit")
            pending.extend(item)


def _reject_task_metadata(params: dict[str, JsonValue]) -> None:
    meta = params.get("_meta")
    if isinstance(meta, dict) and any("task" in key.casefold() for key in meta):
        raise McpProtocolError("MCP task extensions are unsupported")


def _contains_string(value: JsonValue | None) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, list):
        return any(_contains_string(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_string(item) for item in value.values())
    return False


def process_mcp_json_response(handler: StreamHandler, text: str) -> str:
    """Validate one correlated MCP response and reverse only authorized result text."""
    if not text:
        validate_mcp_empty_response(handler)
        return text
    payload = _parse_response_payload(text)
    return _process_mcp_response_payload(handler, payload, text)


def process_mcp_sse_message(handler: StreamHandler, text: str) -> tuple[str, bool]:
    """Classify one SSE JSON-RPC message without treating server calls as final replies."""
    context = handler.mcp_context
    if context is None:
        raise McpProtocolError("MCP response has no request context")
    payload = _parse_response_payload(text)
    if not isinstance(payload, dict):
        raise McpProtocolError("MCP SSE data must be one JSON-RPC object")
    if "method" in payload:
        _validate_server_message(payload)
        if _mcp_pii_enabled(handler):
            _reject_candidates(payload)
        return text, False
    if context.method is None:
        raise McpProtocolError("MCP client response acknowledgement cannot contain a body")
    if context.method.startswith("http/"):
        raise McpProtocolError("MCP GET stream contains an uncorrelated response")
    return _process_mcp_response_payload(handler, payload, text), True


def _process_mcp_response_payload(handler: StreamHandler, payload: Any, text: str) -> str:
    context = handler.mcp_context
    if context is None:
        raise McpProtocolError("MCP response has no request context")
    if context.method is None:
        raise McpProtocolError("MCP client response acknowledgement cannot contain a body")
    if context.method.startswith("http/"):
        if _mcp_pii_enabled(handler):
            _reject_candidates(payload)
        return text
    if context.notification:
        raise McpProtocolError("MCP notification received a response body")
    if not isinstance(payload, dict):
        raise McpProtocolError("MCP response must be one JSON-RPC object")
    _validate_json_bounds(payload)
    _validate_response_envelope(payload, context.request_id)
    if context.method == "initialize":
        result = payload.get("result")
        if not isinstance(result, dict) or result.get("protocolVersion") != MCP_PROTOCOL_VERSION:
            raise McpProtocolError("backend selected an unsupported MCP version")
        if _mcp_pii_enabled(handler):
            _reject_candidates(payload)
        return text
    if context.method != "tools/call":
        if _mcp_pii_enabled(handler):
            _reject_candidates(payload)
        return text
    if not _mcp_pii_enabled(handler):
        return text
    transformed = _reverse_tool_response(handler, payload)
    return text if transformed == payload else _json_dumps(transformed)


def _parse_response_payload(text: str) -> Any:
    try:
        return strict_json_loads(text)
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise McpProtocolError("invalid MCP response JSON") from exc


def _validate_server_message(payload: dict[str, Any]) -> None:
    if set(payload) - {"jsonrpc", "id", "method", "params"} or payload.get("jsonrpc") != "2.0":
        raise McpProtocolError("invalid MCP server JSON-RPC message")
    method = payload.get("method")
    if not isinstance(method, str) or _METHOD_RE.fullmatch(method) is None:
        raise McpProtocolError("invalid MCP server method")
    if method.startswith("tasks/") or method.startswith("notifications/tasks/"):
        raise McpProtocolError("MCP task extensions are unsupported")
    if "id" in payload:
        _request_id(payload)
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise McpProtocolError("MCP server params must be an object")
    _reject_task_metadata(params)
    _validate_json_bounds(payload)


def _mcp_pii_enabled(handler: StreamHandler) -> bool:
    policy = handler.destination_policy
    if not isinstance(policy, McpDestinationPolicy):
        raise McpProtocolError("MCP response has no destination policy")
    return policy.pii_enabled


def validate_mcp_empty_response(handler: StreamHandler) -> None:
    """Allow empty transport responses only where MCP lifecycle semantics permit them."""
    context = handler.mcp_context
    if context is None:
        raise McpProtocolError("MCP response has no request context")
    if context.method is None:
        return
    if context.method == "tools/call" or context.method == "initialize":
        raise McpProtocolError("MCP operation requires a response body")
    if not context.notification and context.method not in {"http/delete"}:
        raise McpProtocolError("MCP request requires a response body")


class McpSseResponseProcessor:
    """Validate and transform one request-scoped Streamable HTTP SSE response."""

    def __init__(self, handler: StreamHandler) -> None:
        """Keep framing and correlation state local to one extProc stream."""
        self._handler = handler
        self._decoder = SseDecoder(emit_empty_frames=True)
        self._final_responses = 0

    def feed(self, text: str, *, final: bool) -> str:
        """Transform complete SSE message events and reject resumable framing."""
        output: list[str] = []
        for event in self._decoder.feed(text, final=final):
            _reject_sse_metadata_candidates(event.lines)
            if event.event != "message" or any(name in {"id", "retry"} for name, _ in event.lines):
                raise McpProtocolError("resumable MCP SSE is unsupported")
            if not event.data:
                output.append(event.render())
                continue
            transformed, is_final = process_mcp_sse_message(self._handler, event.data)
            if not is_final and self._final_responses:
                raise McpProtocolError("MCP SSE contains a message after its final response")
            if is_final:
                self._final_responses += 1
            if self._final_responses > 1:
                raise McpProtocolError("MCP SSE returned multiple JSON-RPC responses")
            if transformed != event.data:
                event.replace_data(transformed)
            output.append(event.render())
        context = self._handler.mcp_context
        if (
            final
            and self._final_responses != 1
            and (
                context is None or context.method is None or not context.method.startswith("http/")
            )
        ):
            validate_mcp_empty_response(self._handler)
        return "".join(output)


def _reject_sse_metadata_candidates(lines: list[tuple[str, str]]) -> None:
    for name, value in lines:
        if name != "data":
            _reject_candidates(name)
            _reject_candidates(value)


def _validate_response_envelope(payload: dict[str, Any], request_id: str | int | None) -> None:
    allowed = {"jsonrpc", "id", "result", "error"}
    if (
        set(payload) - allowed
        or payload.get("jsonrpc") != "2.0"
        or (("result" in payload) == ("error" in payload))
    ):
        raise McpProtocolError("invalid MCP response envelope")
    response_id = payload.get("id")
    if type(response_id) is not type(request_id) or response_id != request_id:
        raise McpProtocolError("MCP response ID mismatch")
    if "error" not in payload:
        return
    error = payload["error"]
    if not isinstance(error, dict) or set(error) - {"code", "message", "data"}:
        raise McpProtocolError("invalid MCP JSON-RPC error")
    code = error.get("code")
    if (
        isinstance(code, bool)
        or not isinstance(code, int)
        or not isinstance(error.get("message"), str)
    ):
        raise McpProtocolError("invalid MCP JSON-RPC error")


def _reverse_tool_response(handler: StreamHandler, payload: dict[str, Any]) -> dict[str, Any]:
    def transform(value: Any, path: tuple[str | int, ...]) -> Any:
        if isinstance(value, str):
            if _is_allowed_result_text(payload, path):
                rewritten, _hits, misses = reverse_placeholders(value, handler.reversal_map)
                handler.reversal_misses += misses
                if misses or RESERVED_PLACEHOLDER_PREFIX_RE.search(rewritten):
                    raise InvalidReversalError("MCP response contained an unknown placeholder")
                return rewritten
            if RESERVED_PLACEHOLDER_PREFIX_RE.search(value):
                raise InvalidReversalError("MCP response placeholder is outside result text")
            return value
        if isinstance(value, list):
            return [transform(item, (*path, index)) for index, item in enumerate(value)]
        if isinstance(value, dict):
            if any(RESERVED_PLACEHOLDER_PREFIX_RE.search(key) for key in value):
                raise InvalidReversalError("MCP response contains a placeholder-shaped key")
            return {key: transform(item, (*path, key)) for key, item in value.items()}
        return value

    return transform(payload, ())


def _is_allowed_result_text(payload: dict[str, Any], path: tuple[str | int, ...]) -> bool:
    if len(path) >= 2 and path[:2] == ("result", "structuredContent"):
        return True
    if len(path) < 4 or path[:2] != ("result", "content") or not isinstance(path[2], int):
        return False
    content = payload.get("result")
    blocks = content.get("content") if isinstance(content, dict) else None
    if not isinstance(blocks, list) or not 0 <= path[2] < len(blocks):
        return False
    block = blocks[path[2]]
    if not isinstance(block, dict):
        return False
    if path[3:] == ("text",):
        return bool(block.get("type") == "text")
    if path[3:] == ("resource", "text"):
        return bool(block.get("type") == "resource" and isinstance(block.get("resource"), dict))
    return False


def _reject_candidates(value: Any) -> None:
    if isinstance(value, str):
        if RESERVED_PLACEHOLDER_PREFIX_RE.search(value):
            raise InvalidReversalError("MCP response contains an unauthorized placeholder")
        return
    if isinstance(value, list):
        for item in value:
            _reject_candidates(item)
        return
    if isinstance(value, dict):
        if any(RESERVED_PLACEHOLDER_PREFIX_RE.search(key) for key in value):
            raise InvalidReversalError("MCP response contains a placeholder-shaped key")
        for item in value.values():
            _reject_candidates(item)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
