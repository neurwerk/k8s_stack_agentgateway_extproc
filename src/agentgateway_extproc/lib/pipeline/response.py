"""Semantic response reversal and notice injection for ext_proc streams."""

# ruff: noqa: ANN401, C901, TRY003

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.lib.masking.reversal import PlaceholderStreamRewriter
from agentgateway_extproc.lib.notice.inject import render_notice
from agentgateway_extproc.lib.notice.report import render_report
from agentgateway_extproc.lib.pipeline.guard import GuardStreamStripper, strip_guard_instruction
from agentgateway_extproc.lib.pipeline.mcp import (
    McpSseResponseProcessor,
    process_mcp_json_response,
)
from agentgateway_extproc.lib.pipeline.sse import SseDecoder, SseEvent
from agentgateway_extproc.metrics import invalid_placeholder_total
from agentgateway_extproc.models.exceptions import InvalidReversalError
from agentgateway_extproc.models.types import RESERVED_PLACEHOLDER_PREFIX_RE

if TYPE_CHECKING:
    from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler

_RESPONSES_ITEM_DONE = {
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.function_call_arguments.done",
}
_RESPONSES_TERMINALS = {
    "response.completed",
    "response.failed",
    "response.incomplete",
    "error",
}
_OPAQUE_RESPONSE_REASONING_FIELDS = (
    "reasoning_content",
    "reasoning",
    "reasoning_details",
    "thinking_blocks",
    "reasoning_signature",
)
type OpaqueResponseReasoning = list[tuple[int, dict[str, Any]]]


def split_safe_prefix(text: str) -> tuple[str, str]:
    """Retain a possible reserved placeholder suffix for compatibility callers."""
    last_open = text.rfind("<")
    if last_open < 0 or ">" in text[last_open:]:
        return text, ""
    suffix = text[last_open:]
    prefixes = ("<REV_", "<ENCRYPTED_")
    if any(prefix.startswith(suffix) or suffix.startswith(prefix) for prefix in prefixes):
        return text[:last_open], suffix
    return text, ""


def strict_json_loads(value: str) -> Any:
    """Decode one strict JSON value without duplicate names or non-finite numbers."""

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("JSON object contains duplicate keys")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON contains unsupported constant: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("JSON contains a non-finite number")
        return parsed

    return json.loads(
        value,
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _record(
    handler: StreamHandler,
    hits: dict[str, int],
    misses: int,
    *,
    report_target: bool = False,
    allow_invalid: bool = False,
) -> None:
    if report_target:
        for entity_type, count in hits.items():
            handler.restored_counts[entity_type] = (
                handler.restored_counts.get(entity_type, 0) + count
            )
    handler.reversal_misses += misses
    if misses:
        if allow_invalid:
            invalid_placeholder_total.inc(misses)
            return
        raise InvalidReversalError("response contained an unknown placeholder")


def _reverse_complete(
    handler: StreamHandler,
    value: str,
    *,
    report_target: bool = False,
    allow_invalid: bool = False,
) -> str:
    """Reverse placeholders in one complete semantic string."""
    rewriter = PlaceholderStreamRewriter(
        handler.reversal_map,
        mark_invalid=allow_invalid,
        entity_prefixes=handler.reversal_entity_prefixes or None,
    )
    guarded = strip_guard_instruction(value) if handler.guard_injected else value
    result, hits, misses = rewriter.feed(guarded, final=True)
    _record(
        handler,
        hits,
        misses,
        report_target=report_target,
        allow_invalid=allow_invalid,
    )
    return result


def _reverse_json_values(
    handler: StreamHandler,
    value: Any,
    *,
    protocol_fields: bool = True,
    path: tuple[str | int, ...] = (),
) -> Any:
    """Reverse complete JSON string values, never keys or protocol structure."""
    if isinstance(value, str):
        if not protocol_fields:
            return _reverse_complete(handler, value)
        if RESERVED_PLACEHOLDER_PREFIX_RE.search(value):
            raise InvalidReversalError("response protocol field contains a placeholder")
        return value
    if isinstance(value, list):
        return [
            _reverse_json_values(
                handler,
                item,
                protocol_fields=protocol_fields,
                path=(*path, index),
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if any(RESERVED_PLACEHOLDER_PREFIX_RE.search(key) for key in value):
            raise InvalidReversalError("response contains a placeholder-shaped JSON key")
        transformed: dict[str, Any] = {}
        for key, item in value.items():
            if (
                protocol_fields
                and isinstance(item, str)
                and _is_nested_json_field(handler, value, key, path)
            ):
                transformed[key] = _reverse_nested_json(handler, item)
            else:
                transformed[key] = _reverse_json_values(
                    handler,
                    item,
                    protocol_fields=protocol_fields,
                    path=(*path, key),
                )
        return transformed
    return value


def _is_nested_json_field(
    handler: StreamHandler,
    container: dict[str, Any],
    key: str,
    path: tuple[str | int, ...],
) -> bool:
    if key == "arguments" and _is_function_arguments_path(handler, container, path):
        return True
    if not handler.response_structured_json or key != "text":
        return False
    return container.get("type") in {"output_text", "response.output_text.done"}


def _is_function_arguments_path(
    handler: StreamHandler,
    container: dict[str, Any],
    path: tuple[str | int, ...],
) -> bool:
    """Recognize only documented Chat and Responses function argument carriers."""
    if handler.response_api_kind == "chat":
        return (
            len(path) == 6
            and path[0] == "choices"
            and isinstance(path[1], int)
            and path[2] in {"message", "delta"}
            and path[3] == "tool_calls"
            and isinstance(path[4], int)
            and path[5] == "function"
        )
    if handler.response_api_kind != "responses":
        return False
    if not path and container.get("type") == "response.function_call_arguments.done":
        return True
    if container.get("type") != "function_call":
        return False
    return (
        (len(path) == 2 and path[0] == "output" and isinstance(path[1], int))
        or path == ("item",)
        or (len(path) == 3 and path[0:2] == ("response", "output") and isinstance(path[2], int))
    )


def _reverse_nested_json(handler: StreamHandler, value: str) -> str:
    """Reverse semantic values in a complete JSON-encoded protocol string."""
    guarded = strip_guard_instruction(value) if handler.guard_injected else value
    if guarded == value and not RESERVED_PLACEHOLDER_PREFIX_RE.search(value):
        return value
    try:
        parsed = strict_json_loads(guarded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise InvalidReversalError("response contains invalid nested JSON") from exc
    return _json_dumps(_reverse_json_values(handler, parsed, protocol_fields=False))


def _reverse_structured_chat_content(handler: StreamHandler, payload: Any) -> None:
    if not handler.response_structured_json or not isinstance(payload, dict):
        return
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = _reverse_nested_json(handler, message["content"])


def process_json_response(handler: StreamHandler, text: str) -> str:
    """Transform one complete JSON document and insert one valid notice carrier."""
    payload = strict_json_loads(text)
    opaque_reasoning = _take_opaque_chat_reasoning(handler, payload, "message")
    if handler.response_api_kind == "chat":
        _reverse_structured_chat_content(handler, payload)
    target = _json_notice_target(handler, payload)
    if target is not None:
        container, field = target
        container[field] = _reverse_complete(
            handler,
            container[field],
            report_target=True,
            allow_invalid=not handler.response_structured_json,
        )
    if not handler.response_structured_json:
        _reverse_human_json_text(handler, payload)
    payload = _reverse_json_values(handler, payload)
    notice = _render_handler_notice(handler)
    transformed_target = _json_notice_target(handler, payload)
    if notice and transformed_target is not None:
        container, field = transformed_target
        container[field] += notice
    _restore_opaque_chat_reasoning(payload, "message", opaque_reasoning)
    return _json_dumps(payload)


def _take_opaque_chat_reasoning(
    handler: StreamHandler, payload: Any, carrier: str
) -> OpaqueResponseReasoning:
    """Detach exact Chat reasoning fields from all response processing."""
    if handler.response_api_kind != "chat" or not isinstance(payload, dict):
        return []
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return []
    detached: OpaqueResponseReasoning = []
    for index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        container = choice.get(carrier)
        if not isinstance(container, dict):
            continue
        values = {
            field: container.pop(field)
            for field in _OPAQUE_RESPONSE_REASONING_FIELDS
            if field in container
        }
        if values:
            detached.append((index, values))
    return detached


def _restore_opaque_chat_reasoning(
    payload: Any, carrier: str, reasoning: OpaqueResponseReasoning
) -> None:
    """Restore detached reasoning values without inspecting or transforming them."""
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list):
        return
    for index, values in reasoning:
        if index >= len(choices) or not isinstance(choices[index], dict):
            continue
        container = choices[index].get(carrier)
        if not isinstance(container, dict):
            continue
        container.update(values)


def _reverse_human_json_text(handler: StreamHandler, payload: Any) -> None:
    """Reverse ordinary assistant text while marking unauthorized model tokens."""
    if not isinstance(payload, dict):
        return
    if handler.response_api_kind == "responses":
        output = payload.get("output")
        if isinstance(output, list):
            for item in output:
                _reverse_output_item_text(handler, item)
        return
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = _reverse_complete(handler, message["content"], allow_invalid=True)


def _reverse_output_item_text(handler: StreamHandler, item: Any) -> None:
    """Reverse every ordinary output_text part in one Responses message item."""
    if not isinstance(item, dict) or item.get("type") != "message":
        return
    content = item.get("content")
    if not isinstance(content, list):
        return
    for part in content:
        if (
            isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            part["text"] = _reverse_complete(handler, part["text"], allow_invalid=True)


def _json_notice_target(handler: StreamHandler, payload: Any) -> tuple[dict[str, Any], str] | None:
    if (
        not handler.response_notice_allowed
        or not _notice_configured(handler)
        or not isinstance(payload, dict)
        or (
            handler.response_api_kind == "responses"
            and payload.get("status") in {"failed", "incomplete"}
        )
    ):
        return None
    if handler.response_api_kind == "responses":
        return _responses_json_notice_target(payload)
    return _chat_json_notice_target(payload)


def _chat_json_notice_target(payload: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if not isinstance(choice, dict) or choice.get("index") not in (0, None):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message, "content"
    return None


def _responses_json_notice_target(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str] | None:
    output = payload.get("output")
    if not isinstance(output, list):
        return None
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                return part, "text"
    return None


def _notice_configured(handler: StreamHandler) -> bool:
    return handler.request_stats is not None or any(
        message.strip() for message in handler.notice_messages
    )


def _render_handler_notice(handler: StreamHandler) -> str:
    if not handler.response_notice_allowed:
        return ""
    report = ""
    if stats := handler.request_stats:
        report = render_report(
            stats.report,
            stats.analysis,
            handler.restored_counts,
            decision=stats.decision,
            route_class=stats.route_class,
        )
    return render_notice(handler.notice_messages, report)


class SseResponseProcessor:
    """Transform Chat and Responses SSE events without buffering the whole response."""

    def __init__(self, handler: StreamHandler) -> None:
        """Initialize event framing and stream-local output channels."""
        self._handler = handler
        self._decoder = SseDecoder()
        self._rewriters: dict[str, PlaceholderStreamRewriter] = {}
        self._guard_strippers: dict[str, GuardStreamStripper] = {}
        self._held: list[SseEvent] = []
        self._terminal_seen = False
        self._chat_metadata: dict[str, Any] = {}
        self._responses_text_coordinates: dict[str, tuple[str, int, int]] = {}
        self._chat_notice_index: int | None = None
        self._chat_notice_emitted = False
        self._responses_coordinates: tuple[str, int, int] | None = None
        self._responses_count_source: str | None = None
        self._responses_tail_held = False
        self._responses_terminal_type: str | None = None
        self._plain_sse = False

    def feed(self, text: str, *, final: bool) -> str:
        """Transform complete SSE events and retain terminal frames for notices."""
        output: list[str] = []
        for event in self._decoder.feed(text, final=final):
            self._reject_event_metadata(event)
            if self._terminal_seen:
                if event.data != "[DONE]":
                    raise InvalidReversalError("SSE stream contains data after its terminal event")
                self._held.append(event)
                continue
            self._transform_event(event)
            terminal = self._is_terminal(event)
            if terminal:
                self._terminal_seen = True
            if self._hold_event(event, terminal=terminal):
                self._held.append(event)
                continue
            notice_event = self._notice_before_chat_finish(event)
            if notice_event is not None:
                output.append(notice_event.render())
            output.append(event.render())
        if final:
            detached: list[tuple[str, str]] = []
            for key, rewriter in self._rewriters.items():
                guarded = ""
                if guard_stripper := self._guard_strippers.get(key):
                    guarded = guard_stripper.feed("", final=True)
                tail, hits, misses = rewriter.feed(guarded, final=True)
                _record(
                    self._handler,
                    hits,
                    misses,
                    allow_invalid=rewriter.marks_invalid,
                )
                if tail and not rewriter.marks_invalid:
                    raise InvalidReversalError("response ended with detached placeholder output")
                if tail:
                    detached.append((key, tail))
            output.extend(self._render_detached_tails(detached))
            output.extend(self._render_terminal())
        return "".join(output)

    def _reject_event_metadata(self, event: SseEvent) -> None:
        for name, value in event.lines:
            if name != "data" and (
                RESERVED_PLACEHOLDER_PREFIX_RE.search(name)
                or RESERVED_PLACEHOLDER_PREFIX_RE.search(value)
            ):
                raise InvalidReversalError("SSE metadata contains an unauthorized placeholder")

    def _transform_event(self, event: SseEvent) -> None:
        if not event.data:
            return
        if event.data == "[DONE]":
            return
        try:
            payload = strict_json_loads(event.data)
        except (ValueError, json.JSONDecodeError):
            if RESERVED_PLACEHOLDER_PREFIX_RE.search(event.data):
                raise InvalidReversalError(
                    "SSE response has unstructured placeholder output"
                ) from None
            self._plain_sse = True
            return
        if not isinstance(payload, dict):
            raise InvalidReversalError("SSE event data must be a JSON object")
        opaque_reasoning = _take_opaque_chat_reasoning(self._handler, payload, "delta")
        if self._handler.response_api_kind == "responses":
            self._transform_responses(payload)
        else:
            self._transform_chat(payload)
        validated = _reverse_json_values(self._handler, payload)
        payload.clear()
        payload.update(validated)
        _restore_opaque_chat_reasoning(payload, "delta", opaque_reasoning)
        event.replace_data(_json_dumps(payload))

    def _transform_chat(self, payload: dict[str, Any]) -> None:
        for key in ("id", "model", "created", "object", "system_fingerprint"):
            if key in payload:
                self._chat_metadata[key] = payload[key]
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict) or not isinstance(choice.get("index"), int):
                continue
            index = choice["index"]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if (
                self._chat_notice_index is None
                and isinstance(delta.get("content"), str)
                and delta["content"]
            ):
                self._chat_notice_index = index
            self._transform_stream_string(
                delta,
                "content",
                f"chat:{index}:content",
                json_string_fragment=self._handler.response_structured_json,
                report_target=(
                    index == self._chat_notice_index
                    and self._handler.response_notice_allowed
                    and _notice_configured(self._handler)
                ),
                allow_invalid=not self._handler.response_structured_json,
            )
            self._transform_chat_tools(delta, index)

    def _transform_chat_tools(self, delta: dict[str, Any], choice_index: int) -> None:
        tools = delta.get("tool_calls")
        if not isinstance(tools, list):
            return
        for position, tool in enumerate(tools):
            if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
                continue
            tool_index = tool.get("index") if isinstance(tool.get("index"), int) else position
            self._transform_stream_string(
                tool["function"],
                "arguments",
                f"chat:{choice_index}:tool:{tool_index}:arguments",
                json_string_fragment=True,
            )

    def _transform_responses(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if not isinstance(event_type, str):
            raise InvalidReversalError("Responses SSE event has no type")
        item_id = payload.get("item_id")
        output_index = payload.get("output_index")
        content_index = payload.get("content_index")
        if (
            event_type == "response.output_text.delta"
            and isinstance(item_id, str)
            and isinstance(output_index, int)
            and isinstance(content_index, int)
        ):
            coordinates = (item_id, output_index, content_index)
            key = f"responses:{item_id}:{output_index}:{content_index}:text"
            self._responses_text_coordinates[key] = coordinates
            if self._responses_coordinates is None:
                self._responses_coordinates = coordinates
                self._responses_count_source = "delta"
            self._transform_stream_string(
                payload,
                "delta",
                key,
                json_string_fragment=self._handler.response_structured_json,
                report_target=(
                    coordinates == self._responses_coordinates
                    and self._responses_count_source == "delta"
                    and self._handler.response_notice_allowed
                    and _notice_configured(self._handler)
                ),
                allow_invalid=not self._handler.response_structured_json,
            )
        elif event_type == "response.function_call_arguments.delta" and isinstance(item_id, str):
            self._transform_stream_string(
                payload,
                "delta",
                f"responses:{item_id}:{output_index}:arguments",
                json_string_fragment=True,
            )
        elif event_type in _RESPONSES_ITEM_DONE | _RESPONSES_TERMINALS:
            self._select_responses_snapshot_target(payload)
            if not self._handler.response_structured_json:
                self._reverse_responses_event_text(payload)
            transformed = _reverse_json_values(self._handler, payload)
            payload.clear()
            payload.update(transformed)

    def _select_responses_snapshot_target(self, payload: dict[str, Any]) -> None:
        if self._responses_coordinates is not None:
            return
        target = _responses_snapshot_text_target(payload)
        if target is None:
            return
        coordinates, container, field = target
        self._responses_coordinates = coordinates
        self._responses_count_source = "snapshot"
        self._responses_tail_held = self._handler.response_notice_allowed and _notice_configured(
            self._handler
        )
        if self._handler.response_structured_json:
            return
        container[field] = _reverse_complete(
            self._handler,
            container[field],
            report_target=(
                self._handler.response_notice_allowed and _notice_configured(self._handler)
            ),
            allow_invalid=not self._handler.response_structured_json,
        )

    def _reverse_responses_event_text(self, payload: dict[str, Any]) -> None:
        """Mark invalid tokens in every ordinary Responses snapshot text field."""
        event_type = payload.get("type")
        if event_type == "response.output_text.done" and isinstance(payload.get("text"), str):
            payload["text"] = _reverse_complete(self._handler, payload["text"], allow_invalid=True)
            return
        if event_type == "response.content_part.done":
            part = payload.get("part")
            if (
                isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                part["text"] = _reverse_complete(self._handler, part["text"], allow_invalid=True)
            return
        if event_type == "response.output_item.done":
            _reverse_output_item_text(self._handler, payload.get("item"))
            return
        if event_type in {"response.completed", "response.incomplete", "response.failed"}:
            response = payload.get("response")
            output = response.get("output") if isinstance(response, dict) else None
            if isinstance(output, list):
                for item in output:
                    _reverse_output_item_text(self._handler, item)

    def _transform_stream_string(
        self,
        payload: dict[str, Any],
        field: str,
        key: str,
        *,
        json_string_fragment: bool = False,
        report_target: bool = False,
        allow_invalid: bool = False,
    ) -> None:
        value = payload.get(field)
        if not isinstance(value, str):
            return
        guarded = value
        if self._handler.guard_injected:
            guard_stripper = self._guard_strippers.setdefault(key, GuardStreamStripper())
            guarded = guard_stripper.feed(value)
        rewriter = self._rewriters.setdefault(
            key,
            PlaceholderStreamRewriter(
                self._handler.reversal_map,
                json_string_fragment=json_string_fragment,
                mark_invalid=allow_invalid,
                entity_prefixes=self._handler.reversal_entity_prefixes or None,
            ),
        )
        transformed, hits, misses = rewriter.feed(guarded)
        _record(
            self._handler,
            hits,
            misses,
            report_target=report_target,
            allow_invalid=allow_invalid,
        )
        payload[field] = transformed

    def _hold_event(self, event: SseEvent, *, terminal: bool) -> bool:
        if self._handler.response_api_kind != "responses":
            return terminal
        if self._responses_tail_held:
            return True
        payload = _event_payload(event)
        coordinates = self._responses_coordinates
        if (
            coordinates is not None
            and self._handler.response_notice_allowed
            and _notice_configured(self._handler)
            and payload is not None
            and payload.get("type") == "response.output_text.done"
            and payload.get("item_id") == coordinates[0]
            and payload.get("output_index") == coordinates[1]
            and payload.get("content_index") == coordinates[2]
        ):
            self._responses_tail_held = True
            return True
        return terminal

    def _is_terminal(self, event: SseEvent) -> bool:
        if event.data == "[DONE]":
            return True
        try:
            payload = strict_json_loads(event.data)
        except (ValueError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        if self._handler.response_api_kind == "responses":
            event_type = payload.get("type")
            if event_type in _RESPONSES_TERMINALS:
                self._responses_terminal_type = event_type
                return True
        return False

    def _notice_before_chat_finish(self, event: SseEvent) -> SseEvent | None:
        """Insert a Chat notice before the selected choice finishes."""
        notice = _render_handler_notice(self._handler)
        if not notice or not self._handler.response_notice_allowed:
            return None
        payload = _event_payload(event)
        if payload is None or self._handler.response_api_kind == "responses":
            return None
        if self._chat_notice_emitted or not self._chat_finishes_notice_target(payload):
            return None
        notice_event = self._chat_notice_event(notice)
        self._chat_notice_emitted = notice_event is not None
        return notice_event

    def _chat_finishes_notice_target(self, payload: dict[str, Any]) -> bool:
        target = self._chat_notice_index
        choices = payload.get("choices")
        return (
            target is not None
            and isinstance(choices, list)
            and any(
                isinstance(choice, dict)
                and choice.get("index") == target
                and choice.get("finish_reason") is not None
                for choice in choices
            )
        )

    def _render_terminal(self) -> list[str]:
        notice = _render_handler_notice(self._handler)
        if not notice or not self._handler.response_notice_allowed:
            return [event.render() for event in self._held]
        if self._handler.response_api_kind == "responses":
            if self._responses_terminal_type in {
                "response.failed",
                "response.incomplete",
                "error",
            }:
                return [event.render() for event in self._held]
            return self._render_responses_notice(notice)
        if self._chat_notice_emitted:
            return [event.render() for event in self._held]
        return self._render_chat_notice(notice)

    def _render_detached_tails(self, detached: list[tuple[str, str]]) -> list[str]:
        """Emit safe markers retained until the semantic channel reached EOF."""
        chat_events: list[str] = []
        responses: list[tuple[tuple[str, int, int], str]] = []
        for key, tail in detached:
            if key.startswith("chat:") and key.endswith(":content"):
                index = int(key.split(":", 2)[1])
                payload = {
                    **self._chat_metadata,
                    "object": self._chat_metadata.get("object", "chat.completion.chunk"),
                    "choices": [
                        {"index": index, "delta": {"content": tail}, "finish_reason": None}
                    ],
                }
                chat_events.append(SseEvent(lines=[("data", _json_dumps(payload))]).render())
            elif coordinates := self._responses_text_coordinates.get(key):
                responses.append((coordinates, tail))
        if not responses:
            return chat_events
        sequence = self._first_held_sequence()
        response_events: list[str] = []
        for offset, (coordinates, tail) in enumerate(responses):
            item_id, output_index, content_index = coordinates
            payload = {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "delta": tail,
                "sequence_number": sequence + offset,
            }
            response_events.append(
                SseEvent(
                    lines=[("event", "response.output_text.delta"), ("data", _json_dumps(payload))]
                ).render()
            )
        self._increment_held_sequences(len(responses))
        return [*chat_events, *response_events]

    def _first_held_sequence(self) -> int:
        for event in self._held:
            payload = _event_payload(event)
            sequence = payload.get("sequence_number") if payload is not None else None
            if isinstance(sequence, int):
                return sequence
        return 0

    def _increment_held_sequences(self, amount: int) -> None:
        for event in self._held:
            payload = _event_payload(event)
            if payload is not None and isinstance(payload.get("sequence_number"), int):
                payload["sequence_number"] += amount
                event.replace_data(_json_dumps(payload))

    def _render_chat_notice(self, notice: str) -> list[str]:
        if self._plain_sse:
            return [SseEvent(lines=[("data", notice)]).render()] + [
                event.render() for event in self._held
            ]
        notice_event = self._chat_notice_event(notice)
        if notice_event is None:
            return [event.render() for event in self._held]
        return [notice_event.render()] + [event.render() for event in self._held]

    def _chat_notice_event(self, notice: str) -> SseEvent | None:
        index = self._chat_notice_index
        if index is None:
            return None
        payload = {
            **self._chat_metadata,
            "object": self._chat_metadata.get("object", "chat.completion.chunk"),
            "choices": [{"index": index, "delta": {"content": notice}, "finish_reason": None}],
        }
        return SseEvent(lines=[("data", _json_dumps(payload))])

    def _render_responses_notice(self, notice: str) -> list[str]:
        coordinates = self._responses_coordinates
        if coordinates is None:
            return [event.render() for event in self._held]
        sequence = 0
        for event in self._held:
            if event.data == "[DONE]":
                continue
            payload = strict_json_loads(event.data)
            if isinstance(payload, dict) and isinstance(payload.get("sequence_number"), int):
                sequence = payload["sequence_number"]
                break
        delta = {
            "type": "response.output_text.delta",
            "item_id": coordinates[0],
            "output_index": coordinates[1],
            "content_index": coordinates[2],
            "delta": notice,
            "sequence_number": sequence,
        }
        notice_event = SseEvent(
            lines=[("event", "response.output_text.delta"), ("data", _json_dumps(delta))]
        )
        output = [notice_event.render()]
        for event in self._held:
            if event.data != "[DONE]":
                payload = strict_json_loads(event.data)
                if isinstance(payload, dict):
                    if isinstance(payload.get("sequence_number"), int):
                        payload["sequence_number"] += 1
                    _append_notice_to_snapshots(payload, coordinates, notice)
                    event.replace_data(_json_dumps(payload))
            output.append(event.render())
        return output


def _responses_snapshot_text_target(
    payload: dict[str, Any],
) -> tuple[tuple[str, int, int], dict[str, Any], str] | None:
    """Find the first logical Responses output text when no deltas were emitted."""
    event_type = payload.get("type")
    if event_type == "response.output_text.done":
        return _coordinate_text_target(payload, payload, "text")
    if event_type == "response.content_part.done":
        part = payload.get("part")
        if isinstance(part, dict) and part.get("type") == "output_text":
            return _coordinate_text_target(payload, part, "text")
    if event_type == "response.output_item.done":
        output_index = payload.get("output_index")
        if isinstance(output_index, int):
            return _item_text_target(payload.get("item"), output_index)
    if event_type == "response.completed":
        response = payload.get("response")
        output = response.get("output") if isinstance(response, dict) else None
        if isinstance(output, list):
            for output_index, item in enumerate(output):
                if target := _item_text_target(item, output_index):
                    return target
    return None


def _coordinate_text_target(
    payload: dict[str, Any], container: dict[str, Any], field: str
) -> tuple[tuple[str, int, int], dict[str, Any], str] | None:
    item_id = payload.get("item_id")
    output_index = payload.get("output_index")
    content_index = payload.get("content_index")
    if (
        isinstance(item_id, str)
        and isinstance(output_index, int)
        and isinstance(content_index, int)
        and isinstance(container.get(field), str)
    ):
        return (item_id, output_index, content_index), container, field
    return None


def _item_text_target(
    value: Any, output_index: int
) -> tuple[tuple[str, int, int], dict[str, Any], str] | None:
    if not isinstance(value, dict) or not isinstance(value.get("id"), str):
        return None
    content = value.get("content")
    if not isinstance(content, list):
        return None
    for content_index, part in enumerate(content):
        if (
            isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            return (value["id"], output_index, content_index), part, "text"
    return None


def _append_notice_to_snapshots(
    payload: dict[str, Any], coordinates: tuple[str, int, int], notice: str
) -> None:
    """Update only the selected Responses text snapshots after inserting a delta."""
    item_id, output_index, content_index = coordinates
    if (
        payload.get("type") == "response.output_text.done"
        and payload.get("item_id") == item_id
        and payload.get("output_index") == output_index
        and payload.get("content_index") == content_index
        and isinstance(payload.get("text"), str)
    ):
        payload["text"] += notice
        return
    if (
        payload.get("type") == "response.content_part.done"
        and payload.get("item_id") == item_id
        and payload.get("output_index") == output_index
        and payload.get("content_index") == content_index
    ):
        part = payload.get("part")
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            part["text"] += notice
        return
    if payload.get("type") == "response.output_item.done":
        if payload.get("output_index") == output_index:
            _append_item_notice(payload.get("item"), item_id, content_index, notice)
        return
    if payload.get("type") != "response.completed":
        return
    response = payload.get("response")
    output = response.get("output") if isinstance(response, dict) else None
    if isinstance(output, list) and 0 <= output_index < len(output):
        _append_item_notice(output[output_index], item_id, content_index, notice)


def _append_item_notice(value: Any, item_id: str, content_index: int, notice: str) -> None:
    if not isinstance(value, dict) or value.get("id") not in {None, item_id}:
        return
    content = value.get("content")
    if not isinstance(content, list) or not 0 <= content_index < len(content):
        return
    part = content[content_index]
    if (
        isinstance(part, dict)
        and part.get("type") == "output_text"
        and isinstance(part.get("text"), str)
    ):
        part["text"] += notice


def _event_payload(event: SseEvent) -> dict[str, Any] | None:
    """Return one parsed object payload for already validated event data."""
    if not event.data or event.data == "[DONE]":
        return None
    try:
        payload = strict_json_loads(event.data)
    except (ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def process_response_chunk(
    handler: StreamHandler, chunk: str, end_of_stream: bool
) -> ext_proc_pb2.ProcessingResponse:
    """Transform a decoded body chunk according to the negotiated response format."""
    if handler.response_api_kind == "mcp" and handler.response_format == "json":
        transformed = process_mcp_json_response(handler, chunk)
    elif handler.response_api_kind == "mcp" and handler.response_format == "sse":
        if handler.sse_processor is None:
            handler.sse_processor = McpSseResponseProcessor(handler)
        if not isinstance(handler.sse_processor, McpSseResponseProcessor):
            raise InvalidReversalError("MCP response processor state is invalid")
        transformed = handler.sse_processor.feed(chunk, final=end_of_stream)
    elif handler.response_format == "sse":
        if handler.sse_processor is None:
            handler.sse_processor = SseResponseProcessor(handler)
        transformed = handler.sse_processor.feed(chunk, final=end_of_stream)
    elif handler.response_format == "json":
        transformed = process_json_response(handler, chunk)
    else:
        guarded = chunk
        if handler.guard_injected:
            if handler.plain_guard_stripper is None:
                handler.plain_guard_stripper = GuardStreamStripper()
            guarded = handler.plain_guard_stripper.feed(chunk, final=end_of_stream)
        if handler.plain_rewriter is None:
            handler.plain_rewriter = PlaceholderStreamRewriter(
                handler.reversal_map,
                mark_invalid=True,
                entity_prefixes=handler.reversal_entity_prefixes or None,
            )
        transformed, hits, misses = handler.plain_rewriter.feed(guarded, final=end_of_stream)
        _record(
            handler,
            hits,
            misses,
            report_target=(handler.response_notice_allowed and _notice_configured(handler)),
            allow_invalid=True,
        )
        if end_of_stream:
            transformed += _render_handler_notice(handler)
            handler.plain_rewriter = None
            handler.plain_guard_stripper = None
    if end_of_stream:
        handler.reversal_map.clear()
        handler.notice_messages.clear()
    return _streamed(transformed, end_of_stream)


def _streamed(text: str, end_of_stream: bool) -> ext_proc_pb2.ProcessingResponse:
    return ext_proc_pb2.ProcessingResponse(
        response_body=ext_proc_pb2.BodyResponse(
            response=ext_proc_pb2.CommonResponse(
                body_mutation={
                    "streamed_response": {"body": text.encode(), "end_of_stream": end_of_stream}
                }
            )
        )
    )
