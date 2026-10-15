"""Bounded native JSON/SSE restoration for text and client function tools."""

# Native REST field spelling and content-free validation errors match the wire.
# ruff: noqa: N815, TRY003

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal

from neurwerk_request_segments.native import (
    AnthropicText,
    AnthropicToolUse,
    GoogleContent,
    NativeModel,
)
from pydantic import Field, TypeAdapter

from agentgateway_extproc.lib.json_limits import strict_json_loads
from agentgateway_extproc.lib.masking.reversal import PlaceholderStreamRewriter
from agentgateway_extproc.lib.pipeline.guard import GuardStreamStripper
from agentgateway_extproc.lib.pipeline.sse import SseDecoder, SseEvent
from agentgateway_extproc.models.exceptions import InvalidReversalError
from agentgateway_extproc.models.types import RESERVED_PLACEHOLDER_PREFIX_RE

if TYPE_CHECKING:
    from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler

type Count = Annotated[int, Field(ge=0)]
type Index = Annotated[int, Field(ge=0, le=255)]
type Stop = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal"]
type OutputBlock = Annotated[AnthropicText | AnthropicToolUse, Field(discriminator="type")]


class CacheCreation(NativeModel):
    """Native cache counts are preserved as provider facts."""

    ephemeral_5m_input_tokens: Count | None = None
    ephemeral_1h_input_tokens: Count | None = None


class AnthropicUsage(NativeModel):
    """Supported token accounting fields, never fabricated from response text."""

    input_tokens: Count
    output_tokens: Count
    cache_creation_input_tokens: Count | None = None
    cache_read_input_tokens: Count | None = None
    cache_creation: CacheCreation | None = None
    service_tier: Literal["standard", "priority", "batch"] | None = None


class AnthropicResponse(NativeModel):
    """Text/function Messages response envelope."""

    id: str = Field(min_length=1, max_length=256)
    type: Literal["message"]
    role: Literal["assistant"]
    model: str = Field(min_length=1, max_length=256)
    content: list[OutputBlock] = Field(max_length=256)
    stop_reason: Stop | None
    stop_sequence: str | None = None
    usage: AnthropicUsage


class MessageStart(NativeModel):
    """Start exactly one Messages stream."""

    type: Literal["message_start"]
    message: AnthropicResponse


class BlockStart(NativeModel):
    """Start one native text or tool block."""

    type: Literal["content_block_start"]
    index: Index
    content_block: OutputBlock


class TextDelta(NativeModel):
    """Text fragment in a Messages block."""

    type: Literal["text_delta"]
    text: str


class InputDelta(NativeModel):
    """A fragment of tool-input JSON, held until the complete object is validated."""

    type: Literal["input_json_delta"]
    partial_json: str


class BlockDelta(NativeModel):
    """Only explicitly qualified content delta variants are accepted."""

    type: Literal["content_block_delta"]
    index: Index
    delta: Annotated[TextDelta | InputDelta, Field(discriminator="type")]


class BlockStop(NativeModel):
    """Close exactly one existing block."""

    type: Literal["content_block_stop"]
    index: Index


class StopDelta(NativeModel):
    """Terminal message controls."""

    stop_reason: Stop
    stop_sequence: str | None = None


class DeltaUsage(NativeModel):
    """Cumulative accounting at message completion."""

    output_tokens: Count
    input_tokens: Count | None = None
    cache_creation_input_tokens: Count | None = None
    cache_read_input_tokens: Count | None = None


class MessageDelta(NativeModel):
    """Final controls after every content block has closed."""

    type: Literal["message_delta"]
    delta: StopDelta
    usage: DeltaUsage


class MessageStop(NativeModel):
    """Native terminal event."""

    type: Literal["message_stop"]


class Ping(NativeModel):
    """Content-free provider keepalive."""

    type: Literal["ping"]


_ANTHROPIC_EVENT = TypeAdapter(
    Annotated[
        MessageStart | BlockStart | BlockDelta | BlockStop | MessageDelta | MessageStop | Ping,
        Field(discriminator="type"),
    ]
)


class TokenDetail(NativeModel):
    """Modality counts are native provider metadata."""

    modality: Literal["TEXT", "IMAGE", "VIDEO", "AUDIO", "DOCUMENT"]
    tokenCount: Count


class GoogleUsage(NativeModel):
    """Preserve supported Gemini usage fields without claiming completeness."""

    promptTokenCount: Count | None = None
    candidatesTokenCount: Count | None = None
    totalTokenCount: Count | None = None
    cachedContentTokenCount: Count | None = None
    thoughtsTokenCount: Count | None = None
    toolUsePromptTokenCount: Count | None = None
    promptTokensDetails: list[TokenDetail] | None = Field(default=None, max_length=16)
    candidatesTokensDetails: list[TokenDetail] | None = Field(default=None, max_length=16)
    cacheTokensDetails: list[TokenDetail] | None = Field(default=None, max_length=16)
    toolUsePromptTokensDetails: list[TokenDetail] | None = Field(default=None, max_length=16)


class SafetyRating(NativeModel):
    """Provider safety classification, excluding arbitrary diagnostic text."""

    category: str = Field(max_length=128, pattern=r"^HARM_CATEGORY_[A-Z_]+$")
    probability: Literal["NEGLIGIBLE", "LOW", "MEDIUM", "HIGH", "HARM_PROBABILITY_UNSPECIFIED"]
    blocked: bool | None = None


class PromptFeedback(NativeModel):
    """A provider block can terminate a response without candidates."""

    blockReason: (
        Literal["SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY"] | None
    ) = None
    safetyRatings: list[SafetyRating] | None = Field(default=None, max_length=16)


class GoogleCandidate(NativeModel):
    """Native text/function candidate; retrieval, media and thought parts are unqualified."""

    index: Annotated[int, Field(ge=0, le=7)] = 0
    content: GoogleContent | None = None
    finishReason: (
        Literal[
            "STOP",
            "MAX_TOKENS",
            "SAFETY",
            "RECITATION",
            "OTHER",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "MALFORMED_FUNCTION_CALL",
            "UNEXPECTED_TOOL_CALL",
            "NO_IMAGE",
        ]
        | None
    ) = None
    safetyRatings: list[SafetyRating] | None = Field(default=None, max_length=16)
    tokenCount: Count | None = None
    avgLogprobs: float | None = None


class GoogleResponse(NativeModel):
    """Qualified generateContent response and stream-chunk shape."""

    candidates: list[GoogleCandidate] = Field(default_factory=list, max_length=8)
    promptFeedback: PromptFeedback | None = None
    usageMetadata: GoogleUsage | None = None
    modelVersion: str | None = Field(default=None, max_length=256)
    responseId: str | None = Field(default=None, max_length=256)


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _control(value: object) -> None:
    """Never restore model-selected placeholders in protocol keys or controls."""
    if RESERVED_PLACEHOLDER_PREFIX_RE.search(_dump(value)):
        raise InvalidReversalError("native response contains a placeholder in protocol data")


def _restore_values(handler: StreamHandler, value: object) -> object:
    from agentgateway_extproc.lib.pipeline.response import _reverse_complete

    if isinstance(value, str):
        return _reverse_complete(handler, value)
    if isinstance(value, list):
        return [_restore_values(handler, item) for item in value]
    if isinstance(value, dict):
        _control(list(value))
        return {key: _restore_values(handler, item) for key, item in value.items()}
    return value


def _block(handler: StreamHandler, block: dict[str, object]) -> None:
    from agentgateway_extproc.lib.pipeline.response import _reverse_complete

    if block["type"] == "text":
        value = block.pop("text")
        _control(block)
        if not isinstance(value, str):
            raise InvalidReversalError
        block["text"] = _reverse_complete(handler, value, allow_invalid=True)
    else:
        value = block.pop("input")
        _control(block)
        block["input"] = _restore_values(handler, value)


def process_native_json(handler: StreamHandler, text: str) -> str:
    """Restore text and function data only after the whole native envelope validates."""
    from agentgateway_extproc.lib.pipeline.response import _reverse_complete, _reverse_nested_json

    raw = strict_json_loads(text)
    if handler.response_api_kind == "anthropic":
        response = AnthropicResponse.model_validate(raw)
        if response.stop_reason is None:
            raise InvalidReversalError("incomplete native message")
        data = response.model_dump(exclude_unset=True)
        blocks = data.pop("content")
        _control(data)
        for block in blocks:
            _block(handler, block)
        data["content"] = blocks
    else:
        response = GoogleResponse.model_validate(raw)
        _validate_google(response, complete=True)
        data = response.model_dump(exclude_unset=True)
        for candidate in data.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                if "text" in part:
                    value = part.pop("text")
                    _control(part)
                    part["text"] = (
                        _reverse_nested_json(handler, value)
                        if handler.response_structured_json
                        else _reverse_complete(handler, value, allow_invalid=True)
                    )
                else:
                    call = part["functionCall"]
                    args = call.pop("args")
                    _control(call)
                    call["args"] = _restore_values(handler, args)
        _google_controls(data)
    return _dump(data)


def _validate_google(response: GoogleResponse, *, complete: bool) -> None:
    indices = [candidate.index for candidate in response.candidates]
    if len(indices) != len(set(indices)) or (
        not response.candidates
        and response.usageMetadata is None
        and not (response.promptFeedback and response.promptFeedback.blockReason)
    ):
        raise InvalidReversalError("invalid native candidate envelope")
    for candidate in response.candidates:
        if complete and candidate.finishReason is None:
            raise InvalidReversalError("incomplete native candidate")
        if candidate.content is not None and (
            candidate.content.role != "model"
            or any(part.functionResponse is not None for part in candidate.content.parts)
        ):
            raise InvalidReversalError("invalid native response content")


def _google_controls(data: dict[str, Any]) -> None:
    """Check metadata independently of mutable candidate content."""
    from copy import deepcopy

    controls = deepcopy(data)
    for candidate in controls.get("candidates", []):
        for part in (candidate.get("content") or {}).get("parts", []):
            part.pop("text", None)
            if "functionCall" in part:
                part["functionCall"].pop("args", None)
    _control(controls)


class NativeSseProcessor:
    """Restore split text tokens and complete tool JSON with native event ordering."""

    def __init__(self, handler: StreamHandler) -> None:
        """Keep response state bounded by the stream's existing decoded-byte budget."""
        self.handler = handler
        self.decoder = SseDecoder()
        self.started = False
        self.terminal = False
        self.message_delta = False
        self.blocks: dict[int, str] = {}
        self.seen_blocks: set[int] = set()
        self.inputs: dict[int, list[str]] = {}
        self.rewriters: dict[tuple[int, int], PlaceholderStreamRewriter] = {}
        self.guards: dict[tuple[int, int], GuardStreamStripper] = {}
        self.candidates: set[int] = set()
        self.finished: set[int] = set()
        self.function_calls: set[tuple[int, int]] = set()

    def feed(self, text: str, *, final: bool) -> str:
        """Validate every frame before emitting it; EOF cannot complete a truncated message."""
        output: list[str] = []
        for event in self.decoder.feed(text, final=final):
            if not event.data:
                # Comments are transport keepalives, never forward opaque diagnostic values.
                continue
            if (
                any(name not in {"event", "data"} for name, _ in event.lines)
                or sum(name == "event" for name, _ in event.lines) > 1
            ):
                raise InvalidReversalError("unsupported native SSE metadata")
            raw = strict_json_loads(event.data)
            if self.handler.response_api_kind == "anthropic":
                output.extend(self._anthropic(event, raw))
            else:
                output.extend(self._google(event, raw))
        if final:
            if self.handler.response_api_kind == "anthropic":
                if not self.terminal:
                    raise InvalidReversalError("incomplete native stream")
            elif not (self.terminal or (self.candidates and self.candidates == self.finished)):
                raise InvalidReversalError("incomplete native stream")
        return "".join(output)

    def _text(self, key: tuple[int, int], value: str, *, final: bool = False) -> str:
        from agentgateway_extproc.lib.pipeline.response import _record

        if not self.handler.text_pii_enabled:
            return value
        rewriter = self.rewriters.setdefault(
            key,
            PlaceholderStreamRewriter(
                self.handler.reversal_map,
                mark_invalid=not self.handler.response_structured_json,
                entity_prefixes=self.handler.reversal_entity_prefixes or None,
                json_string_fragment=self.handler.response_structured_json,
            ),
        )
        if self.handler.guard_injected:
            value = self.guards.setdefault(key, GuardStreamStripper()).feed(value, final=final)
        value, hits, misses = rewriter.feed(value, final=final)
        _record(self.handler, hits, misses, allow_invalid=not self.handler.response_structured_json)
        if final:
            self.rewriters.pop(key, None)
            self.guards.pop(key, None)
        return value

    def _anthropic(self, event: SseEvent, raw: object) -> list[str]:  # noqa: C901
        parsed = _ANTHROPIC_EVENT.validate_python(raw)
        data = parsed.model_dump(exclude_unset=True)
        kind = parsed.type
        if event.event != kind or self.terminal:
            raise InvalidReversalError("invalid native event order")
        extra: list[str] = []
        if isinstance(parsed, Ping):
            return [event.render()]
        if isinstance(parsed, MessageStart):
            if self.started or parsed.message.content or parsed.message.stop_reason is not None:
                raise InvalidReversalError("invalid native message start")
            _control(data)
            self.started = True
        elif not self.started:
            raise InvalidReversalError("native stream has no message start")
        elif isinstance(parsed, BlockStart):
            if self.message_delta or parsed.index in self.seen_blocks:
                raise InvalidReversalError("duplicate native block")
            self.seen_blocks.add(parsed.index)
            self.blocks[parsed.index] = parsed.content_block.type
            block = data["content_block"]
            if isinstance(parsed.content_block, AnthropicText):
                block["text"] = self._text((parsed.index, 0), parsed.content_block.text)
            else:
                if parsed.content_block.input:
                    raise InvalidReversalError("streamed tool input must start empty")
                _control(block)
                self.inputs[parsed.index] = []
        elif isinstance(parsed, BlockDelta):
            expected = "text" if isinstance(parsed.delta, TextDelta) else "tool_use"
            if self.blocks.get(parsed.index) != expected:
                raise InvalidReversalError("native delta has no matching block")
            if isinstance(parsed.delta, TextDelta):
                data["delta"]["text"] = self._text((parsed.index, 0), parsed.delta.text)
            else:
                self.inputs[parsed.index].append(parsed.delta.partial_json)
                return []
        elif isinstance(parsed, BlockStop):
            block_type = self.blocks.pop(parsed.index, None)
            if block_type is None:
                raise InvalidReversalError("native block stopped without a start")
            if block_type == "text":
                tail = self._text((parsed.index, 0), "", final=True)
                if tail:
                    extra.append(
                        _event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": parsed.index,
                                "delta": {"type": "text_delta", "text": tail},
                            },
                        )
                    )
            else:
                value = strict_json_loads("".join(self.inputs.pop(parsed.index)) or "{}")
                if not isinstance(value, dict):
                    raise InvalidReversalError("tool input must be an object")
                restored = _restore_values(self.handler, value)
                extra.append(
                    _event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": parsed.index,
                            "delta": {"type": "input_json_delta", "partial_json": _dump(restored)},
                        },
                    )
                )
        elif isinstance(parsed, MessageDelta):
            if self.blocks or self.message_delta:
                raise InvalidReversalError("native message completed with open blocks")
            self.message_delta = True
            _control(data)
        elif isinstance(parsed, MessageStop):
            if not self.message_delta or self.blocks:
                raise InvalidReversalError("native message has no completion")
            self.terminal = True
        event.replace_data(_dump(data))
        return [*extra, event.render()]

    def _google(self, event: SseEvent, raw: object) -> list[str]:  # noqa: C901
        if event.event != "message" or self.terminal:
            raise InvalidReversalError("invalid native event")
        parsed = GoogleResponse.model_validate(raw)
        _validate_google(parsed, complete=False)
        data = parsed.model_dump(exclude_unset=True)
        for candidate, wire in zip(parsed.candidates, data.get("candidates", []), strict=True):
            index = candidate.index
            if index in self.finished:
                raise InvalidReversalError("native candidate continued after completion")
            self.candidates.add(index)
            parts = (wire.get("content") or {}).get("parts", [])
            for pi, part in enumerate(parts):
                key = (index, pi)
                if "text" in part:
                    part["text"] = self._text(key, part["text"])
                else:
                    if key in self.function_calls:
                        raise InvalidReversalError("partial native function calls are unsupported")
                    self.function_calls.add(key)
                    call = part["functionCall"]
                    call["args"] = _restore_values(self.handler, call["args"])
            if candidate.finishReason is not None:
                self.finished.add(index)
                for key in list(self.rewriters):
                    if key[0] == index:
                        tail = self._text(key, "", final=True)
                        if tail:
                            if key[1] < len(parts) and "text" in parts[key[1]]:
                                parts[key[1]]["text"] += tail
                            else:
                                raise InvalidReversalError("detached native text tail")
        if parsed.promptFeedback and parsed.promptFeedback.blockReason:
            if self.candidates:
                raise InvalidReversalError("conflicting native block response")
            self.terminal = True
        _google_controls(data)
        event.replace_data(_dump(data))
        return [event.render()]


def _event(kind: str, data: object) -> str:
    return f"event: {kind}\ndata: {_dump(data)}\n\n"
