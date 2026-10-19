"""V5 response admission and protocol completion, before any semantic transformation."""

# All rejection messages are fixed; provider content must never become diagnostics.
# ruff: noqa: TRY003

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING

from neurwerk_request_segments.reasoning import CHAT_REASONING_FIELDS

from agentgateway_extproc.lib.json_limits import strict_json_loads
from agentgateway_extproc.lib.pipeline.reasoning import (
    ChatReasoningTracker,
    response_reasoning_digest,
)
from agentgateway_extproc.lib.pipeline.sse import SseEvent
from agentgateway_extproc.models.exceptions import InvalidReversalError
from agentgateway_extproc.models.openai_response import (
    RESPONSES_EVENT,
    ArgumentsDeltaEvent,
    ArgumentsDoneEvent,
    ChatChunk,
    ChatResponse,
    ItemCoordinate,
    ItemEvent,
    LifecycleEvent,
    MetadataEvent,
    OutputFunction,
    OutputItem,
    OutputMessage,
    OutputPart,
    OutputReasoning,
    OutputText,
    PartEvent,
    ReasoningDeltaEvent,
    ReasoningDoneEvent,
    RefusalDoneEvent,
    ResponsesResponse,
    SummaryDeltaEvent,
    SummaryDoneEvent,
    SummaryPartEvent,
    TextDeltaEvent,
    TextDoneEvent,
    ToolDelta,
)

if TYPE_CHECKING:
    from _hashlib import HASH


def _unique(values: list[int] | list[str]) -> None:
    if len(values) != len(set(values)):
        raise InvalidReversalError("duplicate response coordinate")


def _arguments(value: str) -> None:
    if not isinstance(strict_json_loads(value), dict):
        raise InvalidReversalError("function arguments must be a complete JSON object")


def _complete_item(item: OutputItem) -> None:
    if item.status == "in_progress":
        raise InvalidReversalError("incomplete output item")
    if isinstance(item, OutputFunction):
        _arguments(item.arguments)


def _complete_response(response: ResponsesResponse) -> None:
    if response.status not in {"completed", "incomplete"} or (
        (response.status == "incomplete") != (response.incomplete_details is not None)
    ):
        raise InvalidReversalError("response has no valid protocol completion")
    _unique([item.id for item in response.output])
    for item in response.output:
        _complete_item(item)


def validate_json_response(kind: str, payload: object) -> None:
    """Admit only complete, supported JSON envelopes, including all nested output fields."""
    if kind == "responses":
        _complete_response(ResponsesResponse.model_validate(payload))
        return
    response = ChatResponse.model_validate(payload)
    _unique([choice.index for choice in response.choices])
    for choice in response.choices:
        tools = choice.message.tool_calls or []
        _unique([tool.id for tool in tools])
        if choice.finish_reason == "tool_calls" and not tools:
            raise InvalidReversalError("tool completion has no calls")
        for tool in tools:
            _arguments(tool.function.arguments)


@dataclass
class _Arguments:
    fragments: list[str] = field(default_factory=list)
    digest: bytes | None = None

    def append(self, value: str) -> None:
        if self.digest is not None:
            raise InvalidReversalError("function arguments continued after completion")
        self.fragments.append(value)

    def finish(self, snapshot: str | None = None) -> None:
        if self.digest is not None:
            raise InvalidReversalError("duplicate function argument completion")
        value = "".join(self.fragments)
        if snapshot is not None:
            if self.fragments and value != snapshot:
                raise InvalidReversalError("function argument snapshot disagrees")
            value = snapshot
        _arguments(value)
        self.digest = sha256(value.encode()).digest()
        self.fragments.clear()

    def check(self, value: str) -> None:
        if self.digest is None or self.digest != sha256(value.encode()).digest():
            raise InvalidReversalError("function arguments are not closed consistently")


@dataclass
class _ChatTool:
    id: str
    name: str
    arguments: _Arguments = field(default_factory=_Arguments)


@dataclass
class _Part:
    kind: str
    digest: HASH = field(default_factory=sha256)
    text_done: bool = False
    closed: bool = False

    def append(self, value: str) -> None:
        if self.text_done or self.closed:
            raise InvalidReversalError("text continued after completion")
        self.digest.update(value.encode())

    def check(self, value: str) -> None:
        if self.digest.digest() != sha256(value.encode()).digest():
            raise InvalidReversalError("text snapshot disagrees")


def _part_text(part: OutputPart) -> str:
    return part.text if isinstance(part, OutputText) else part.refusal


@dataclass
class _Item:
    initial: OutputItem
    arguments: _Arguments = field(default_factory=_Arguments)
    parts: dict[int, _Part] = field(default_factory=dict)
    summaries: dict[int, _Part] = field(default_factory=dict)
    encrypted_digest: str | None = None
    closed: bool = False

    def check(self, item: OutputItem) -> None:
        if (item.id, item.type) != (self.initial.id, self.initial.type):
            raise InvalidReversalError("output item identity changed")
        _complete_item(item)
        if isinstance(item, OutputFunction) and isinstance(self.initial, OutputFunction):
            if (item.call_id, item.name) != (self.initial.call_id, self.initial.name):
                raise InvalidReversalError("function identity changed")
            self.arguments.check(item.arguments)
        elif isinstance(item, OutputMessage):
            if len(item.content) != len(self.parts) or any(
                not part.closed for part in self.parts.values()
            ):
                raise InvalidReversalError("output item has open content parts")
            for index, part in enumerate(item.content):
                state = self.parts[index]
                if state.kind != part.type:
                    raise InvalidReversalError("output part type changed")
                state.check(_part_text(part))
        elif isinstance(item, OutputReasoning):
            self._check_reasoning(item)

    def _check_reasoning(self, item: OutputReasoning) -> None:
        for parts, snapshots in ((self.summaries, item.summary), (self.parts, item.content or [])):
            if len(parts) != len(snapshots) or any(not part.closed for part in parts.values()):
                raise InvalidReversalError("reasoning item has open text parts")
            for index, part in enumerate(snapshots):
                parts[index].check(part.text)
        digest = response_reasoning_digest(item.model_dump())
        if self.closed and self.encrypted_digest != digest:
            raise InvalidReversalError("reasoning snapshot disagrees")
        self.encrypted_digest = digest


class OpenAIStreamValidator:
    """Track bounded v5 choices/items and require a real protocol terminal at EOF."""

    def __init__(
        self, kind: str, *, openrouter: bool = False, local_created_drift: bool = False
    ) -> None:
        """Keep only stream-local identities, hashes and incomplete tool arguments."""
        self.kind = kind
        self.openrouter = openrouter
        self.local_created_drift = local_created_drift
        self._router_usage_seen = False
        self._finish_reasons: dict[int, str] = {}
        self.terminal = False
        self.identity: tuple[str, str, int] | None = None
        self.choices: set[int] = set()
        self.finished_choices: set[int] = set()
        self.tools: dict[tuple[int, int], _ChatTool] = {}
        self.items: dict[int, _Item] = {}
        self.sequence = -1
        self.reasoning = ChatReasoningTracker()
        self.reasoning_digests: set[str] = set()

    def frame(self, event: SseEvent) -> bool:
        """Drop comment keepalives and reject unknown SSE carriers before forwarding."""
        if (
            any(name not in {"", "event", "data"} for name, _ in event.lines)
            or sum(name == "event" for name, _ in event.lines) > 1
        ):
            raise InvalidReversalError("unsupported profile SSE metadata")
        if not event.data:
            if any(name != "" for name, _ in event.lines):
                raise InvalidReversalError("profile SSE event has no data")
            return False
        if self.terminal:
            raise InvalidReversalError("data after protocol completion")
        event.lines = [(name, value) for name, value in event.lines if name != ""]
        event.raw_lines.clear()
        return True

    def done(self, event: SseEvent) -> None:
        """A Chat sentinel is valid only after every choice and tool call has finished."""
        if (
            self.kind != "chat"
            or event.event != "message"
            or not self.choices
            or self.choices != self.finished_choices
        ):
            raise InvalidReversalError("premature protocol terminal")
        self.terminal = True
        self.reasoning_digests = self.reasoning.digests()

    def accept(self, event: SseEvent, payload: object) -> None:
        """Validate a native event and its lifecycle before the transformer sees it."""
        if self.kind == "chat":
            if event.event != "message":
                raise InvalidReversalError("unsupported Chat SSE event")
            self._chat(ChatChunk.model_validate(payload))
        else:
            self._responses(event, payload)

    def finish(self) -> None:
        """Transport EOF never substitutes for a complete provider protocol."""
        if not self.terminal:
            raise InvalidReversalError("profile stream ended without protocol completion")

    def _chat(self, chunk: ChatChunk) -> None:
        identity = (chunk.id, chunk.model, chunk.created)
        if self.identity is not None and (
            self.identity[:2] != identity[:2]
            or (not self.local_created_drift and self.identity[2] != identity[2])
        ):
            raise InvalidReversalError("Chat response identity changed")
        self.identity = self.identity or identity
        _unique([choice.index for choice in chunk.choices])
        if self._router_usage_echo(chunk):
            return
        if not chunk.choices and (
            chunk.usage is None or not self.choices or self.choices != self.finished_choices
        ):
            raise InvalidReversalError("unexpected usage-only chunk")
        for choice in chunk.choices:
            if choice.index in self.finished_choices:
                raise InvalidReversalError("choice continued after completion")
            self.choices.add(choice.index)
            self.reasoning.append(choice.index, choice.delta.model_dump(exclude_none=True))
            tools = choice.delta.tool_calls or []
            _unique([tool.index for tool in tools])
            for tool in tools:
                self._chat_tool(choice.index, tool)
            if choice.finish_reason is not None:
                calls = [tool for (index, _), tool in self.tools.items() if index == choice.index]
                if choice.finish_reason == "tool_calls" and not calls:
                    raise InvalidReversalError("tool completion has no calls")
                for call in calls:
                    call.arguments.finish()
                self.finished_choices.add(choice.index)
                self._finish_reasons[choice.index] = choice.finish_reason
        self._record_router_usage(chunk)

    def _router_usage_echo(self, chunk: ChatChunk) -> bool:
        # OpenRouter's final usage frame repeats finish reasons with no content.
        if not self.openrouter or not any(c.index in self.finished_choices for c in chunk.choices):
            return False
        if (
            self._router_usage_seen
            or chunk.usage is None
            or self.choices != self.finished_choices
            or {c.index for c in chunk.choices} != self.finished_choices
            or any(
                c.finish_reason != self._finish_reasons[c.index]
                or c.delta.content
                or c.delta.refusal
                or c.delta.tool_calls
                or any(getattr(c.delta, key) for key in CHAT_REASONING_FIELDS)
                for c in chunk.choices
            )
        ):
            raise InvalidReversalError("invalid OpenRouter usage completion")
        self._router_usage_seen = True
        return True

    def _record_router_usage(self, chunk: ChatChunk) -> None:
        if self.openrouter and chunk.usage is not None:
            if self._router_usage_seen or self.choices != self.finished_choices:
                raise InvalidReversalError("invalid OpenRouter usage completion")
            self._router_usage_seen = True

    def _chat_tool(self, choice_index: int, tool: ToolDelta) -> None:
        key = (choice_index, tool.index)
        name = tool.function.name if tool.function is not None else None
        state = self.tools.get(key)
        if state is None:
            if tool.id is None or name is None or tool.type != "function":
                raise InvalidReversalError("function delta has no identity")
            if any(previous.id == tool.id for previous in self.tools.values()):
                raise InvalidReversalError("duplicate function identity")
            state = self.tools[key] = _ChatTool(tool.id, name)
        elif tool.id not in {None, state.id} or name not in {None, state.name}:
            raise InvalidReversalError("function identity changed")
        if tool.function is not None and tool.function.arguments is not None:
            state.arguments.append(tool.function.arguments)

    def _responses(self, event: SseEvent, payload: object) -> None:
        parsed = RESPONSES_EVENT.validate_python(payload)
        if event.event not in {"message", parsed.type} or parsed.sequence_number <= self.sequence:
            raise InvalidReversalError("invalid Responses event sequence")
        self.sequence = parsed.sequence_number
        if isinstance(parsed, LifecycleEvent | MetadataEvent):
            self._lifecycle(parsed)
        elif self.identity is None:
            raise InvalidReversalError("Responses stream has no start")
        elif isinstance(parsed, ItemEvent):
            self._item(parsed)
        elif isinstance(parsed, ArgumentsDeltaEvent | ArgumentsDoneEvent):
            state = self._get_item(parsed)
            if not isinstance(state.initial, OutputFunction):
                raise InvalidReversalError("argument event does not name a function")
            if isinstance(parsed, ArgumentsDeltaEvent):
                state.arguments.append(parsed.delta)
            else:
                state.arguments.finish(parsed.arguments)
        elif isinstance(
            parsed,
            SummaryPartEvent
            | SummaryDeltaEvent
            | SummaryDoneEvent
            | ReasoningDeltaEvent
            | ReasoningDoneEvent,
        ):
            self._reasoning_part(parsed)
        else:
            self._part(parsed)

    def _metadata(self, event: MetadataEvent) -> None:
        if self.identity is None or event.response_id != self.identity[0]:
            raise InvalidReversalError("Responses metadata has no matching start")

    def _lifecycle(self, event: LifecycleEvent | MetadataEvent) -> None:
        if isinstance(event, MetadataEvent):
            self._metadata(event)
            return
        response = event.response
        identity = (response.id, response.model, response.created_at)
        if event.type == "response.created":
            if self.identity is not None:
                raise InvalidReversalError("duplicate Responses start")
            self.identity = identity
        elif self.identity != identity:
            raise InvalidReversalError("Responses identity changed or has no start")
        if event.type in {"response.created", "response.in_progress"}:
            if response.status != "in_progress" or response.output or response.incomplete_details:
                raise InvalidReversalError("invalid Responses start")
            return
        if response.status != event.type.removeprefix("response."):
            raise InvalidReversalError("Responses terminal status disagrees")
        _complete_response(response)
        # Native streams may complete without repeating their already closed items.
        if len(response.output) not in {0, len(self.items)} or any(
            not item.closed for item in self.items.values()
        ):
            raise InvalidReversalError("Responses completed with open output items")
        for index, item in enumerate(response.output):
            self.items[index].check(item)
        self.reasoning_digests = {
            item.encrypted_digest for item in self.items.values() if item.encrypted_digest
        }
        self.terminal = True

    def _item(self, event: ItemEvent) -> None:
        item = event.item
        if event.type == "response.output_item.added":
            if event.output_index != len(self.items) or any(
                state.initial.id == item.id for state in self.items.values()
            ):
                raise InvalidReversalError("duplicate or out-of-order output item")
            if item.status not in {None, "in_progress"} or (
                item.content
                if isinstance(item, OutputMessage)
                else (item.summary or item.content)
                if isinstance(item, OutputReasoning)
                else item.arguments
            ):
                raise InvalidReversalError("output item must start empty")
            self.items[event.output_index] = _Item(item)
        else:
            state = self.items.get(event.output_index)
            if state is None or state.closed:
                raise InvalidReversalError("output item has no open start")
            state.check(item)
            state.closed = True

    def _get_item(self, event: ItemCoordinate) -> _Item:
        state = self.items.get(event.output_index)
        if state is None or state.closed or state.initial.id != event.item_id:
            raise InvalidReversalError("event has no matching open output item")
        return state

    def _part(self, event: PartEvent | TextDeltaEvent | TextDoneEvent | RefusalDoneEvent) -> None:
        item = self._get_item(event)
        if not isinstance(item.initial, OutputMessage):
            raise InvalidReversalError("text event does not name a message")
        if isinstance(event, PartEvent) and event.type == "response.content_part.added":
            if event.content_index != len(item.parts) or _part_text(event.part):
                raise InvalidReversalError("content part must start once and empty")
            item.parts[event.content_index] = _Part(event.part.type)
            return
        part = item.parts.get(event.content_index)
        kind = (
            event.part.type
            if isinstance(event, PartEvent)
            else ("refusal" if event.type.startswith("response.refusal.") else "output_text")
        )
        if part is None or part.closed or part.kind != kind:
            raise InvalidReversalError("event has no matching open content part")
        if isinstance(event, PartEvent):
            if not part.text_done:
                raise InvalidReversalError("content part closed before text completion")
            part.check(_part_text(event.part))
            part.closed = True
        elif isinstance(event, TextDeltaEvent):
            part.append(event.delta)
        else:
            if part.text_done:
                raise InvalidReversalError("duplicate text completion")
            part.check(event.text if isinstance(event, TextDoneEvent) else event.refusal)
            part.text_done = True

    def _reasoning_part(
        self,
        event: SummaryPartEvent
        | SummaryDeltaEvent
        | SummaryDoneEvent
        | ReasoningDeltaEvent
        | ReasoningDoneEvent,
    ) -> None:
        item = self._get_item(event)
        if not isinstance(item.initial, OutputReasoning):
            raise InvalidReversalError("reasoning event does not name a reasoning item")
        if isinstance(event, ReasoningDeltaEvent | ReasoningDoneEvent):
            self._reasoning_text(item, event)
            return
        index = event.summary_index
        if isinstance(event, SummaryPartEvent) and event.type.endswith(".added"):
            if index != len(item.summaries) or event.part.text:
                raise InvalidReversalError("reasoning summary must start once and empty")
            item.summaries[index] = _Part("summary_text")
            return
        part = item.summaries.get(index)
        if part is None or part.closed:
            raise InvalidReversalError("reasoning summary has no open start")
        if isinstance(event, SummaryDeltaEvent):
            part.append(event.delta)
        elif isinstance(event, SummaryDoneEvent):
            if part.text_done:
                raise InvalidReversalError("duplicate reasoning text completion")
            part.check(event.text)
            part.text_done = True
        else:
            if not part.text_done:
                raise InvalidReversalError("reasoning summary closed before text completion")
            part.check(event.part.text)
            part.closed = True

    def _reasoning_text(self, item: _Item, event: ReasoningDeltaEvent | ReasoningDoneEvent) -> None:
        index = event.content_index
        if index not in item.parts:
            if index != len(item.parts):
                raise InvalidReversalError("out-of-order reasoning content")
            item.parts[index] = _Part("reasoning_text")
        part = item.parts[index]
        if isinstance(event, ReasoningDeltaEvent):
            part.append(event.delta)
        else:
            if part.closed:
                raise InvalidReversalError("duplicate reasoning completion")
            part.check(event.text)
            part.closed = part.text_done = True
