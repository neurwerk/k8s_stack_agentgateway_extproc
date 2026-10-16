"""Inspection-owned content shapes with bounded, extensible provider metadata."""

from typing import Annotated, Literal

from neurwerk_request_segments.models import (
    EngineResponseTextConfig,
    EngineResponseToolDefinition,
    ResponseMessagePhase,
)
from neurwerk_request_segments.reasoning import ChatReasoning, ResponseReasoning, SummaryText
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

type Identifier = Annotated[str, Field(min_length=1, max_length=256)]
type Count = Annotated[int, Field(ge=0)]
type OutputIndex = Annotated[int, Field(ge=0, le=255)]
type ChoiceIndex = Annotated[int, Field(ge=0, le=15)]
type FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]
type ItemStatus = Literal["in_progress", "completed", "incomplete"]
type EmptyList = Annotated[list[None], Field(max_length=0)]
type MetadataToken = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$", max_length=64)]
type ControlMetadata = Annotated[dict[MetadataToken, MetadataToken | None], Field(max_length=64)]
type UsageMetadata = Annotated[
    dict[MetadataToken, Annotated[int | float, Field(ge=0)] | bool | UsageMetadata | None],
    Field(max_length=128),
]


class ResponseModel(BaseModel):
    """Validate without coercion or unreviewed extension fields."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class MetadataEnvelope(ResponseModel):
    """Allow new numeric/null/boolean facts only outside content carriers.

    Strings and containers require an explicitly classified field. Original JSON
    is forwarded, not this validation view; native AgentGateway owns accounting.
    """

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, int | float | bool | None] = Field(init=False)


class Function(ResponseModel):
    """A complete client function call."""

    name: Identifier
    arguments: str


class ToolCall(ResponseModel):
    """Only client function tools have qualified restoration."""

    id: Identifier
    type: Literal["function"]
    function: Function


class ChatMessage(ResponseModel, ChatReasoning):
    """An assistant's answer, reasoning and client tool calls."""

    role: Literal["assistant"]
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, max_length=128)
    annotations: EmptyList | None = None


class ChatChoice(ResponseModel):
    """One complete Chat choice."""

    index: ChoiceIndex
    message: ChatMessage
    finish_reason: FinishReason
    logprobs: None = None
    native_finish_reason: Identifier | None = None


class ChatEnvelope(MetadataEnvelope):
    """Stable Chat envelope fields, shared by JSON responses and streaming chunks."""

    id: Identifier
    created: Count
    model: Identifier
    system_fingerprint: Identifier | None = None
    service_tier: MetadataToken | None = None
    usage: UsageMetadata | None = None
    provider: Identifier | None = None


class ChatResponse(ChatEnvelope):
    """Complete Chat Completions response."""

    object: Literal["chat.completion"]
    choices: list[ChatChoice] = Field(min_length=1, max_length=16)


class FunctionDelta(ResponseModel):
    """A streamed function identity or argument fragment."""

    name: Identifier | None = None
    arguments: str | None = None


class ToolDelta(ResponseModel):
    """A function-call fragment with stable choice-local index."""

    index: Annotated[int, Field(ge=0, le=127)]
    id: Identifier | None = None
    type: Literal["function"] | None = None
    function: FunctionDelta | None = None


class ChatDelta(ResponseModel, ChatReasoning):
    """Qualified Chat stream fields only."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolDelta] | None = Field(default=None, max_length=128)


class ChatChunkChoice(ResponseModel):
    """One streaming choice fragment or its explicit finish."""

    index: ChoiceIndex
    delta: ChatDelta
    finish_reason: FinishReason | None
    logprobs: None = None
    native_finish_reason: Identifier | None = None


class ChatChunk(ChatEnvelope):
    """Chat streaming response, including the optional final usage-only chunk."""

    object: Literal["chat.completion.chunk"]
    choices: list[ChatChunkChoice] = Field(max_length=16)
    obfuscation: str | None = Field(default=None, max_length=1024)


class OutputText(ResponseModel):
    """Output text; citations and raw log-probability tokens need separate qualification."""

    type: Literal["output_text"]
    text: str
    annotations: EmptyList
    logprobs: EmptyList | None = None


class Refusal(ResponseModel):
    """A textual provider refusal, subject to the same restoration as other text."""

    type: Literal["refusal"]
    refusal: str


type OutputPart = Annotated[OutputText | Refusal, Field(discriminator="type")]


class OutputMessage(ResponseModel):
    """A native Responses assistant message."""

    type: Literal["message"]
    id: Identifier
    role: Literal["assistant"]
    status: ItemStatus
    phase: ResponseMessagePhase | None = None
    content: list[OutputPart] = Field(max_length=64)


class OutputFunction(ResponseModel):
    """A native Responses client function call."""

    type: Literal["function_call"]
    id: Identifier
    call_id: Identifier
    name: Identifier
    arguments: str
    status: ItemStatus | None = None


class OutputReasoning(ResponseReasoning):
    """A response-side reasoning item has an identity for stream integrity."""

    id: Identifier


type OutputItem = Annotated[
    OutputMessage | OutputFunction | OutputReasoning, Field(discriminator="type")
]


class IncompleteDetails(ResponseModel):
    """Qualified protocol completion reasons for an incomplete generation."""

    reason: Literal["max_output_tokens", "content_filter"]


class FunctionChoice(ResponseModel):
    """A reflected client function selection."""

    type: Literal["function"]
    name: Identifier


class AllowedTools(ResponseModel):
    """A reflected bounded selection of qualified client functions."""

    type: Literal["allowed_tools"]
    mode: Literal["auto", "required"]
    tools: list[FunctionChoice] = Field(min_length=1, max_length=128)


class ReflectedFunctionTool(EngineResponseToolDefinition):
    """Admit the provider's empty output-schema reflection only on responses."""

    output_schema: None = None


class ResponsesResponse(MetadataEnvelope):
    """Validate output and reflected content without mirroring provider accounting."""

    id: Identifier
    object: Literal["response"]
    created_at: Count
    completed_at: Count | None = None
    access_programs: ControlMetadata | None = None
    model: Identifier
    status: ItemStatus
    output: list[OutputItem] = Field(max_length=256)
    error: None = None
    incomplete_details: IncompleteDetails | None = None
    usage: UsageMetadata | None = None
    tool_usage: UsageMetadata | None = None
    background: Literal[False] | None = None
    instructions: str | None = None
    previous_response_id: None = None
    reasoning: ControlMetadata | None = None
    reasoning_effort: MetadataToken | None = None
    service_tier: MetadataToken | None = None
    text: EngineResponseTextConfig | None = None
    tool_choice: Literal["auto", "none", "required"] | FunctionChoice | AllowedTools | None = None
    tools: list[ReflectedFunctionTool] | None = Field(default=None, max_length=128)
    top_logprobs: Literal[0] | None = None
    truncation: MetadataToken | None = None
    user: str | None = Field(default=None, max_length=256)
    safety_identifier: str | None = Field(default=None, max_length=256)
    prompt_cache_key: str | None = Field(default=None, max_length=256)
    prompt_cache_retention: Identifier | None = None
    metadata: (
        Annotated[
            dict[Annotated[str, Field(max_length=64)], Annotated[str, Field(max_length=512)]],
            Field(max_length=64),
        ]
        | None
    ) = None


class ResponseEvent(ResponseModel):
    """Every Responses SSE event carries a monotonic sequence number."""

    sequence_number: Count


class LifecycleEvent(ResponseEvent):
    """Only explicit, non-error response lifecycles are supported."""

    type: Literal[
        "response.created", "response.in_progress", "response.completed", "response.incomplete"
    ]
    response: ResponsesResponse


class ItemEvent(ResponseEvent):
    """Open or close one qualified output item."""

    type: Literal["response.output_item.added", "response.output_item.done"]
    output_index: OutputIndex
    item: OutputItem


class ModerationResult(ResponseModel):
    """Moderation categories carry boolean flags, never model text."""

    category_flags: dict[Identifier, bool]


class ModerationOutput(ResponseModel):
    """One native moderation decision."""

    results: list[ModerationResult]
    is_blocked: bool


class OmniModeration(ResponseModel):
    """The observed prompt moderation envelope."""

    protection_type: Identifier
    outputs: list[ModerationOutput]


class PromptModeration(ResponseModel):
    """The qualified prompt moderation classifier."""

    omnimod: OmniModeration


class ChatGPTModeration(ResponseModel):
    """Only the observed prompt facts and empty later-stage envelopes are supported."""

    prompt: PromptModeration
    generation: Annotated[dict[str, None], Field(max_length=0)]
    tool_call: Annotated[dict[str, None], Field(max_length=0)]
    tool_response: Annotated[dict[str, None], Field(max_length=0)]


class ModerationMetadata(ResponseModel):
    """Explicitly classified native moderation metadata."""

    openai_chatgpt_moderation_metadata: ChatGPTModeration


class MetadataEvent(ResponseEvent):
    """Response-scoped moderation facts, separate from output content."""

    type: Literal["response.metadata"]
    response_id: Identifier
    metadata: ModerationMetadata


class ItemCoordinate(ResponseEvent):
    """Coordinate an event with its already opened output item."""

    item_id: Identifier
    output_index: OutputIndex


class PartCoordinate(ItemCoordinate):
    """Coordinate a text or refusal part."""

    content_index: Annotated[int, Field(ge=0, le=63)]


class PartEvent(PartCoordinate):
    """Open or close one text/refusal part."""

    type: Literal["response.content_part.added", "response.content_part.done"]
    part: OutputPart


class TextDeltaEvent(PartCoordinate):
    """Stream an ordinary text fragment."""

    type: Literal["response.output_text.delta", "response.refusal.delta"]
    delta: str
    logprobs: EmptyList | None = None
    obfuscation: str | None = Field(default=None, max_length=1024)


class TextDoneEvent(PartCoordinate):
    """Close output text with its full snapshot."""

    type: Literal["response.output_text.done"]
    text: str
    logprobs: EmptyList | None = None


class RefusalDoneEvent(PartCoordinate):
    """Close a refusal with its full snapshot."""

    type: Literal["response.refusal.done"]
    refusal: str


class ArgumentsDeltaEvent(ItemCoordinate):
    """Stream function arguments, which must close as a valid JSON object."""

    type: Literal["response.function_call_arguments.delta"]
    delta: str
    obfuscation: str | None = Field(default=None, max_length=1024)


class ArgumentsDoneEvent(ItemCoordinate):
    """Close function arguments with their complete snapshot."""

    type: Literal["response.function_call_arguments.done"]
    arguments: str


class SummaryCoordinate(ItemCoordinate):
    """Coordinate a reasoning summary with its open output item."""

    summary_index: Annotated[int, Field(ge=0, le=63)]


class SummaryPartEvent(SummaryCoordinate):
    """Open or close one reasoning summary part."""

    type: Literal["response.reasoning_summary_part.added", "response.reasoning_summary_part.done"]
    part: SummaryText


class SummaryDeltaEvent(SummaryCoordinate):
    """Stream readable reasoning, separate from final-answer text."""

    type: Literal["response.reasoning_summary_text.delta"]
    delta: str
    obfuscation: str | None = Field(default=None, max_length=1024)


class SummaryDoneEvent(SummaryCoordinate):
    """Close a reasoning summary with its full snapshot."""

    type: Literal["response.reasoning_summary_text.done"]
    text: str


class ReasoningDeltaEvent(PartCoordinate):
    """Stream Responses reasoning content."""

    type: Literal["response.reasoning_text.delta"]
    delta: str
    obfuscation: str | None = Field(default=None, max_length=1024)


class ReasoningDoneEvent(PartCoordinate):
    """Close Responses reasoning content."""

    type: Literal["response.reasoning_text.done"]
    text: str


RESPONSES_EVENT = TypeAdapter(
    Annotated[
        LifecycleEvent
        | MetadataEvent
        | ItemEvent
        | PartEvent
        | TextDeltaEvent
        | TextDoneEvent
        | RefusalDoneEvent
        | ArgumentsDeltaEvent
        | ArgumentsDoneEvent
        | SummaryPartEvent
        | SummaryDeltaEvent
        | SummaryDoneEvent
        | ReasoningDeltaEvent
        | ReasoningDoneEvent,
        Field(discriminator="type"),
    ]
)
