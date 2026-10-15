"""Closed v5 OpenAI text/function response shapes, independent of legacy parsing."""

from typing import Annotated, Literal

from neurwerk_request_segments.models import EngineResponseTextConfig, EngineResponseToolDefinition
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

type Identifier = Annotated[str, Field(min_length=1, max_length=256)]
type Count = Annotated[int, Field(ge=0)]
type OutputIndex = Annotated[int, Field(ge=0, le=255)]
type ChoiceIndex = Annotated[int, Field(ge=0, le=15)]
type FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]
type ItemStatus = Literal["in_progress", "completed", "incomplete"]
type ServiceTier = Literal["auto", "default", "flex", "scale", "priority", "fast"]
type EmptyList = Annotated[list[None], Field(max_length=0)]


class ResponseModel(BaseModel):
    """Validate without coercion or unreviewed extension fields."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PromptTokens(ResponseModel):
    """Known prompt-token breakdown."""

    audio_tokens: Count | None = None
    cached_tokens: Count | None = None
    cache_write_tokens: Count | None = None
    image_tokens: Count | None = None
    text_tokens: Count | None = None
    video_tokens: Count | None = None


class CompletionTokens(ResponseModel):
    """Known completion-token breakdown, containing counts rather than reasoning content."""

    accepted_prediction_tokens: Count | None = None
    audio_tokens: Count | None = None
    reasoning_tokens: Count | None = None
    rejected_prediction_tokens: Count | None = None
    text_tokens: Count | None = None
    image_tokens: Count | None = None


class ChatCost(ResponseModel):
    """Reported OpenRouter cost metadata, without inferring accounting completeness."""

    upstream_inference_cost: float | None = Field(default=None, ge=0)
    upstream_inference_prompt_cost: float = Field(ge=0)
    upstream_inference_completions_cost: float = Field(ge=0)


class ChatUsage(ResponseModel):
    """Preserve provider usage facts without inferring completeness."""

    prompt_tokens: Count
    completion_tokens: Count
    total_tokens: Count
    prompt_tokens_details: PromptTokens | None = None
    completion_tokens_details: CompletionTokens | None = None
    cost: float | None = Field(default=None, ge=0)
    is_byok: bool | None = None
    cost_details: ChatCost | None = None


class Function(ResponseModel):
    """A complete client function call."""

    name: Identifier
    arguments: str


class ToolCall(ResponseModel):
    """Only client function tools have qualified restoration."""

    id: Identifier
    type: Literal["function"]
    function: Function


class ChatMessage(ResponseModel):
    """An assistant's text, refusal or client tool calls; opaque reasoning is excluded."""

    role: Literal["assistant"]
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, max_length=128)
    annotations: EmptyList | None = None
    reasoning: None = None


class ChatChoice(ResponseModel):
    """One complete Chat choice."""

    index: ChoiceIndex
    message: ChatMessage
    finish_reason: FinishReason
    logprobs: None = None
    native_finish_reason: Identifier | None = None


class ChatEnvelope(ResponseModel):
    """Stable Chat envelope fields, shared by JSON responses and streaming chunks."""

    id: Identifier
    created: Count
    model: Identifier
    system_fingerprint: Identifier | None = None
    service_tier: ServiceTier | None = None
    usage: ChatUsage | None = None
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


class ChatDelta(ResponseModel):
    """Qualified Chat stream fields only."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolDelta] | None = Field(default=None, max_length=128)
    reasoning: None = None


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
    content: list[OutputPart] = Field(max_length=64)


class OutputFunction(ResponseModel):
    """A native Responses client function call."""

    type: Literal["function_call"]
    id: Identifier
    call_id: Identifier
    name: Identifier
    arguments: str
    status: ItemStatus | None = None


type OutputItem = Annotated[OutputMessage | OutputFunction, Field(discriminator="type")]


class InputTokens(ResponseModel):
    """Responses cached input count."""

    cached_tokens: Count


class OutputTokens(ResponseModel):
    """Responses reasoning count, never opaque reasoning content."""

    reasoning_tokens: Count


class ResponsesUsage(ResponseModel):
    """Native Responses usage fields."""

    input_tokens: Count
    output_tokens: Count
    total_tokens: Count
    input_tokens_details: InputTokens | None = None
    output_tokens_details: OutputTokens | None = None


class IncompleteDetails(ResponseModel):
    """Qualified protocol completion reasons for an incomplete generation."""

    reason: Literal["max_output_tokens", "content_filter"]


class ReasoningControls(ResponseModel):
    """Non-content controls are distinct from unsupported opaque output items."""

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None
    generate_summary: Literal["auto", "concise", "detailed"] | None = None


class FunctionChoice(ResponseModel):
    """A reflected client function selection."""

    type: Literal["function"]
    name: Identifier


class AllowedTools(ResponseModel):
    """A reflected bounded selection of qualified client functions."""

    type: Literal["allowed_tools"]
    mode: Literal["auto", "required"]
    tools: list[FunctionChoice] = Field(min_length=1, max_length=128)


class ResponsesResponse(ResponseModel):
    """Closed Responses envelope with qualified output and reflected request controls."""

    id: Identifier
    object: Literal["response"]
    created_at: Count
    completed_at: Count | None = None
    access_programs: None = None
    model: Identifier
    status: ItemStatus
    output: list[OutputItem] = Field(max_length=256)
    error: None = None
    incomplete_details: IncompleteDetails | None = None
    usage: ResponsesUsage | None = None
    background: Literal[False] | None = None
    instructions: str | None = None
    max_output_tokens: Annotated[int, Field(ge=1)] | None = None
    max_tool_calls: Annotated[int, Field(ge=1)] | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: None = None
    reasoning: ReasoningControls | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    service_tier: ServiceTier | None = None
    store: bool | None = None
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    text: EngineResponseTextConfig | None = None
    tool_choice: Literal["auto", "none", "required"] | FunctionChoice | AllowedTools | None = None
    tools: list[EngineResponseToolDefinition] | None = Field(default=None, max_length=128)
    top_logprobs: Literal[0] | None = None
    top_p: Annotated[float, Field(ge=0, le=1)] | None = None
    truncation: Literal["disabled", "auto"] | None = None
    user: str | None = Field(default=None, max_length=256)
    safety_identifier: str | None = Field(default=None, max_length=256)
    prompt_cache_key: str | None = Field(default=None, max_length=256)
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
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


RESPONSES_EVENT = TypeAdapter(
    Annotated[
        LifecycleEvent
        | ItemEvent
        | PartEvent
        | TextDeltaEvent
        | TextDoneEvent
        | RefusalDoneEvent
        | ArgumentsDeltaEvent
        | ArgumentsDoneEvent,
        Field(discriminator="type"),
    ]
)
