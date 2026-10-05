"""Authoritative supported provider shapes and reviewed protocol controls."""

from __future__ import annotations

import math
from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationInfo,
    model_validator,
)
from pydantic.json_schema import JsonDict, SkipJsonSchema

from .compatibility import CompatibilitySettings, UnsupportedFeatureError

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type McpJsonValue = (
    Annotated[str, Field(strict=True)]
    | Annotated[int, Field(strict=True)]
    | Annotated[float, Field(strict=True, allow_inf_nan=False)]
    | Annotated[bool, Field(strict=True)]
    | Annotated[list[McpJsonValue], Field(max_length=256)]
    | Annotated[dict[str, McpJsonValue], Field(max_length=256)]
    | None
)
type McpRequestId = (
    Annotated[str, Field(strict=True, min_length=1, max_length=256)]
    | Annotated[int, Field(strict=True, ge=-9_007_199_254_740_991, le=9_007_199_254_740_991)]
)

# Existing provider controls are admitted separately from content. Complex controls
# have typed fields below; extension rules may only introduce scalar controls.
_CONTROLS: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {
    ("chat", "request"): {
        "store": ("boolean", "null"),
        "max_completion_tokens": ("integer", "null"),
        "frequency_penalty": ("number", "null"),
        "presence_penalty": ("number", "null"),
        "seed": ("integer", "null"),
        "logprobs": ("boolean", "null"),
        "top_logprobs": ("integer", "null"),
        "parallel_tool_calls": ("boolean", "null"),
        "reasoning_effort": ("string", "null"),
        "service_tier": ("string", "null"),
        "safety_identifier": ("string", "null"),
        "prompt_cache_key": ("string", "null"),
        "prompt_cache_retention": ("string", "null"),
        "verbosity": ("string", "null"),
    },
    ("responses", "request"): {
        "store": ("boolean", "null"),
        "background": ("boolean", "null"),
        "parallel_tool_calls": ("boolean", "null"),
        "max_tool_calls": ("integer", "null"),
        "top_logprobs": ("integer", "null"),
        "truncation": ("string", "null"),
        "service_tier": ("string", "null"),
        "safety_identifier": ("string", "null"),
        "prompt_cache_key": ("string", "null"),
        "prompt_cache_retention": ("string", "null"),
    },
    ("chat", "function"): {"strict": ("boolean", "null")},
    ("responses", "tool"): {"strict": ("boolean", "null")},
    ("chat", "stream_options"): {"include_obfuscation": ("boolean",)},
    ("responses", "stream_options"): {"include_obfuscation": ("boolean",)},
    ("responses", "message"): {"id": ("string",), "status": ("string",)},
    ("responses", "input_item"): {"id": ("string",), "status": ("string",)},
}


def _omit_none_default(schema: JsonDict) -> None:
    """MCP optional object fields advertise omission, never an explicit null."""
    schema.pop("default", None)


class RequestModel(BaseModel):
    """Preserve explicit fields and validate all additional controls."""

    model_config = ConfigDict(
        extra="allow",
        strict=True,
        validate_assignment=True,
        serialize_by_alias=True,
        allow_inf_nan=False,
    )
    endpoint: ClassVar[str] = ""
    location: ClassVar[str] = ""
    _controls: CompatibilitySettings | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def validate_controls(cls, value: object, info: ValidationInfo) -> object:
        """Reject unreviewed fields before constructing the request model."""
        if not isinstance(value, dict):
            return value
        if cls.location == "request":
            _validate_protocol_controls(value, cls.endpoint)
        fields = {field.alias or name for name, field in cls.model_fields.items()}
        known = _CONTROLS.get((cls.endpoint, cls.location), {})
        settings = info.context.get("controls") if isinstance(info.context, dict) else None
        for key in value.keys() - fields:
            if key in known:
                kinds = known[key]
                kind = {
                    str: "string",
                    bool: "boolean",
                    int: "integer",
                    float: "number",
                    type(None): "null",
                }.get(type(value[key]))
                if kind not in kinds and not (kind == "integer" and "number" in kinds):
                    raise ValueError("invalid control value kind")  # noqa: TRY003
                if isinstance(value[key], float) and not math.isfinite(value[key]):
                    raise ValueError("non-finite control value")  # noqa: TRY003
                continue
            rule = (
                next(
                    (
                        rule
                        for rule in settings.controls
                        if (rule.endpoint, rule.location, rule.field)
                        == (cls.endpoint, cls.location, key)
                    ),
                    None,
                )
                if isinstance(settings, CompatibilitySettings)
                else None
            )
            if rule is None:
                raise UnsupportedFeatureError(cls.location, str(key))
            if not rule.accepts(value[key]):
                raise ValueError("invalid reviewed control value")  # noqa: TRY003
        return value


class EngineTextPart(RequestModel):
    """Chat textual content."""

    type: Literal["text"]
    text: str


class EngineAttachmentPart(BaseModel):
    """Opaque attachment data handled by the adapter's attachment boundary."""

    model_config = ConfigDict(extra="allow", strict=True, validate_assignment=True)
    type: Literal[
        "image_url",
        "input_audio",
        "file",
        "input_image",
        "input_file",
        "image",
        "audio",
        "resource",
        "resource_link",
    ]


type EngineMessageContent = (
    str | Annotated[list[EngineTextPart | EngineAttachmentPart], Field(max_length=64)]
)


class EngineFunction(RequestModel):
    """A function invocation; arguments contain independently analyzed strings."""

    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    arguments: JsonValue = ""


class EngineToolCall(RequestModel):
    """Chat tool call."""

    id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    type: Literal["function"]
    function: EngineFunction


class EngineToolFunction(RequestModel):
    """Chat function definition; parameter schemas intentionally remain open."""

    endpoint = "chat"
    location = "function"
    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    description: str | None = Field(default=None, max_length=20_000)
    parameters: dict[str, JsonValue] | None = None


class EngineToolDefinition(RequestModel):
    """Nested Chat function tool definition."""

    endpoint = "chat"
    location = "tool"
    type: Literal["function"]
    function: EngineToolFunction


class EngineResponseToolDefinition(RequestModel):
    """Responses uses flat function definitions, unlike Chat."""

    endpoint = "responses"
    location = "tool"
    type: Literal["function"]
    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    description: str | None = Field(default=None, max_length=20_000)
    parameters: dict[str, JsonValue] | None = None


class EngineMessage(RequestModel):
    """Chat message with current and deprecated function-call forms."""

    endpoint = "chat"
    location = "message"
    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: EngineMessageContent | None = None
    name: str | None = Field(default=None, max_length=256)
    tool_calls: list[EngineToolCall] | None = Field(default_factory=list, max_length=32)
    tool_call_id: str | None = Field(default=None, max_length=256)
    function_call: EngineFunction | None = None
    refusal: str | None = None

    @model_validator(mode="after")
    def validate_role_fields(self) -> EngineMessage:
        """Require role-specific call/result identities."""
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")  # noqa: TRY003
        if (self.tool_calls or self.function_call) and self.role != "assistant":
            raise ValueError("tool calls require an assistant message")  # noqa: TRY003
        if (
            self.role == "assistant"
            and self.content is None
            and not (self.tool_calls or self.function_call or self.refusal)
        ):
            raise ValueError("assistant messages require content or tool calls")  # noqa: TRY003
        return self


class EngineChatStreamOptions(RequestModel):
    """Chat stream controls."""

    endpoint = "chat"
    location = "stream_options"
    include_usage: bool | None = None


class EngineChatRequest(RequestModel):
    """Supported Chat body, retaining omitted/null/false distinctions."""

    endpoint = "chat"
    location = "request"
    model: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_./:-]+$")
    messages: list[EngineMessage] = Field(min_length=1, max_length=256)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool | None = False
    stream_options: EngineChatStreamOptions | None = None
    n: int | None = Field(default=None, ge=1, le=16)
    stop: str | list[str] | None = None
    tools: list[EngineToolDefinition] | None = Field(default_factory=list, max_length=128)
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    response_format: dict[str, JsonValue] | None = None
    user: str | None = Field(default=None, max_length=256)
    metadata: (
        Annotated[
            dict[Annotated[str, Field(max_length=64)], Annotated[str, Field(max_length=512)]],
            Field(max_length=64),
        ]
        | None
    ) = None
    logit_bias: dict[str, float] | None = None
    functions: list[EngineToolFunction] | None = None
    function_call: Literal["none", "auto"] | dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_stream_options(self) -> EngineChatRequest:
        """Require streaming when stream options are present."""
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options require stream to be enabled")  # noqa: TRY003
        return self


class EngineResponseTextPart(RequestModel):
    """Responses text part."""

    type: Literal["input_text", "output_text"]
    text: str


class EngineResponseMessage(RequestModel):
    """Responses message, including the short string-content form."""

    endpoint = "responses"
    location = "message"
    type: Literal["message"] = "message"
    role: Literal["system", "developer", "user", "assistant"]
    content: (
        str
        | Annotated[
            list[EngineResponseTextPart | EngineAttachmentPart], Field(min_length=1, max_length=64)
        ]
    )


class EngineResponseFunctionCall(RequestModel):
    """Responses function invocation."""

    endpoint = "responses"
    location = "input_item"
    type: Literal["function_call"]
    call_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    arguments: JsonValue


class EngineResponseFunctionOutput(RequestModel):
    """Responses function output JSON values."""

    endpoint = "responses"
    location = "input_item"
    type: Literal["function_call_output"]
    call_id: str = Field(min_length=1, max_length=256)
    output: JsonValue


class EngineResponseTextFormatText(RequestModel):
    """Ordinary text output."""

    type: Literal["text"]


class EngineResponseTextFormatObject(RequestModel):
    """JSON object output."""

    type: Literal["json_object"]


class EngineResponseTextFormatSchema(RequestModel):
    """Structured Responses output with an intentionally open JSON schema."""

    endpoint = "responses"
    location = "text_format"
    type: Literal["json_schema"]
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = Field(default=None, max_length=4_000)
    schema_value: dict[str, JsonValue] = Field(alias="schema", max_length=256)
    strict: bool | None = None


type EngineResponseTextFormat = Annotated[
    EngineResponseTextFormatText | EngineResponseTextFormatObject | EngineResponseTextFormatSchema,
    Field(discriminator="type"),
]


class EngineResponseTextConfig(RequestModel):
    """Responses text output controls."""

    endpoint = "responses"
    location = "text"
    format: EngineResponseTextFormat | None = None
    verbosity: Literal["low", "medium", "high"] | None = None


class ReasoningConfig(RequestModel):
    """Non-content Responses reasoning controls."""

    effort: str | None = None
    summary: str | None = None
    generate_summary: str | None = None


class ResponseStreamOptions(RequestModel):
    """Responses streaming controls."""

    endpoint = "responses"
    location = "stream_options"


type EngineResponseInputItem = (
    EngineResponseMessage | EngineResponseFunctionCall | EngineResponseFunctionOutput
)
type EngineResponseInput = (
    str | Annotated[list[EngineResponseInputItem], Field(min_length=1, max_length=256)]
)


class EngineResponsesRequest(RequestModel):
    """Supported Responses body."""

    endpoint = "responses"
    location = "request"
    model: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_./:-]+$")
    input: EngineResponseInput
    instructions: str | None = None
    tools: list[EngineResponseToolDefinition] | None = Field(default_factory=list, max_length=128)
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    text: EngineResponseTextConfig | None = None
    stream: bool | None = False
    stream_options: ResponseStreamOptions | None = None
    previous_response_id: str | None = Field(default=None, max_length=256)
    user: str | None = Field(default=None, max_length=256)
    metadata: (
        Annotated[
            dict[Annotated[str, Field(max_length=64)], Annotated[str, Field(max_length=512)]],
            Field(max_length=64),
        ]
        | None
    ) = None
    reasoning: ReasoningConfig | None = None
    include: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_uninspected_history(cls, value: object) -> object:
        """Provider-stored prior input has no request-bound inspection provenance."""
        if isinstance(value, dict) and value.get("previous_response_id") is not None:
            raise UnsupportedFeatureError("request", "previous_response_id")
        return value


class EngineMcpParams(RequestModel):
    """MCP arguments and immutable metadata."""

    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    arguments: Annotated[dict[str, McpJsonValue], Field(max_length=256)] | SkipJsonSchema[None] = (
        Field(
            default=None,
            exclude_if=lambda value: value is None,
            json_schema_extra=_omit_none_default,
        )
    )
    meta: (
        Annotated[dict[Annotated[str, Field(max_length=256)], McpJsonValue], Field(max_length=64)]
        | SkipJsonSchema[None]
    ) = Field(
        default=None,
        alias="_meta",
        exclude_if=lambda value: value is None,
        json_schema_extra=_omit_none_default,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_optional_objects(cls, value: object) -> object:
        """MCP optional objects cannot be explicit null."""
        if isinstance(value, dict) and any(
            key in value and value[key] is None for key in ("arguments", "_meta")
        ):
            raise ValueError("optional MCP params must be objects when present")  # noqa: TRY003
        return value


class EngineMcpRequest(RequestModel):
    """Supported MCP tools/call request."""

    jsonrpc: Literal["2.0"]
    id: McpRequestId
    method: Literal["tools/call"]
    params: EngineMcpParams


type SupportedRequest = EngineChatRequest | EngineResponsesRequest | EngineMcpRequest
type EngineRequest = SupportedRequest
ENGINE_REQUEST_ADAPTER: TypeAdapter[SupportedRequest] = TypeAdapter(SupportedRequest)


def reserved_fields(endpoint: str, location: str) -> set[str]:
    """Reserve every existing shape field, even on alternate endpoint branches."""
    result = set(_CONTROLS.get((endpoint, location), {})) | {
        "audio",
        "prediction",
        "modalities",
        "prompt",
        "conversation",
        "context_management",
        "reasoning_content",
        "reasoning_signature",
        "reasoning",
        "encrypted_content",
        "annotations",
        "content",
        "messages",
        "input",
        "output",
        "arguments",
        "tools",
    }
    for model in RequestModel.__subclasses__():
        # Content names cannot become compatibility controls on another location.
        result.update(field.alias or name for name, field in model.model_fields.items())
    return result


def _closed_keys(value: dict[str, object], allowed: set[str], location: str) -> None:
    if value.keys() - allowed:
        raise UnsupportedFeatureError(location, "protocol_field")


def _function_choice(value: object, endpoint: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("function choice must be an object")  # noqa: TRY003, TRY004
    _closed_keys(
        value, {"type", "function"} if endpoint == "chat" else {"type", "name"}, "tool_choice"
    )
    if value.get("type") != "function":
        raise UnsupportedFeatureError("tool_choice", "type")
    function = value.get("function") if endpoint == "chat" else {"name": value.get("name")}
    _function_name(function)


def _function_name(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("function selector must be an object")  # noqa: TRY003, TRY004
    _closed_keys(value, {"name"}, "tool_choice")
    name = value.get("name")
    if not isinstance(name, str) or not 1 <= len(name) <= 256:
        raise ValueError("function selector requires a bounded name")  # noqa: TRY003


def _validate_protocol_controls(value: dict[str, object], endpoint: str) -> None:
    """Close protocol envelopes while keeping actual user schemas open."""
    choice = value.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") == "allowed_tools":
            _allowed_tools_choice(choice, endpoint)
        else:
            _function_choice(choice, endpoint)
    if endpoint == "chat":
        if isinstance(value.get("function_call"), dict):
            _function_name(value["function_call"])
        response_format = value.get("response_format")
        if isinstance(response_format, dict):
            _chat_response_format(response_format)


def _allowed_tools_choice(choice: dict[str, object], endpoint: str) -> None:
    allowed = choice
    if endpoint == "chat":
        _closed_keys(choice, {"type", "allowed_tools"}, "tool_choice")
        nested = choice.get("allowed_tools")
        if not isinstance(nested, dict):
            raise ValueError("allowed tools require an object")  # noqa: TRY003
        allowed = nested
        _closed_keys(allowed, {"mode", "tools"}, "tool_choice")
    else:
        _closed_keys(choice, {"type", "mode", "tools"}, "tool_choice")
    if allowed.get("mode") not in {"auto", "required"}:
        raise ValueError("invalid allowed-tools mode")  # noqa: TRY003
    tools = allowed.get("tools")
    if not isinstance(tools, list) or not 1 <= len(tools) <= 128:
        raise ValueError("allowed tools must be a bounded list")  # noqa: TRY003
    for tool in tools:
        _function_choice(tool, endpoint)


def _chat_response_format(value: dict[str, object]) -> None:
    kind = value.get("type")
    if kind in {"text", "json_object"}:
        _closed_keys(value, {"type"}, "response_format")
    elif kind == "json_schema":
        _closed_keys(value, {"type", "json_schema"}, "response_format")
        schema = value.get("json_schema")
        if not isinstance(schema, dict):
            raise ValueError("response format requires a schema envelope")  # noqa: TRY003
        _closed_keys(schema, {"name", "description", "schema", "strict"}, "response_format")
        name = schema.get("name")
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= 64
            or not isinstance(schema.get("schema"), dict)
        ):
            raise ValueError("invalid response schema envelope")  # noqa: TRY003
        if schema.get("strict") is not None and type(schema["strict"]) is not bool:
            raise ValueError("schema strict must be boolean or null")  # noqa: TRY003
        description = schema.get("description")
        if description is not None and (
            not isinstance(description, str) or len(description) > 4_000
        ):
            raise ValueError("schema description must be bounded text")  # noqa: TRY003
    else:
        raise UnsupportedFeatureError("response_format", "type")
