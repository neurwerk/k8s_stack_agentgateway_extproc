"""Closed native text/function-tool requests; unsupported content never passes through."""

# Preserve native REST field spelling throughout validation and reconstruction.
# ruff: noqa: N815

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, model_validator

from .compatibility import CompatibilitySettings, UnsupportedFeatureError

type Name = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
type NativeModelId = Annotated[
    str, Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_./:-]+$")
]
type JsonObject = Annotated[dict[str, JsonValue], Field(max_length=256)]


class NativeModel(BaseModel):
    """Keep native field names and omitted controls, with no compatibility escape hatch."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def reject_unknown(cls, value: object) -> object:
        """Distinguish unqualified features from malformed supported values."""
        if isinstance(value, dict) and value.keys() - cls.model_fields.keys():
            raise UnsupportedFeatureError("request", "native_feature")
        return value


class AnthropicText(NativeModel):
    """A native text content block."""

    type: Literal["text"]
    text: str


class AnthropicToolUse(NativeModel):
    """A client tool call, with only input string values inspected."""

    type: Literal["tool_use"]
    id: Name
    name: Name
    input: JsonObject


class AnthropicToolResult(NativeModel):
    """A client tool result with text-only content."""

    type: Literal["tool_result"]
    tool_use_id: Name
    content: str | Annotated[list[AnthropicText], Field(max_length=64)]
    is_error: bool | None = None


type AnthropicBlock = Annotated[
    AnthropicText | AnthropicToolUse | AnthropicToolResult, Field(discriminator="type")
]


class AnthropicMessage(NativeModel):
    """Native user/assistant roles with bounded text and function blocks."""

    role: Literal["user", "assistant"]
    content: str | Annotated[list[AnthropicBlock], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def validate_tools(self) -> AnthropicMessage:
        """Keep calls in assistant messages and results in user messages."""
        if isinstance(self.content, list) and any(
            (isinstance(block, AnthropicToolUse) and self.role != "assistant")
            or (isinstance(block, AnthropicToolResult) and self.role != "user")
            for block in self.content
        ):
            raise ValueError("native tool role mismatch")  # noqa: TRY003
        return self


class AnthropicTool(NativeModel):
    """A user-defined client function, never a provider-hosted tool."""

    name: Name
    description: str | None = Field(default=None, max_length=20_000)
    input_schema: JsonObject


class AnthropicToolChoice(NativeModel):
    """Select a client tool without opening a provider-specific control object."""

    type: Literal["auto", "any", "tool", "none"]
    name: Name | None = None
    disable_parallel_tool_use: bool | None = None

    @model_validator(mode="after")
    def validate_name(self) -> AnthropicToolChoice:
        """A named choice must identify exactly one tool."""
        if (self.type == "tool") != (self.name is not None):
            raise ValueError("native tool choice requires a matching name")  # noqa: TRY003
        return self


class AnthropicRequest(NativeModel):
    """Supported Messages create request, kept in its native representation."""

    _controls: CompatibilitySettings | None = PrivateAttr(default=None)
    model: NativeModelId
    max_tokens: int = Field(ge=1)
    messages: list[AnthropicMessage] = Field(min_length=1)
    system: str | Annotated[list[AnthropicText], Field(max_length=64)] | None = None
    tools: list[AnthropicTool] | None = Field(default=None, max_length=128)
    tool_choice: AnthropicToolChoice | None = None
    stream: bool | None = None
    temperature: float | None = Field(default=None, ge=0, le=1)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    stop_sequences: list[str] | None = Field(default=None, max_length=16)
    service_tier: Literal["auto", "standard_only"] | None = None


class GoogleFunctionCall(NativeModel):
    """A complete native function call."""

    name: Name
    args: JsonObject
    id: Name | None = None


class GoogleFunctionResponse(NativeModel):
    """A complete native function result."""

    name: Name
    response: JsonObject
    id: Name | None = None


class GooglePart(NativeModel):
    """Exactly one supported part; media and opaque thought signatures are unqualified."""

    text: str | None = None
    functionCall: GoogleFunctionCall | None = None
    functionResponse: GoogleFunctionResponse | None = None

    @model_validator(mode="after")
    def validate_oneof(self) -> GooglePart:
        """Reject ambiguous part variants, including explicit null content."""
        if len(self.model_fields_set) != 1 or all(
            value is None for value in (self.text, self.functionCall, self.functionResponse)
        ):
            raise ValueError("native part requires exactly one content field")  # noqa: TRY003
        return self


class GoogleContent(NativeModel):
    """Native user/model content."""

    role: Literal["user", "model"] | None = None
    parts: list[GooglePart] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_tools(self) -> GoogleContent:
        """Calls and results must have their native conversation roles."""
        if any(
            (part.functionCall is not None and self.role != "model")
            or (part.functionResponse is not None and self.role not in {None, "user"})
            for part in self.parts
        ):
            raise ValueError("native tool role mismatch")  # noqa: TRY003
        return self


class GoogleSystem(NativeModel):
    """Text-only system instruction; its role is ignored by the provider."""

    role: Literal["user", "model", "system"] | None = None
    parts: list[GooglePart] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_text(self) -> GoogleSystem:
        """Never hide a function or opaque content in the instruction channel."""
        if any(part.text is None for part in self.parts):
            raise ValueError("system instruction must be text")  # noqa: TRY003
        return self


class GoogleFunction(NativeModel):
    """A client function declaration with inspected schema prose."""

    name: Name
    description: str | None = Field(default=None, max_length=20_000)
    parameters: JsonObject | None = None
    response: JsonObject | None = None


class GoogleTools(NativeModel):
    """Explicit client functions, excluding hosted retrieval and execution tools."""

    functionDeclarations: list[GoogleFunction] = Field(min_length=1, max_length=128)


class GoogleFunctionConfig(NativeModel):
    """Bounded function selection controls."""

    mode: Literal["AUTO", "ANY", "NONE", "VALIDATED"] | None = None
    allowedFunctionNames: list[Name] | None = Field(default=None, max_length=128)


class GoogleToolConfig(NativeModel):
    """Reviewed tool configuration."""

    functionCallingConfig: GoogleFunctionConfig


class GoogleGenerationConfig(NativeModel):
    """Qualified text-generation controls, with native JSON output schemas."""

    stopSequences: list[str] | None = Field(default=None, max_length=16)
    responseMimeType: Literal["text/plain", "application/json"] | None = None
    responseSchema: JsonObject | None = None
    responseJsonSchema: JsonObject | None = None
    candidateCount: int | None = Field(default=None, ge=1, le=8)
    maxOutputTokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    topP: float | None = Field(default=None, ge=0, le=1)
    topK: int | None = Field(default=None, ge=1)
    seed: int | None = None

    @model_validator(mode="after")
    def validate_schema(self) -> GoogleGenerationConfig:
        """A response schema must select JSON output and one schema dialect."""
        if (self.responseSchema is not None or self.responseJsonSchema is not None) and (
            self.responseMimeType != "application/json"
            or (self.responseSchema is not None and self.responseJsonSchema is not None)
        ):
            raise ValueError("native response schema requires JSON output")  # noqa: TRY003
        return self


class GoogleRequest(NativeModel):
    """Supported generateContent body; model is bound to its validated HTTP path."""

    _controls: CompatibilitySettings | None = PrivateAttr(default=None)
    model: NativeModelId | None = None
    contents: list[GoogleContent] = Field(min_length=1)
    systemInstruction: GoogleSystem | None = None
    tools: list[GoogleTools] | None = Field(default=None, max_length=32)
    toolConfig: GoogleToolConfig | None = None
    generationConfig: GoogleGenerationConfig | None = None
