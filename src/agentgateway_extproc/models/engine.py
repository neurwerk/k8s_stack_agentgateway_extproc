"""Strict wire models mirroring the PII engine adapter contract."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from agentgateway_extproc.models.exceptions import (
    MAX_ENGINE_ERROR_MESSAGE_LENGTH,
    EngineErrorCode,
    is_safe_engine_error_message,
)

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
    | Annotated[
        int,
        Field(strict=True, ge=-9_007_199_254_740_991, le=9_007_199_254_740_991),
    ]
)


class EngineModel(BaseModel):
    """Reject undocumented fields at the trusted engine boundary."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False, validate_assignment=True)


class EngineErrorDetail(EngineModel):
    """Validate the bounded detail of a versioned engine rejection."""

    code: EngineErrorCode
    message: str = Field(min_length=1, max_length=MAX_ENGINE_ERROR_MESSAGE_LENGTH)
    retryable: bool

    @field_validator("message")
    @classmethod
    def validate_safe_message(cls, value: str) -> str:
        """Reject controls, format characters, and surrounding whitespace."""
        if not is_safe_engine_error_message(value):
            raise ValueError("engine error message is unsafe")  # noqa: TRY003
        return value


class EngineErrorReply(EngineModel):
    """Represent the exact versioned PII Engine non-success envelope."""

    api_version: Literal["v1"]
    error: EngineErrorDetail


class EngineTextPart(EngineModel):
    """Represent one OpenAI Chat text part."""

    type: Literal["text"]
    text: str = Field(min_length=1)


class EngineAttachmentPart(BaseModel):
    """Mirror attachment blocks that the engine accepts only to reject by policy."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=False, validate_assignment=True)

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


class EngineFunction(EngineModel):
    """Represent a function call in an OpenAI-compatible request."""

    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    arguments: JsonValue = ""


class EngineToolCall(EngineModel):
    """Represent an assistant tool call."""

    id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    type: Literal["function"]
    function: EngineFunction


class EngineToolFunction(EngineModel):
    """Describe a function tool offered to the model."""

    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    description: str | None = Field(default=None, max_length=4_000)
    parameters: dict[str, JsonValue] | None = None


class EngineToolDefinition(EngineModel):
    """Describe one function tool in an engine request."""

    type: Literal["function"]
    function: EngineToolFunction


class EngineMessage(EngineModel):
    """Represent a supported Chat message."""

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: EngineMessageContent | None = None
    name: str | None = Field(default=None, max_length=256)
    tool_calls: list[EngineToolCall] = Field(default_factory=list, max_length=32)
    tool_call_id: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def validate_role_fields(self) -> EngineMessage:
        """Require fields that distinguish assistant calls and tool results."""
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")  # noqa: TRY003
        if self.tool_calls and self.role != "assistant":
            raise ValueError("tool_calls require an assistant message")  # noqa: TRY003
        if self.role == "assistant" and self.content is None and not self.tool_calls:
            raise ValueError("assistant messages require content or tool_calls")  # noqa: TRY003
        return self


class EngineChatRequest(EngineModel):
    """Represent a bounded OpenAI Chat Completions request."""

    model: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_./:-]+$")
    messages: list[EngineMessage] = Field(min_length=1, max_length=256)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    stream: bool = False
    n: int | None = Field(default=None, ge=1, le=16)
    stop: str | list[str] | None = None
    tools: list[EngineToolDefinition] = Field(default_factory=list, max_length=128)
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    response_format: dict[str, JsonValue] | None = None
    user: str | None = Field(default=None, max_length=256)


class EngineResponseTextPart(EngineModel):
    """Represent one Responses API text content part."""

    type: Literal["input_text", "output_text"]
    text: str = Field(min_length=1)


class EngineResponseMessage(EngineModel):
    """Represent one Responses API message."""

    type: Literal["message"] = "message"
    role: Literal["system", "developer", "user", "assistant"]
    content: list[EngineResponseTextPart | EngineAttachmentPart] = Field(
        min_length=1, max_length=64
    )


class EngineResponseFunctionCall(EngineModel):
    """Represent a Responses API function call."""

    type: Literal["function_call"]
    call_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    arguments: JsonValue


class EngineResponseFunctionOutput(EngineModel):
    """Represent nested textual output returned by a tool."""

    type: Literal["function_call_output"]
    call_id: str = Field(min_length=1, max_length=256)
    output: JsonValue


class EngineResponseTextFormatText(EngineModel):
    """Select ordinary text output from the Responses API."""

    type: Literal["text"]


class EngineResponseTextFormatObject(EngineModel):
    """Select the legacy JSON object response format."""

    type: Literal["json_object"]


class EngineResponseTextFormatSchema(EngineModel):
    """Bound one Responses API structured-output JSON schema."""

    model_config = ConfigDict(serialize_by_alias=True)

    type: Literal["json_schema"]
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = Field(default=None, max_length=4_000)
    schema_value: dict[str, JsonValue] = Field(alias="schema", max_length=256)
    strict: bool | None = None


type EngineResponseTextFormat = Annotated[
    EngineResponseTextFormatText | EngineResponseTextFormatObject | EngineResponseTextFormatSchema,
    Field(discriminator="type"),
]


class EngineResponseTextConfig(EngineModel):
    """Configure bounded Responses API text output."""

    format: EngineResponseTextFormat | None = None
    verbosity: Literal["low", "medium", "high"] | None = None


type EngineResponseInputItem = (
    EngineResponseMessage | EngineResponseFunctionCall | EngineResponseFunctionOutput
)
type EngineResponseInput = (
    str | Annotated[list[EngineResponseInputItem], Field(min_length=1, max_length=256)]
)


class EngineResponsesRequest(EngineModel):
    """Represent a bounded OpenAI Responses request."""

    model: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_./:-]+$")
    input: EngineResponseInput
    instructions: str | None = None
    tools: list[EngineToolDefinition] = Field(default_factory=list, max_length=128)
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_output_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    text: EngineResponseTextConfig | None = None
    stream: bool = False
    previous_response_id: str | None = Field(default=None, max_length=256)
    user: str | None = Field(default=None, max_length=256)


class EngineMcpParams(EngineModel):
    """Represent narrowed MCP tool input and immutable protocol metadata."""

    model_config = ConfigDict(serialize_by_alias=True)

    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    arguments: Annotated[dict[str, McpJsonValue], Field(max_length=256)] | None = None
    meta: (
        Annotated[
            dict[Annotated[str, Field(max_length=256)], McpJsonValue],
            Field(max_length=64),
        ]
        | None
    ) = Field(default=None, alias="_meta")

    @model_validator(mode="before")
    @classmethod
    def validate_optional_objects(cls, value: object) -> object:
        """Reject explicit null where MCP permits only an omitted or object field."""
        if isinstance(value, dict) and any(
            key in value and value[key] is None for key in ("arguments", "_meta")
        ):
            raise ValueError("optional MCP params must be objects when present")  # noqa: TRY003
        return value


class EngineMcpRequest(EngineModel):
    """Represent one bounded MCP ``tools/call`` analysis request."""

    jsonrpc: Literal["2.0"]
    id: McpRequestId
    method: Literal["tools/call"]
    params: EngineMcpParams


type EngineRequest = EngineChatRequest | EngineResponsesRequest | EngineMcpRequest
ENGINE_REQUEST_ADAPTER: TypeAdapter[EngineRequest] = TypeAdapter(EngineRequest)


class AnalysisMetadata(EngineModel):
    """Represent bounded engine analysis facts without request values."""

    source: Literal["current_request", "cached_decision"]
    scan_performed: bool
    duration_ms: int | None = Field(ge=0, le=615_000)
    overlap_count: int = Field(ge=0, le=10_000_000)
    overlap_resolution: Literal["strictest_action"]
    policy_version: str = Field(min_length=1, max_length=64)
    text_leaf_count: int = Field(ge=0, le=2_048)
    cached_decision_applied: bool

    @model_validator(mode="after")
    def validate_provenance(self) -> AnalysisMetadata:
        """Require scan timing and cache provenance to agree."""
        if self.scan_performed != (self.duration_ms is not None):
            raise ValueError(  # noqa: TRY003
                "scan duration must exist exactly when a scan was performed"
            )
        if self.scan_performed and self.source != "current_request":
            raise ValueError("performed scans must describe the current request")  # noqa: TRY003
        if self.source == "cached_decision" and not self.cached_decision_applied:
            raise ValueError("cached analysis metadata must apply a cached decision")  # noqa: TRY003
        if not self.scan_performed and self.source == "current_request" and self.overlap_count:
            raise ValueError("unscanned current requests cannot report overlaps")  # noqa: TRY003
        return self


class Notices(EngineModel):
    """Represent policy-owned prompt and response messages."""

    request: list[Annotated[str, Field(max_length=4_000)]] = Field(max_length=16)
    response: list[Annotated[str, Field(max_length=4_000)]] = Field(max_length=16)


type PIIAction = Literal[
    "pass",
    "block",
    "reroute",
    "mask",
    "replace",
    "redact",
    "hash",
    "encrypt",
    "reversible_replace",
]


class PIIReportRow(EngineModel):
    """Describe one entity action and its logical request counts."""

    entity_type: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    action: PIIAction
    detected_count: int = Field(ge=1, le=10_000_000)
    transformed_count: int = Field(ge=0, le=10_000_000)
    unique_transformed_count: int = Field(ge=0, le=10_000_000)

    @model_validator(mode="after")
    def validate_counts(self) -> PIIReportRow:
        """Require transformed and unique counts to describe detected values."""
        if self.transformed_count > self.detected_count:
            raise ValueError("transformed_count cannot exceed detected_count")  # noqa: TRY003
        if self.unique_transformed_count > self.transformed_count:
            raise ValueError(  # noqa: TRY003
                "unique_transformed_count cannot exceed transformed_count"
            )
        if self.action in {"pass", "block"} and self.transformed_count:
            raise ValueError("pass and block rows cannot claim transformations")  # noqa: TRY003
        return self


class PIIReport(EngineModel):
    """Carry the engine's detailed request report."""

    rows: list[PIIReportRow] = Field(max_length=64)

    @model_validator(mode="after")
    def validate_rows(self) -> PIIReport:
        """Require deterministic rows with one entry per entity type."""
        entity_types = [row.entity_type for row in self.rows]
        if len(entity_types) != len(set(entity_types)):
            raise ValueError("report rows must contain unique entity types")  # noqa: TRY003
        if entity_types != sorted(entity_types):
            raise ValueError("report rows must be sorted by entity_type")  # noqa: TRY003
        return self


class EngineReply(EngineModel):
    """Validate the complete adapter response before any control-flow use."""

    api_version: Literal["v1"]
    decision: Literal["pass", "block", "apply_actions", "reroute"]
    entities: list[Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")]] = Field(
        default_factory=list, max_length=64
    )
    entity_counts: dict[str, int] = Field(default_factory=dict, max_length=64)
    applied_actions: list[str] = Field(default_factory=list, max_length=16)
    remote_allowed: bool
    route_class: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_.:/-]+$")
    request: EngineRequest | None = None
    analysis: AnalysisMetadata
    notices: Notices
    report: PIIReport
    safety_rule: str | None = Field(default=None, max_length=128)
    reversal: dict[
        Annotated[
            str,
            Field(
                min_length=3,
                max_length=256,
                pattern=r"^<(?:REV|ENCRYPTED)_[A-Z][A-Z0-9_]*_[0-9a-f]{16}_[0-9a-f]{16}>$",
            ),
        ],
        Annotated[str, Field(min_length=1, max_length=4_000_000)],
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_decision_shape(self) -> EngineReply:  # noqa: C901
        """Require mutations and routing flags to agree with the decision."""
        actions = set(self.applied_actions)
        if len(self.entities) != len(set(self.entities)):
            raise ValueError("engine reply contains duplicate entity types")  # noqa: TRY003
        if set(self.entity_counts) != set(self.entities) or any(
            count <= 0 or count > 10_000_000 for count in self.entity_counts.values()
        ):
            raise ValueError("engine reply entity counts are inconsistent")  # noqa: TRY003
        report_counts = {row.entity_type: row.detected_count for row in self.report.rows}
        if self.analysis.cached_decision_applied:
            if self.decision not in {"block", "reroute"}:
                raise ValueError(  # noqa: TRY003
                    "cached reports require a cached terminal decision"
                )
            if any(
                entity_type not in self.entity_counts or count > self.entity_counts[entity_type]
                for entity_type, count in report_counts.items()
            ):
                raise ValueError("cached report rows exceed engine entity counts")  # noqa: TRY003
        elif report_counts != self.entity_counts:
            raise ValueError("current report rows must match engine entity counts")  # noqa: TRY003
        unscanned_current_success = (
            self.analysis.source == "current_request"
            and not self.analysis.scan_performed
            and self.decision != "block"
        )
        if unscanned_current_success:
            if not _is_no_text_mcp_request(self.request):
                raise ValueError(  # noqa: TRY003
                    "unscanned current success requires a no-text MCP request"
                )
            if (
                self.decision != "pass"
                or not self.remote_allowed
                or self.entities
                or self.entity_counts
                or self.applied_actions
                or self.route_class is not None
                or self.analysis.text_leaf_count
                or self.analysis.cached_decision_applied
                or self.notices.request
                or self.notices.response
                or self.safety_rule is not None
                or self.report.rows
                or self.reversal
            ):
                raise ValueError(  # noqa: TRY003
                    "no-text MCP success must be an unchanged unscanned pass"
                )
        if isinstance(self.request, EngineMcpRequest) and (
            self.decision == "reroute"
            or self.route_class is not None
            or self.notices.request
            or self.notices.response
        ):
            raise ValueError("MCP analysis cannot expose model routing or notices")  # noqa: TRY003
        row_actions = {row.action for row in self.report.rows}
        transformed = any(row.transformed_count for row in self.report.rows)
        if self.decision == "pass" and row_actions - {"pass"}:
            raise ValueError("pass decisions require pass report rows")  # noqa: TRY003
        if self.decision == "apply_actions" and (
            not transformed or row_actions & {"block", "reroute"}
        ):
            raise ValueError(  # noqa: TRY003
                "action decisions require transformed non-terminal report rows"
            )
        if self.decision == "reroute":
            if "block" in row_actions:
                raise ValueError("reroute decisions cannot contain block report rows")  # noqa: TRY003
            if (
                self.analysis.source == "cached_decision"
                or not self.analysis.cached_decision_applied
            ) and "reroute" not in row_actions:
                raise ValueError("reroute decisions require a matching report row")  # noqa: TRY003
        if self.decision == "block" and (
            transformed or (self.report.rows and "block" not in row_actions)
        ):
            raise ValueError("block decisions require an untransformed block report")  # noqa: TRY003
        if self.decision == "block":
            if (
                self.request is not None
                or self.reversal
                or self.remote_allowed
                or self.route_class is not None
                or actions != {"block"}
            ):
                raise ValueError("blocked engine replies contain invalid forwarding data")  # noqa: TRY003
            return self
        if self.request is None:
            raise ValueError("non-blocked engine replies must include request")  # noqa: TRY003
        if self.decision == "reroute":
            if self.route_class is None or self.remote_allowed or "reroute" not in actions:
                raise ValueError("rerouted replies require a trusted local route")  # noqa: TRY003
        elif self.decision == "pass":
            if not self.remote_allowed or self.reversal or actions - {"pass"}:
                raise ValueError("pass replies contain action or routing data")  # noqa: TRY003
        elif not self.remote_allowed or not actions or actions & {"block", "reroute"}:
            raise ValueError("action replies contain invalid routing or action data")  # noqa: TRY003
        return self


def _is_no_text_mcp_request(request: EngineRequest | None) -> bool:
    return isinstance(request, EngineMcpRequest) and not _mcp_contains_string(
        request.params.arguments
    )


def _mcp_contains_string(value: McpJsonValue | None) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, list):
        return any(_mcp_contains_string(item) for item in value)
    if isinstance(value, dict):
        return any(_mcp_contains_string(item) for item in value.values())
    return False
