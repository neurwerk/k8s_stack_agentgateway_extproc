"""Strict wire models mirroring the PII engine adapter contract."""

from __future__ import annotations

from typing import Annotated, Literal

from neurwerk_request_segments.models import (
    ENGINE_REQUEST_ADAPTER as ENGINE_REQUEST_ADAPTER,
)
from neurwerk_request_segments.models import (
    EngineAttachmentPart as EngineAttachmentPart,
)
from neurwerk_request_segments.models import (
    EngineChatRequest as EngineChatRequest,
)
from neurwerk_request_segments.models import (
    EngineChatStreamOptions as EngineChatStreamOptions,
)
from neurwerk_request_segments.models import (
    EngineFunction as EngineFunction,
)
from neurwerk_request_segments.models import (
    EngineMcpParams as EngineMcpParams,
)
from neurwerk_request_segments.models import (
    EngineMcpRequest as EngineMcpRequest,
)
from neurwerk_request_segments.models import (
    EngineMessage as EngineMessage,
)
from neurwerk_request_segments.models import (
    EngineMessageContent as EngineMessageContent,
)
from neurwerk_request_segments.models import (
    EngineRequest as EngineRequest,
)
from neurwerk_request_segments.models import (
    EngineResponseFunctionCall as EngineResponseFunctionCall,
)
from neurwerk_request_segments.models import (
    EngineResponseFunctionOutput as EngineResponseFunctionOutput,
)
from neurwerk_request_segments.models import (
    EngineResponseInput as EngineResponseInput,
)
from neurwerk_request_segments.models import (
    EngineResponseInputItem as EngineResponseInputItem,
)
from neurwerk_request_segments.models import (
    EngineResponseMessage as EngineResponseMessage,
)
from neurwerk_request_segments.models import (
    EngineResponsesRequest as EngineResponsesRequest,
)
from neurwerk_request_segments.models import (
    EngineResponseTextConfig as EngineResponseTextConfig,
)
from neurwerk_request_segments.models import (
    EngineResponseTextFormat as EngineResponseTextFormat,
)
from neurwerk_request_segments.models import (
    EngineResponseTextFormatObject as EngineResponseTextFormatObject,
)
from neurwerk_request_segments.models import (
    EngineResponseTextFormatSchema as EngineResponseTextFormatSchema,
)
from neurwerk_request_segments.models import (
    EngineResponseTextFormatText as EngineResponseTextFormatText,
)
from neurwerk_request_segments.models import (
    EngineResponseTextPart as EngineResponseTextPart,
)
from neurwerk_request_segments.models import (
    EngineTextPart as EngineTextPart,
)
from neurwerk_request_segments.models import (
    EngineToolCall as EngineToolCall,
)
from neurwerk_request_segments.models import (
    EngineToolDefinition as EngineToolDefinition,
)
from neurwerk_request_segments.models import (
    EngineToolFunction as EngineToolFunction,
)
from neurwerk_request_segments.models import (
    JsonScalar as JsonScalar,
)
from neurwerk_request_segments.models import (
    JsonValue as JsonValue,
)
from neurwerk_request_segments.models import (
    McpJsonValue as McpJsonValue,
)
from neurwerk_request_segments.models import (
    McpRequestId as McpRequestId,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentgateway_extproc.models.exceptions import (
    MAX_ENGINE_ERROR_MESSAGE_LENGTH,
    EngineErrorCode,
    LimitDetail,
    is_safe_engine_error_message,
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


class EngineLimitErrorDetail(EngineErrorDetail):
    """Bind measured limit facts to the size rejection code."""

    code: Literal["request_too_large"]
    limit: LimitDetail


class EngineLimitErrorReply(EngineModel):
    """Validate the extended error envelope independently of v1 acceptance."""

    api_version: Literal["v2"]
    error: EngineLimitErrorDetail


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


class FaceFindings(EngineModel):
    """Carry aggregate local detection facts, never pixels or identities."""

    scan_status: Literal["complete", "not_scanned", "failed"]
    count: Annotated[int, Field(strict=True, ge=0, le=10_000_000)] | None

    @model_validator(mode="after")
    def validate_scan(self) -> FaceFindings:
        """Do not turn a missing or failed scan into a clean result."""
        if (self.scan_status == "complete") != (self.count is not None):
            raise ValueError("face count requires a complete scan")  # noqa: TRY003
        return self


class VisualFindings(EngineModel):
    """Versioned document-envelope visual findings."""

    faces: FaceFindings


class EngineDocumentRequest(EngineModel):
    """Add trusted controls to a converted text-only Chat or Responses request."""

    api_version: Literal["v1"]
    request: EngineChatRequest | EngineResponsesRequest
    text_pii_enabled: Annotated[bool, Field(strict=True)]
    visual_findings: VisualFindings


type PIIAction = Literal[
    "pass",
    "block",
    "reroute",
    "text-only",
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
        if self.entity_type == "FACE":
            if self.action not in {"block", "text-only", "reroute"} or self.transformed_count:
                raise ValueError("FACE rows require a visual action without transformations")  # noqa: TRY003
        elif self.action == "text-only":
            raise ValueError("text-only is a FACE action")  # noqa: TRY003
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
    visual_findings: VisualFindings | None = None
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
    def validate_visual_findings(self) -> EngineReply:
        """Bind FACE counts and report rows to fresh local visual findings."""
        if self.visual_findings is None:
            if "FACE" in self.entities:
                raise ValueError("FACE requires visual findings")  # noqa: TRY003
            return self
        if (
            self.analysis.source != "current_request"
            or self.analysis.cached_decision_applied
            or isinstance(self.request, EngineMcpRequest)
        ):
            raise ValueError("visual findings require fresh document analysis")  # noqa: TRY003
        faces = self.visual_findings.faces
        count = faces.count or 0
        if self.entity_counts.get("FACE", 0) != count:
            raise ValueError("FACE counts must match visual findings")  # noqa: TRY003
        if self.decision != "block" and any(
            row.entity_type == "FACE" and row.action not in self.applied_actions
            for row in self.report.rows
        ):
            raise ValueError("FACE report action must be applied")  # noqa: TRY003
        if faces.scan_status == "failed" and self.decision != "block":
            raise ValueError("failed visual scans must block")  # noqa: TRY003
        if not self.analysis.scan_performed and (set(self.entities) - {"FACE"} or self.reversal):
            raise ValueError("unscanned text cannot report text findings or reversal")  # noqa: TRY003
        return self

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
        if unscanned_current_success and self.visual_findings is None:
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
            (not transformed and "text-only" not in row_actions)
            or row_actions & {"block", "reroute"}
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
