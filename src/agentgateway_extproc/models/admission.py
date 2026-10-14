"""Immutable gateway admission and separately resolved model selection."""

from __future__ import annotations

import json
from typing import Annotated, Literal, cast

from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.models.destination import (
    DESTINATION_POLICY_NAMESPACE,
    DestinationPolicy,
    ModelDestinationPolicy,
    ModelId,
    _validated_principal,
    destination_policy_from_payload,
)
from agentgateway_extproc.models.exceptions import GatewayPolicyError, TrustedMetadataError
from agentgateway_extproc.models.immutable import FrozenMap

type Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
type Permission = Annotated[
    str, Field(min_length=1, max_length=320, pattern=r"^[A-Za-z0-9_./:-]+$")
]
type ProfileId = Literal[
    "openai-chat-completions-v1",
    "openai-responses-v1",
    "anthropic-messages-v1",
    "google-generate-content-v1",
    "chatgpt-web-v1",
    "gamma-web-v1",
]


def _permission_tuple(value: object) -> object:
    if isinstance(value, list):
        value = tuple(value)
    if isinstance(value, tuple) and len(value) != len(set(value)):
        raise ValueError("permission entries must be unique")  # noqa: TRY003
    return value


class AdmissionModel(BaseModel):
    """Reject coercion and extra fields in trusted admission metadata."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class AdmissionContext(AdmissionModel):
    """Retain only gateway-verified facts, never provider credential values."""

    request_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    principal_id: str
    credential_context_version: Literal["1"] | None
    credential_id: str | None
    credential_kind: Literal["personal", "managed"] | None
    entry_mode: Literal["api", "forward_proxy"]
    destination_kind: Literal["model", "mcp", "application"]
    provider_id: Identifier | None
    binding_id: Identifier
    protocol_profile: ProfileId
    operation_id: Identifier
    provider_credential_source: Literal["caller_supplied", "company", "none"]
    policy_revision: Identifier
    permissions: tuple[Permission, ...] = Field(max_length=1024)

    @field_validator("binding_id")
    @classmethod
    def validate_binding_path_segment(cls, value: str) -> str:
        """Keep the verified binding usable as exactly one literal URL path segment."""
        if value in {".", ".."}:
            raise ValueError("binding cannot be a dot path segment")  # noqa: TRY003
        return value

    @field_validator("permissions", mode="before")
    @classmethod
    def validate_permissions(cls, value: object) -> object:
        """Accept JSON arrays as immutable, unique permission lists."""
        return _permission_tuple(value)

    @field_validator("principal_id", "credential_id")
    @classmethod
    def validate_identity(cls, value: str | None) -> str | None:
        """Share the existing bounded opaque identity contract."""
        return _validated_principal(value) if value is not None else None

    @model_validator(mode="after")
    def validate_credential_binding(self) -> AdmissionContext:
        """Require a complete Neurwerk key identity and an identifiable billed provider."""
        present = (
            self.credential_context_version is not None,
            self.credential_id is not None,
            self.credential_kind is not None,
        )
        if any(present) and not all(present):
            raise ValueError("credential fields must be paired")  # noqa: TRY003
        if self.provider_id is None and (
            self.provider_credential_source == "company"
            or (self.destination_kind != "mcp" and self.provider_credential_source != "none")
        ):
            raise ValueError("credential mode requires a provider")  # noqa: TRY003
        return self


class ModelBinding(AdmissionModel):
    """Bind one canonical model to its provider name and all-of permissions."""

    upstream_model: ModelId
    required_permissions: tuple[Permission, ...] = Field(min_length=2, max_length=64)
    content_tracing_enabled: bool

    @field_validator("required_permissions", mode="before")
    @classmethod
    def validate_permissions(cls, value: object) -> object:
        """Keep the binding's permission set immutable and unambiguous."""
        return _permission_tuple(value)


class PiiPolicy(AdmissionModel):
    """Select the existing Engine policy switch, not an arbitrary policy override."""

    enabled: bool


class RequestContext(AdmissionContext):
    """Freeze one authorized selection after body validation."""

    destination_id: ModelId
    requested_model: ModelId
    resolved_model: ModelId
    pii_policy: PiiPolicy
    content_tracing_enabled: bool
    billing_classification: Literal["company_api", "personal_api", "subscription", "unknown"]
    usage_completeness: Literal["known", "partial", "unknown"] = "unknown"


class AdmissionEnvelope(AdmissionModel):
    """Lock a single upstream binding and its body-selectable model catalog."""

    contract_version: Literal[5]
    request_context: AdmissionContext
    destination_policy: Annotated[DestinationPolicy, Field(discriminator="destination_kind")]
    model_bindings: FrozenMap[ModelId, ModelBinding] = Field(max_length=256)

    @field_validator("destination_policy", mode="before")
    @classmethod
    def validate_inner_policy(cls, value: object) -> object:
        """Reuse legacy validation, including protobuf number normalization."""
        if isinstance(value, dict):
            return destination_policy_from_payload(value)
        return value

    @model_validator(mode="after")
    def validate_bindings(self) -> AdmissionEnvelope:
        """Reject conflicting identity, incomplete coverage and ambiguous model aliases."""
        context, policy = self.request_context, self.destination_policy
        if (context.principal_id, context.destination_kind) != (
            policy.principal_id,
            policy.destination_kind,
        ):
            raise ValueError("admission identity or destination kind disagrees")  # noqa: TRY003
        if isinstance(policy, ModelDestinationPolicy):
            fields = {"credential_context_version", "credential_id", "credential_kind"}
            if fields & policy.model_fields_set and any(
                getattr(context, field) != getattr(policy, field) for field in fields
            ):
                raise ValueError("admission credential identity disagrees")  # noqa: TRY003
            if self.model_bindings.keys() != policy.models.keys():
                raise ValueError("model bindings must cover the policy exactly")  # noqa: TRY003
        elif self.model_bindings:
            raise ValueError("MCP admission cannot contain model bindings")  # noqa: TRY003
        aliases: set[str] = set()
        for model, binding in self.model_bindings.items():
            alias = binding.upstream_model
            if alias in aliases or (alias != model and alias in self.model_bindings):
                raise ValueError("model aliases collide")  # noqa: TRY003
            aliases.add(alias)
            if not {"llm:invoke", f"model:{model}:invoke"} <= set(binding.required_permissions):
                raise ValueError("model binding omits required invocation permissions")  # noqa: TRY003
        return self

    def resolve_model(self, requested_model: str) -> RequestContext:
        """Make the sole model selection used by policy, authorization and final dispatch."""
        policy = self.destination_policy
        if not isinstance(policy, ModelDestinationPolicy):
            raise GatewayPolicyError("unsupported_feature")
        model = next(
            (
                model
                for model, binding in self.model_bindings.items()
                if requested_model in {model, binding.upstream_model}
            ),
            None,
        )
        if model is None:
            raise GatewayPolicyError("unknown_model")
        binding = self.model_bindings[model]
        required = set(binding.required_permissions)
        context = self.request_context
        if context.provider_credential_source == "company":
            required.add(f"provider:{context.provider_id}:company-use")
        if not required <= set(context.permissions):
            raise GatewayPolicyError("permission_denied")
        return RequestContext(
            **context.model_dump(),
            destination_id=model,
            requested_model=requested_model,
            resolved_model=binding.upstream_model,
            pii_policy=PiiPolicy(enabled=policy.models[model]),
            content_tracing_enabled=binding.content_tracing_enabled,
            billing_classification=(
                "company_api" if context.provider_credential_source == "company" else "unknown"
            ),
        )


def metadata_from_request(
    request: ext_proc_pb2.ProcessingRequest,
) -> DestinationPolicy | AdmissionEnvelope:
    """Parse the new envelope independently from the unchanged legacy policy schemas."""
    try:
        payload = cast(
            dict[str, object],
            MessageToDict(
                request.metadata_context.filter_metadata[DESTINATION_POLICY_NAMESPACE],
                preserving_proto_field_name=True,
            ),
        )
        if payload.get("contract_version") == 5:
            return AdmissionEnvelope.model_validate({**payload, "contract_version": 5})
        return destination_policy_from_payload(payload)
    except TrustedMetadataError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TrustedMetadataError from exc


def admission_snapshot(request: ext_proc_pb2.ProcessingRequest) -> str:
    """Compare the complete wire envelope, including omitted versus explicit defaults."""
    return json.dumps(
        MessageToDict(request.metadata_context.filter_metadata[DESTINATION_POLICY_NAMESPACE]),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def is_admission_request(request: ext_proc_pb2.ProcessingRequest) -> bool:
    """Suppress raw capture even for malformed or changed v5 metadata."""
    if DESTINATION_POLICY_NAMESPACE not in request.metadata_context.filter_metadata:
        return False
    fields = request.metadata_context.filter_metadata[DESTINATION_POLICY_NAMESPACE].fields
    return bool(
        "request_context" in fields
        or "model_bindings" in fields
        or (
            "contract_version" in fields
            and (
                fields["contract_version"].number_value == 5
                or fields["contract_version"].string_value == "5"
            )
        )
    )
