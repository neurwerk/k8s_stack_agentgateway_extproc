"""Strict trusted destination metadata supplied by AgentGateway."""

from __future__ import annotations

from typing import Annotated, Literal, cast

from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.models.exceptions import TrustedMetadataError

DESTINATION_POLICY_NAMESPACE = "neurwerk.destination_policy"
MAX_PRINCIPAL_BYTES = 256

type ModelId = Annotated[
    str,
    Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_./:-]+$"),
]


class DestinationModel(BaseModel):
    """Reject coercion and undocumented trusted metadata fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelDestinationPolicy(DestinationModel):
    """Select independent PII and attachment behavior from a trusted model catalog."""

    contract_version: Literal[1, 2, 3]
    destination_kind: Literal["model"]
    principal_id: str
    models: dict[ModelId, bool] = Field(min_length=1, max_length=256)
    attachment_modes: dict[ModelId, Literal["block", "extract", "process", "passthrough"]] = Field(
        default_factory=dict, max_length=256
    )
    image_forwarding: dict[ModelId, Literal["none", "if-no-pii-detected", "pii-unchecked"]] = Field(
        default_factory=dict, max_length=256
    )
    face_protection: dict[ModelId, bool] = Field(default_factory=dict, max_length=256)
    local_models: dict[ModelId, bool] = Field(default_factory=dict, max_length=256)
    image_models: dict[ModelId, bool] = Field(default_factory=dict, max_length=256)
    image_reroutes: dict[ModelId, Annotated[dict[ModelId, ModelId], Field(max_length=256)]] = Field(
        default_factory=dict, max_length=256
    )

    @model_validator(mode="before")
    @classmethod
    def validate_version_fields(cls, value: object) -> object:
        """Keep older versions strict, including explicitly empty new maps."""
        if not isinstance(value, dict):
            return value
        if value.get("contract_version") in {1, 2} and (
            {"image_models", "image_reroutes"} & value.keys()
        ):
            raise ValueError("image capability bindings require contract version three")  # noqa: TRY003
        modes = value.get("attachment_modes")
        if value.get("contract_version") == 1 and (
            {"image_forwarding", "face_protection", "local_models"} & value.keys()
            or (isinstance(modes, dict) and "process" in modes.values())
        ):
            raise ValueError("image policy requires contract version two")  # noqa: TRY003
        return value

    def protects_faces(self, model: str) -> bool:
        """Default processed attachments to protection, preserving legacy passthrough."""
        return self.face_protection.get(model, self.attachment_modes.get(model) != "passthrough")

    @field_validator("principal_id")
    @classmethod
    def validate_principal(cls, value: str) -> str:
        """Require a bounded printable opaque principal."""
        return _validated_principal(value)

    @model_validator(mode="after")
    def validate_attachment_modes(self) -> ModelDestinationPolicy:
        """Reject unknown destinations and raw forwarding through enabled PII."""
        if any(
            mapping.keys() - self.models.keys()
            for mapping in (
                self.attachment_modes,
                self.image_forwarding,
                self.face_protection,
                self.local_models,
                self.image_models,
                self.image_reroutes,
            )
        ):
            raise ValueError("attachment modes require known model IDs")  # noqa: TRY003
        for model, pii in self.models.items():
            mode = self.attachment_modes.get(model, "block")
            forwarding = self.image_forwarding.get(model, "none")
            face = self.protects_faces(model)
            if mode == "passthrough" and (pii or face or model in self.image_forwarding):
                raise ValueError("passthrough requires protections and forwarding disabled")  # noqa: TRY003
            if forwarding != "none" and mode not in {"process", "extract"}:
                raise ValueError("image forwarding requires processing")  # noqa: TRY003
            if forwarding == "pii-unchecked" and (face or not self.local_models.get(model, False)):
                raise ValueError("unchecked images require an unprotected concrete local route")  # noqa: TRY003
            if (
                self.contract_version == 3
                and forwarding == "pii-unchecked"
                and not self.image_models.get(model, False)
            ):
                raise ValueError("unchecked images require a proven local image model")  # noqa: TRY003
            if forwarding == "if-no-pii-detected" and not (pii and face):
                raise ValueError("conditional images require PII and face protection")  # noqa: TRY003
        return self


class McpDestinationPolicy(DestinationModel):
    """Identify one canonical PII-enabled MCP route."""

    contract_version: Literal[1]
    destination_kind: Literal["mcp"]
    principal_id: str
    destination_id: str = Field(
        min_length=1,
        max_length=48,
        pattern=r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*$",
    )
    pii_enabled: bool

    @field_validator("principal_id")
    @classmethod
    def validate_principal(cls, value: str) -> str:
        """Require a bounded printable opaque principal."""
        return _validated_principal(value)


type DestinationPolicy = ModelDestinationPolicy | McpDestinationPolicy
DESTINATION_POLICY_ADAPTER: TypeAdapter[DestinationPolicy] = TypeAdapter(
    Annotated[DestinationPolicy, Field(discriminator="destination_kind")]
)


def destination_policy_from_request(
    request: ext_proc_pb2.ProcessingRequest,
) -> DestinationPolicy:
    """Read one strict policy from the trusted protobuf metadata namespace."""
    try:
        metadata = request.metadata_context.filter_metadata[DESTINATION_POLICY_NAMESPACE]
        payload = cast(
            dict[str, object],
            MessageToDict(metadata, preserving_proto_field_name=True),
        )
        version = payload.get("contract_version")
        # google.protobuf.Struct represents every JSON number as a double.
        if type(version) is float and version in {1.0, 2.0, 3.0}:
            payload["contract_version"] = int(version)
        return DESTINATION_POLICY_ADAPTER.validate_python(payload, strict=True)
    except TrustedMetadataError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TrustedMetadataError from exc


def _validated_principal(value: str) -> str:
    if (
        not value
        or value != value.strip()
        or not value.isprintable()
        or len(value.encode("utf-8")) > MAX_PRINCIPAL_BYTES
    ):
        raise ValueError("principal_id is invalid")  # noqa: TRY003
    return value
