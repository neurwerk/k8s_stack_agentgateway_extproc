"""Strict trusted destination metadata supplied by AgentGateway."""

from __future__ import annotations

from typing import Annotated, Literal, cast

from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

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
    """Select PII behavior from a bounded exact model catalog."""

    contract_version: Literal[1]
    destination_kind: Literal["model"]
    principal_id: str
    models: dict[ModelId, bool] = Field(min_length=1, max_length=256)

    @field_validator("principal_id")
    @classmethod
    def validate_principal(cls, value: str) -> str:
        """Require a bounded printable opaque principal."""
        return _validated_principal(value)


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
        if type(version) is float and version == 1.0:
            payload["contract_version"] = 1
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
