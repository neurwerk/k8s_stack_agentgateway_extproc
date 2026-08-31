"""Environment-backed settings for the ext_proc adapter."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MEBIBYTE = 1_048_576
MAX_REQUEST_BYTES = 5 * MEBIBYTE
MAX_RESPONSE_BYTES = 10 * MEBIBYTE
DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_BYTES = 6 * MEBIBYTE + 65_536
MAX_ENGINE_TIMEOUT_SECONDS = 615.0


class EngineSettings(BaseModel):
    """Configure the typed PII engine HTTP client."""

    base_url: str = "http://pii-engine:8000"
    ca_cert: str | None = None
    client_cert: str | None = None
    client_key: str | None = None
    timeout: float = Field(default=5.0, gt=0, le=MAX_ENGINE_TIMEOUT_SECONDS)
    readiness_timeout: float = Field(default=1.0, gt=0, le=1)
    max_response_bytes: int = Field(default=MAX_RESPONSE_BYTES, ge=1_024, le=MAX_RESPONSE_BYTES)

    @model_validator(mode="after")
    def validate_client_certificate(self) -> EngineSettings:
        """Require both client certificate files when mTLS is enabled."""
        if (self.client_cert is None) != (self.client_key is None):
            raise ValueError("client_cert and client_key must be configured together")  # noqa: TRY003
        return self


class Settings(BaseSettings):
    """Load adapter configuration from ``EXTPROC_`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="EXTPROC_", env_nested_delimiter="__")

    engine: EngineSettings = Field(default_factory=EngineSettings)
    debug: bool = False
    max_request_bytes: int = Field(default=MAX_REQUEST_BYTES, ge=1_024, le=MAX_REQUEST_BYTES)
    max_response_bytes: int = Field(default=MAX_RESPONSE_BYTES, ge=1_024, le=MAX_RESPONSE_BYTES)
    max_transformed_request_bytes: int = Field(
        default=MAX_RESPONSE_BYTES, ge=1_024, le=MAX_RESPONSE_BYTES
    )
    grpc_max_receive_message_bytes: int = Field(
        default=DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_BYTES,
        ge=1_024,
        le=DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_BYTES,
    )

    @model_validator(mode="after")
    def validate_transport_limits(self) -> Settings:
        """Keep the gRPC envelope larger than every accepted request body."""
        if self.grpc_max_receive_message_bytes <= self.max_request_bytes:
            raise ValueError("grpc receive limit must exceed max_request_bytes")  # noqa: TRY003
        return self


def get_settings() -> Settings:
    """Create the current process settings."""
    return Settings()
