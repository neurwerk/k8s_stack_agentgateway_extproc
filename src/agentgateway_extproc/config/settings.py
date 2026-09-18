"""Environment-backed settings for the ext_proc adapter."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MEBIBYTE = 1_048_576
MAX_REQUEST_BYTES = 5 * MEBIBYTE
MAX_RESPONSE_BYTES = 10 * MEBIBYTE
DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_BYTES = 6 * MEBIBYTE + 65_536
MAX_UPLOAD_REQUEST_BYTES = 64 * MEBIBYTE
MAX_GRPC_RECEIVE_MESSAGE_BYTES = 65 * MEBIBYTE + 65_536
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


class DoclingSettings(BaseModel):
    """Bound the private conversion service and one admitted document batch."""

    enabled: bool = False
    base_url: str = "https://docling.docling.svc"
    ca_cert: str | None = None
    api_key: SecretStr | None = None
    inference_mode: Literal["internal-standard", "private-vlm", "cpu", "remote"] = (
        "internal-standard"
    )
    timeout: float = Field(default=360, gt=0, le=3660)
    document_timeout: float = Field(default=300, gt=0, le=3600)
    file_bytes: int = Field(default=20 * MEBIBYTE, ge=1, le=40 * MEBIBYTE)
    total_bytes: int = Field(default=40 * MEBIBYTE, ge=1, le=40 * MEBIBYTE)
    count: int = Field(default=5, ge=1, le=20)
    pages: int = Field(default=200, ge=1, le=1000)
    max_response_bytes: int = Field(default=16 * MEBIBYTE, ge=1024, le=16 * MEBIBYTE)

    @model_validator(mode="after")
    def validate_service(self) -> DoclingSettings:
        """Require verified HTTPS and a service-only credential when enabled."""
        url = urlsplit(self.base_url)
        if self.enabled and (
            url.scheme != "https"
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
            or not self.api_key
            or not self.api_key.get_secret_value()
        ):
            raise ValueError("Docling requires an HTTPS origin and API key")  # noqa: TRY003
        return self


class Settings(BaseSettings):
    """Load adapter configuration from ``EXTPROC_`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="EXTPROC_", env_nested_delimiter="__")

    engine: EngineSettings = Field(default_factory=EngineSettings)
    docling: DoclingSettings = Field(default_factory=DoclingSettings)
    debug: bool = False
    max_request_bytes: int = Field(default=MAX_REQUEST_BYTES, ge=1_024, le=MAX_UPLOAD_REQUEST_BYTES)
    max_response_bytes: int = Field(default=MAX_RESPONSE_BYTES, ge=1_024, le=MAX_RESPONSE_BYTES)
    max_transformed_request_bytes: int = Field(
        default=MAX_RESPONSE_BYTES, ge=1_024, le=MAX_RESPONSE_BYTES
    )
    grpc_max_receive_message_bytes: int = Field(
        default=DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_BYTES,
        ge=1_024,
        le=MAX_GRPC_RECEIVE_MESSAGE_BYTES,
    )
    grpc_maximum_concurrent_rpcs: int = Field(default=4, ge=1, le=16)

    @model_validator(mode="after")
    def validate_transport_limits(self) -> Settings:
        """Keep the gRPC envelope larger than every accepted request body."""
        if self.grpc_max_receive_message_bytes <= self.max_request_bytes:
            raise ValueError("grpc receive limit must exceed max_request_bytes")  # noqa: TRY003
        return self


def get_settings() -> Settings:
    """Create the current process settings."""
    return Settings()
