"""Domain exceptions for engine and stream processing failures."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LimitDetail(BaseModel):
    """Carry content-free measurements from the boundary that rejected work."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    component: Literal["pii_engine", "extproc", "request_segments"] = "extproc"
    stage: Literal[
        "admission",
        "json",
        "inspection",
        "engine_request",
        "engine_response",
        "provider_response",
        "output",
    ]
    reason: Literal[
        "bytes",
        "declared_bytes",
        "encoded_bytes",
        "decoded_bytes",
        "transformed_bytes",
        "depth",
        "tokens",
        "nodes",
        "text_characters",
        "segments",
        "text_leaves",
        "empty_chunks",
    ]
    measured: int = Field(ge=0)
    maximum: int = Field(ge=0)
    unit: Literal["bytes", "characters", "items", "levels"]
    exact: bool

    @model_validator(mode="after")
    def validate_measurement(self) -> LimitDetail:
        """Reject contradictory units or a measurement that did not exceed its limit."""
        expected_unit = {
            "bytes": "bytes",
            "declared_bytes": "bytes",
            "encoded_bytes": "bytes",
            "decoded_bytes": "bytes",
            "transformed_bytes": "bytes",
            "depth": "levels",
            "text_characters": "characters",
        }.get(self.reason, "items")
        if self.measured <= self.maximum or self.unit != expected_unit:
            raise ValueError("invalid limit measurement")  # noqa: TRY003
        return self


type EngineErrorCode = Literal[
    "invalid_request",
    "request_too_large",
    "capacity_unavailable",
    "runtime_unavailable",
    "analysis_timeout",
    "internal_error",
]

ENGINE_ERROR_CONTRACT: Final[dict[EngineErrorCode, tuple[int, str, bool]]] = {
    "invalid_request": (400, "The analysis request is invalid.", False),
    "request_too_large": (
        413,
        "The analysis request exceeds the configured size limit.",
        False,
    ),
    "capacity_unavailable": (
        503,
        "Analysis capacity is temporarily unavailable.",
        True,
    ),
    "runtime_unavailable": (503, "The analysis runtime is unavailable.", True),
    "analysis_timeout": (504, "Analysis timed out.", True),
    "internal_error": (500, "Analysis failed.", False),
}
MAX_ENGINE_ERROR_MESSAGE_LENGTH: Final = 512


class EngineUnavailableError(Exception):
    """Indicate that the policy engine could not provide a safe answer."""


class InvalidEngineReplyError(Exception):
    """Indicate that an engine reply did not match the strict adapter contract."""

    def __init__(
        self,
        message: str = "",
        *,
        limit: LimitDetail | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Retain measured response limits while preserving rejection behavior."""
        super().__init__(message)
        self.limit = limit
        self.correlation_id = correlation_id


class EnginePolicyError(Exception):
    """Carry one validated and status-bound PII Engine rejection."""

    __slots__ = ("code", "correlation_id", "limit", "message", "retryable", "status_code")

    def __init__(
        self,
        code: EngineErrorCode,
        *,
        limit: LimitDetail | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Derive every client-visible field from one recognized error code."""
        contract = ENGINE_ERROR_CONTRACT.get(code)
        if contract is None:
            raise ValueError
        super().__init__()
        self.status_code, self.message, self.retryable = contract
        self.code = code
        self.limit = limit
        self.correlation_id = correlation_id


class InvalidReversalError(Exception):
    """Indicate that a response contains an untrusted or unknown placeholder."""


class McpHttpError(Exception):
    """Carry an HTTP rejection without trusting backend bodies or authentication headers."""

    def __init__(self, status_code: int, headers: dict[str, str]) -> None:
        """Keep only bounded transport hints with a closed value vocabulary."""
        if not 400 <= status_code <= 599:
            raise ValueError
        super().__init__()
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        allow = headers.get("allow", "")
        methods = [method.strip() for method in allow.split(",")]
        if (
            status_code == 405
            and len(allow) <= 128
            and all(
                method
                in {"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"}
                for method in methods
            )
        ):
            self.headers["allow"] = ", ".join(dict.fromkeys(methods))
        retry_after = headers.get("retry-after", "")
        if (
            status_code in {429, 503}
            and 0 < len(retry_after) <= 5
            and (retry_after.isascii() and retry_after.isdecimal() and int(retry_after) <= 86400)
        ):
            self.headers["retry-after"] = retry_after


class TrustedMetadataError(Exception):
    """Indicate missing, malformed, or changing trusted destination metadata."""


class ContextForgeAccountRequiredError(Exception):
    """Reject a ContextForge destination without a valid verified account email."""


def is_safe_engine_error_message(message: object) -> bool:
    """Accept one bounded printable line suitable for a JSON client error."""
    return (
        isinstance(message, str)
        and 0 < len(message) <= MAX_ENGINE_ERROR_MESSAGE_LENGTH
        and message == message.strip()
        and all(character.isprintable() for character in message)
    )
