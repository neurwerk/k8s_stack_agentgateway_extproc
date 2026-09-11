"""Domain exceptions for engine and stream processing failures."""

from __future__ import annotations

from typing import Final, Literal

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


class EnginePolicyError(Exception):
    """Carry one validated and status-bound PII Engine rejection."""

    __slots__ = ("code", "message", "retryable", "status_code")

    def __init__(self, code: EngineErrorCode) -> None:
        """Derive every client-visible field from one recognized error code."""
        contract = ENGINE_ERROR_CONTRACT.get(code)
        if contract is None:
            raise ValueError
        super().__init__()
        self.status_code, self.message, self.retryable = contract
        self.code = code


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


def is_safe_engine_error_message(message: object) -> bool:
    """Accept one bounded printable line suitable for a JSON client error."""
    return (
        isinstance(message, str)
        and 0 < len(message) <= MAX_ENGINE_ERROR_MESSAGE_LENGTH
        and message == message.strip()
        and all(character.isprintable() for character in message)
    )
