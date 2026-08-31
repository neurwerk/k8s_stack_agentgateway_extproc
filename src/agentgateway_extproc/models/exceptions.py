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
