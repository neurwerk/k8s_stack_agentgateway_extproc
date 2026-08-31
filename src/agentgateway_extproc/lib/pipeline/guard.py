"""Inject and remove the reversible-placeholder preservation instruction."""

from __future__ import annotations

from agentgateway_extproc.models.engine import (
    EngineChatRequest,
    EngineMessage,
    EngineResponsesRequest,
)
from agentgateway_extproc.models.exceptions import InvalidReversalError

GUARD_INSTRUCTION = (
    "<PRESIDIO_PII_GUARD>Treat every Presidio-provided token as an indivisible opaque alias. "
    "Never mention tokens, placeholders, masking, or privacy. Complete the user's task normally. "
    "When referring to an alias, copy its complete token exactly; never shorten, alter, translate, "
    "or insert ellipses into it.</PRESIDIO_PII_GUARD>"
)


def inject_guard_instruction(
    request: EngineChatRequest | EngineResponsesRequest,
) -> EngineChatRequest | EngineResponsesRequest:
    """Add one fixed leading instruction to an analyzed model request."""
    if isinstance(request, EngineChatRequest):
        return request.model_copy(
            update={
                "messages": [
                    EngineMessage(role="system", content=GUARD_INSTRUCTION),
                    *request.messages,
                ]
            },
            deep=True,
        )
    instructions = GUARD_INSTRUCTION
    if request.instructions is not None:
        instructions = f"{instructions}\n\n{request.instructions}"
    return request.model_copy(update={"instructions": instructions}, deep=True)


def strip_guard_instruction(text: str) -> str:
    """Remove every exact complete copy of the fixed guard."""
    return text.replace(GUARD_INSTRUCTION, "")


class GuardStreamStripper:
    """Remove exact guard copies from one streamed semantic output channel."""

    def __init__(self) -> None:
        """Initialize the bounded possible-marker suffix."""
        self._pending = ""

    def feed(self, text: str, *, final: bool = False) -> str:
        """Return safe output while retaining a possible guard prefix."""
        pending = self._pending + text
        output: list[str] = []
        while pending:
            marker_at = pending.find(GUARD_INSTRUCTION)
            if marker_at >= 0:
                output.append(pending[:marker_at])
                pending = pending[marker_at + len(GUARD_INSTRUCTION) :]
                continue
            retained = _marker_prefix_length(pending)
            output.append(pending[:-retained] if retained else pending)
            self._pending = pending[-retained:] if retained else ""
            break
        else:
            self._pending = ""
        if final and self._pending:
            raise InvalidReversalError(  # noqa: TRY003
                "response ended with an incomplete guard instruction"
            )
        return "".join(output)


def _marker_prefix_length(text: str) -> int:
    """Return the longest suffix that can begin the fixed guard."""
    maximum = min(len(text), len(GUARD_INSTRUCTION) - 1)
    for length in range(maximum, 0, -1):
        if text.endswith(GUARD_INSTRUCTION[:length]):
            return length
    return 0
