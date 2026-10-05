"""Bound JSON structure before the decoder allocates its object tree."""

from __future__ import annotations

import re

from agentgateway_extproc.models.exceptions import LimitDetail

MAX_JSON_DEPTH = 64
MAX_JSON_TOKENS = 200_000
# Possessive string runs avoid both per-character objects and regex backtracking
# stacks for long base64 or escaped text. Matches are consumed lazily, not copied.
_TOKENS = re.compile(r'"(?:[^"\\]++|\\.)*+"|[{}\[\]]|[^ \t\r\n{}\[\]",:]+|[ \t\r\n,:]++')


class JsonBudgetError(ValueError):
    """Reject structural expansion independently of the wire-byte limit."""

    def __init__(self, *, limit: LimitDetail | None = None) -> None:
        """Retain the observed prefix count when scanning stops early."""
        super().__init__()
        self.limit = limit


def bounded_json_text(value: str | bytes) -> str:
    """Check depth and lexical tokens without replacing strict JSON validation."""
    text = value.decode("utf-8") if isinstance(value, bytes) else value
    stack: list[str] = []
    position = count = 0
    while position < len(text):
        # Anchor every token: retrying at later quotes in a malformed, unterminated
        # string would repeatedly scan its suffix and turn this guard quadratic.
        token = _TOKENS.match(text, position)
        if token is None:
            raise ValueError("invalid JSON token")  # noqa: TRY003
        char = text[position]
        position = token.end()
        if char in " \t\r\n,:":
            continue
        count += 1
        if count > MAX_JSON_TOKENS:
            raise JsonBudgetError(
                limit=LimitDetail(
                    stage="json",
                    reason="tokens",
                    measured=count,
                    maximum=MAX_JSON_TOKENS,
                    unit="items",
                    exact=False,
                )
            )
        if char in "{[":
            stack.append(char)
            if len(stack) > MAX_JSON_DEPTH:
                raise JsonBudgetError(
                    limit=LimitDetail(
                        stage="json",
                        reason="depth",
                        measured=len(stack),
                        maximum=MAX_JSON_DEPTH,
                        unit="levels",
                        exact=False,
                    )
                )
        elif char in "}]" and (not stack or stack.pop() != {"}": "{", "]": "["}[char]):
            raise ValueError("invalid JSON structure")  # noqa: TRY003
    return text
