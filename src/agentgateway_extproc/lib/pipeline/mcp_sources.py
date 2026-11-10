"""Bounded, unauthenticated MCP source labels carried in ordinary text."""

from __future__ import annotations

import re
from collections.abc import Iterable

from agentgateway_extproc.models.exceptions import GatewayPolicyError

SOURCE_PATTERN = r"^MCP_TOOL_[A-Z0-9_]+$"
DEFAULT_SOURCE = "MCP_TOOL_DEFAULT"
MAX_SOURCES = 32
_OPEN = "<NEURWERK_MCP_SOURCES>"
_CLOSE = "</NEURWERK_MCP_SOURCES>"
_TAGS = re.compile(f"({re.escape(_OPEN)}|{re.escape(_CLOSE)})")


def source_union(*groups: Iterable[str]) -> list[str]:
    """Keep every classification or reject, never truncate a restriction."""
    sources = set().union(*groups)
    if len(sources) > MAX_SOURCES:
        raise GatewayPolicyError("invalid_request")
    return sorted(sources)


def collect_sources(texts: Iterable[str]) -> list[str]:
    """Recognize exact envelopes, assigning malformed ones to the default policy."""
    sources: list[str] = []
    for text in texts:
        pending: str | None = None
        for part in _TAGS.split(text):
            if part == _OPEN:
                if pending is not None:
                    sources = source_union(sources, [DEFAULT_SOURCE])
                pending = ""
            elif part == _CLOSE:
                values = [] if pending is None else pending.split(",")
                values = [value.strip(" \t\r\n\v\f") for value in values]
                valid = bool(values) and all(
                    len(value) <= 64 and re.fullmatch(SOURCE_PATTERN, value) for value in values
                )
                sources = source_union(sources, values if valid else [DEFAULT_SOURCE])
                pending = None
            elif pending is not None:
                pending += part
        if pending is not None:
            sources = source_union(sources, [DEFAULT_SOURCE])
    return sources


def source_marker(sources: list[str]) -> str:
    """Render the canonical complete label before any generated text."""
    return f"{_OPEN}{','.join(sources)}{_CLOSE}\n" if sources else ""


def prefix_sources(text: str, sources: list[str]) -> str:
    """Prefix a complete carrier without duplicating an already canonical prefix."""
    marker = source_marker(sources)
    return text if not marker or text.startswith(marker) else marker + text
