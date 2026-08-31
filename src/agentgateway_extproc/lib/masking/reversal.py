"""Reverse only placeholders recorded on the current ext_proc stream."""

# ruff: noqa: TRY003

from __future__ import annotations

import hashlib
import json
import logging
import re

from agentgateway_extproc.metrics import reversal_count_total
from agentgateway_extproc.models.types import (
    REVERSIBLE_CANDIDATE_RE,
    REVERSIBLE_TOKEN_RE,
)

_logger = logging.getLogger(__name__)
_PREFIXES = ("<REV_", "<ENCRYPTED_")
_MALFORMED_PLACEHOLDER_RE = re.compile(r"<(?:REV|ENCRYPTED)_[^<>\r\n]*(?=<|\r|\n|$)")
_INVALID_CLOSED_PLACEHOLDER_RE = re.compile(r"<(?:REV|ENCRYPTED)_[^<>\r\n]*>")


class PlaceholderStreamRewriter:
    """Reverse one semantic output channel without buffering the stream."""

    def __init__(
        self,
        reversal_map: dict[str, str],
        *,
        json_string_fragment: bool = False,
        mark_invalid: bool = False,
        entity_prefixes: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        """Store the stream-local validated reversal entries."""
        self._reversal_map = reversal_map
        self._json_string_fragment = json_string_fragment
        self._mark_invalid = mark_invalid
        self._entity_prefixes = (
            entity_prefixes
            if entity_prefixes is not None
            else placeholder_entity_prefixes(reversal_map)
            if mark_invalid
            else ()
        )
        self._pending = ""

    @property
    def marks_invalid(self) -> bool:
        """Return whether this human-readable channel degrades invalid tokens safely."""
        return self._mark_invalid

    def feed(self, text: str, *, final: bool = False) -> tuple[str, dict[str, int], int]:
        """Return safe output while retaining only a possible token suffix."""
        combined = self._pending + text
        try:
            safe, self._pending = _split_candidate_suffix(combined)
        except ValueError:
            if not self._mark_invalid:
                raise
            last_open = combined.rfind("<")
            safe = combined[:last_open] + invalid_placeholder_marker(
                combined[last_open:], self._reversal_map, self._entity_prefixes
            )
            self._pending = ""
            pending_misses = 1
        else:
            pending_misses = 0
        reversed_text, hits, misses = reverse_placeholders(
            safe,
            self._reversal_map,
            json_string_fragment=self._json_string_fragment,
            mark_invalid=self._mark_invalid,
            entity_prefixes=self._entity_prefixes,
        )
        if final:
            if self._pending:
                if not self._mark_invalid:
                    raise ValueError("response ended with an incomplete placeholder")
                reversed_text += invalid_placeholder_marker(
                    self._pending, self._reversal_map, self._entity_prefixes
                )
                self._pending = ""
                pending_misses += 1
            return reversed_text, hits, misses + pending_misses
        return reversed_text, hits, misses + pending_misses


def _split_candidate_suffix(text: str) -> tuple[str, str]:
    """Retain a bounded possible reserved placeholder only at the tail."""
    last_open = text.rfind("<")
    if last_open < 0:
        return text, ""
    suffix = text[last_open:]
    if ">" in suffix:
        return text, ""
    if any(prefix.startswith(suffix) or suffix.startswith(prefix) for prefix in _PREFIXES):
        if len(suffix) > 258:
            raise ValueError("response contained an oversized placeholder")
        return text[:last_open], suffix
    return text, ""


def reverse_placeholders(
    text: str,
    reversal_map: dict[str, str],
    *,
    json_string_fragment: bool = False,
    mark_invalid: bool = False,
    entity_prefixes: tuple[tuple[str, str], ...] | None = None,
) -> tuple[str, dict[str, int], int]:
    """Replace only request-scoped placeholders known to this ext_proc stream."""
    hits: dict[str, int] = {}
    misses = 0
    prefixes = (
        entity_prefixes
        if entity_prefixes is not None
        else placeholder_entity_prefixes(reversal_map)
        if mark_invalid
        else ()
    )
    matches = list(REVERSIBLE_CANDIDATE_RE.finditer(text))
    result = text
    for match in sorted(matches, key=lambda item: item.start(), reverse=True):
        placeholder = match.group(0)
        validated = REVERSIBLE_TOKEN_RE.fullmatch(placeholder)
        entity_type = validated.group(2) if validated is not None else "INVALID"
        original = reversal_map.get(placeholder)
        if original is None:
            misses += 1
            _logger.debug("No stream reversal entry for placeholder")
            if mark_invalid:
                marker = invalid_placeholder_marker(placeholder, reversal_map, prefixes)
                result = result[: match.start()] + marker + result[match.end() :]
            continue
        replacement = (
            json.dumps(original, ensure_ascii=False)[1:-1] if json_string_fragment else original
        )
        result = result[: match.start()] + replacement + result[match.end() :]
        hits[entity_type] = hits.get(entity_type, 0) + 1
        reversal_count_total.labels(entity_type=entity_type).inc()
    if mark_invalid:
        malformed = _remaining_malformed_placeholders(result)
        for match in reversed(malformed):
            marker = invalid_placeholder_marker(match.group(0), reversal_map, prefixes)
            result = result[: match.start()] + marker + result[match.end() :]
        misses += len(malformed)
    elif _remaining_malformed_placeholders(result):
        raise ValueError("response contained a malformed placeholder")
    return result, hits, misses


def invalid_placeholder_marker(
    placeholder: str,
    reversal_map: dict[str, str],
    entity_prefixes: tuple[tuple[str, str], ...] | None = None,
) -> str:
    """Return a bounded history-safe marker for one unauthorized placeholder."""
    prefixes = (
        entity_prefixes
        if entity_prefixes is not None
        else placeholder_entity_prefixes(reversal_map)
    )
    entity = next(
        (entity for prefix, entity in prefixes if placeholder.startswith(prefix)), "TOKEN"
    )
    fingerprint = hashlib.sha256(placeholder.encode()).hexdigest()[:6]
    return f"<PII_INVALID_{entity}_{fingerprint}>"


def placeholder_entity_prefixes(
    reversal_map: dict[str, str],
) -> tuple[tuple[str, str], ...]:
    """Index the bounded token prefixes needed for diagnostic entity attribution."""
    prefixes: set[tuple[str, str]] = set()
    for candidate in reversal_map:
        validated = REVERSIBLE_TOKEN_RE.fullmatch(candidate)
        if validated is not None:
            kind, entity_type = validated.group(1), validated.group(2)
            prefixes.add((f"<{kind}_{entity_type}_", entity_type))
    return tuple(sorted(prefixes, key=lambda item: len(item[0]), reverse=True))


def _remaining_malformed_placeholders(text: str) -> list[re.Match[str]]:
    """Find reserved-looking fragments not handled by the complete candidate grammar."""
    closed = [
        match
        for match in _INVALID_CLOSED_PLACEHOLDER_RE.finditer(text)
        if REVERSIBLE_CANDIDATE_RE.fullmatch(match.group(0)) is None
    ]
    return sorted(
        [*closed, *_MALFORMED_PLACEHOLDER_RE.finditer(text)], key=lambda item: item.start()
    )
