"""Validate engine mutations and request-local reversal provenance."""

from __future__ import annotations

import re
from typing import cast

from agentgateway_extproc.lib.request_segments import extract_request
from agentgateway_extproc.models.engine import EngineReply, EngineRequest
from agentgateway_extproc.models.exceptions import InvalidEngineReplyError
from agentgateway_extproc.models.types import (
    RESERVED_PLACEHOLDER_PREFIX_RE,
    REVERSIBLE_CANDIDATE_RE,
    REVERSIBLE_TOKEN_RE,
)

type PathPart = str | int
type TextLeaves = dict[tuple[PathPart, ...], str]


def validate_reversal(original: EngineRequest, reply: EngineReply) -> None:
    """Require reversal tokens, plaintext and counts to match the current request."""
    transformed_request = reply.request
    if RESERVED_PLACEHOLDER_PREFIX_RE.search(original.model_dump_json(exclude_none=True)):
        raise InvalidEngineReplyError(  # noqa: TRY003
            "reversal placeholder already existed in request"
        )
    if transformed_request is None:
        if reply.reversal:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "transformed request and reversal entries differ"
            )
        return
    original_leaves = _mutable_text_leaves(original)
    transformed_leaves = _mutable_text_leaves(transformed_request)
    entities_by_placeholder = _reversal_entities(reply)
    seen, restored_by_entity = _restore_reversal_leaves(
        original_leaves,
        transformed_leaves,
        reply.reversal,
        entities_by_placeholder,
    )
    if seen != set(reply.reversal):
        raise InvalidEngineReplyError(  # noqa: TRY003
            "transformed request and reversal entries differ"
        )
    if restored_by_entity != _expected_reversal_counts(reply):
        raise InvalidEngineReplyError(  # noqa: TRY003
            "reversal occurrence counts disagree with the current request report"
        )


def _restore_reversal_leaves(
    original_leaves: TextLeaves,
    transformed_leaves: TextLeaves,
    reversal: dict[str, str],
    entities_by_placeholder: dict[str, str],
) -> tuple[set[str], dict[str, int]]:
    """Scan candidate tokens once and validate plaintext against its source leaf."""
    seen: set[str] = set()
    restored_by_entity: dict[str, int] = {}
    consumed: dict[tuple[tuple[PathPart, ...], str], int] = {}
    current_path: tuple[PathPart, ...] = ()

    def restore(match: re.Match[str]) -> str:
        placeholder = match.group(0)
        plaintext = reversal.get(placeholder)
        if plaintext is None:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "transformed request and reversal entries differ"
            )
        seen.add(placeholder)
        entity_type = entities_by_placeholder[placeholder]
        restored_by_entity[entity_type] = restored_by_entity.get(entity_type, 0) + 1
        key = (current_path, plaintext)
        consumed[key] = consumed.get(key, 0) + 1
        return plaintext

    for path, transformed_text in transformed_leaves.items():
        current_path = path
        REVERSIBLE_CANDIDATE_RE.sub(restore, transformed_text)
    for (path, plaintext), count in consumed.items():
        if original_leaves.get(path, "").count(plaintext) < count:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "reversal plaintext does not match its original text leaf"
            )
    return seen, restored_by_entity


def _reversal_entities(reply: EngineReply) -> dict[str, str]:
    """Validate reversal keys against the current report in one linear pass."""
    if reply.analysis.source != "current_request":
        if reply.reversal:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "reversal entries require a current request report"
            )
        return {}
    rows = {row.entity_type: row for row in reply.report.rows}
    entities: dict[str, str] = {}
    for placeholder in reply.reversal:
        token = REVERSIBLE_TOKEN_RE.fullmatch(placeholder)
        if token is None:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "reversal contains an invalid placeholder"
            )
        entity_type = cast(str, token.group(2))
        row = rows.get(entity_type)
        if row is None or row.action not in {"encrypt", "reversible_replace"}:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "reversal entries disagree with the current request report"
            )
        if not row.transformed_count:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "reversal entries require a transformed report row"
            )
        entities[placeholder] = entity_type
    return entities


def _expected_reversal_counts(reply: EngineReply) -> dict[str, int]:
    """Return placeholder counts required by current-request analysis."""
    if reply.analysis.source == "cached_decision":
        return {}
    return {
        row.entity_type: row.transformed_count
        for row in reply.report.rows
        if row.action in {"encrypt", "reversible_replace"} and row.transformed_count
    }


def validate_request_mutation(original: EngineRequest, reply: EngineReply) -> None:
    """Reject engine mutations outside schema-designated model-visible text leaves."""
    if reply.request is None:
        return
    if (
        type(reply.request) is not type(original)
        or extract_request(reply.request).control_shape()
        != extract_request(original).control_shape()
    ):
        raise InvalidEngineReplyError(  # noqa: TRY003
            "engine reply changed request protocol controls"
        )


def _mutable_text_leaves(request: EngineRequest) -> TextLeaves:
    """Return model-visible strings keyed by structural path."""
    extracted = extract_request(request)
    return {extracted.diagnostic_path(segment.id): segment.text for segment in extracted.segments}
