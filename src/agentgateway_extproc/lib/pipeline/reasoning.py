"""Digest-only continuation provenance, shared by JSON, SSE and request replay."""

# ruff: noqa: TRY003

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from neurwerk_request_segments.models import EngineChatRequest, EngineResponsesRequest

from agentgateway_extproc.models.exceptions import (
    GatewayPolicyError,
    InvalidReversalError,
    LimitDetail,
)

if TYPE_CHECKING:
    from _hashlib import HASH

    from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
    from agentgateway_extproc.models.admission import RequestContext

MAX_RESPONSE_BLOCKS = 256


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reasoning_scope(context: RequestContext) -> str:
    """Bind evidence to verified identity, credential mode, dispatch and gateway policy."""
    return _digest(
        [
            "reasoning-scope-v1",
            context.principal_id,
            context.credential_id,
            context.provider_id,
            context.provider_credential_source,
            context.protocol_profile,
            context.dispatch_binding_id,
            context.dispatch_destination_id,
            context.resolved_model,
            context.policy_revision,
            context.pii_policy.enabled,
        ]
    )


@dataclass
class _Fingerprint:
    carrier: str
    values: dict[str, HASH] = field(default_factory=dict)
    opaque: bool = False

    def append(self, value: dict[str, Any]) -> None:
        self.opaque |= bool(value.get("signature") or value.get("reasoning_signature")) or (
            value.get("type") in {"reasoning.encrypted", "redacted_thinking"}
        )
        for key, text in value.items():
            if key == "index" or not isinstance(text, str) or not text:
                continue
            if key in {"type", "id", "format"} and key in self.values:
                if self.values[key].digest() != sha256(text.encode()).digest():
                    raise InvalidReversalError("reasoning identity changed")
            else:
                self.values.setdefault(key, sha256()).update(text.encode())

    def digest(self) -> str:
        return _digest(
            [self.carrier, {key: value.hexdigest() for key, value in self.values.items()}]
        )


class ChatReasoningTracker:
    """Hash fragmented signed fields without retaining their text or signature bytes."""

    def __init__(self) -> None:
        """Retain bounded hash state only for the current response."""
        self.blocks: dict[tuple[int, str, int], _Fingerprint] = {}

    def append(self, choice: int, message: dict[str, Any]) -> None:
        """Accept validated complete fields or indexed fragments."""
        signature = {
            key: message[key]
            for key in ("reasoning", "reasoning_content", "reasoning_signature")
            if message.get(key) is not None
        }
        if signature:
            self._append((choice, "signature", 0), signature)
        for field_name in ("reasoning_details", "thinking_blocks"):
            seen: set[int] = set()
            for position, block in enumerate(message.get(field_name) or []):
                index = block.get("index")
                index = position if index is None else index
                if index in seen:
                    raise InvalidReversalError("duplicate reasoning coordinate")
                seen.add(index)
                self._append((choice, field_name, index), block)

    def _append(self, key: tuple[int, str, int], block: dict[str, Any]) -> None:
        if key not in self.blocks:
            if len(self.blocks) >= MAX_RESPONSE_BLOCKS:
                raise InvalidReversalError("too many reasoning blocks")
            self.blocks[key] = _Fingerprint("chat." + key[1])
        self.blocks[key].append(block)

    def digests(self) -> set[str]:
        """Return evidence for opaque fields, including signatures received last."""
        return {block.digest() for block in self.blocks.values() if block.opaque}


def response_reasoning_digest(item: dict[str, Any]) -> str | None:
    """Only the encrypted value is immutable; summaries remain editable input text."""
    encrypted = item.get("encrypted_content")
    return _digest(["responses.encrypted_content", encrypted]) if encrypted else None


def response_receipts(kind: str, payload: dict[str, Any]) -> set[str]:
    """Fingerprint a complete, already validated provider response."""
    if kind == "chat":
        tracker = ChatReasoningTracker()
        for choice in payload["choices"]:
            tracker.append(choice["index"], choice["message"])
        return tracker.digests()
    return {
        digest
        for item in payload["output"]
        if item.get("type") == "reasoning" and (digest := response_reasoning_digest(item))
    }


async def verify_reasoning(
    handler: StreamHandler,
    request: EngineChatRequest | EngineResponsesRequest,
    *,
    policy_version: str | None = None,
) -> None:
    """Require approved evidence before an immutable history field bypasses inspection."""
    digests: set[str] = set()
    if isinstance(request, EngineChatRequest):
        for message in request.messages:
            tracker = ChatReasoningTracker()
            tracker.append(0, message.model_dump(exclude_none=True))
            digests.update(tracker.digests())
    elif isinstance(request.input, list):
        digests = {
            digest
            for item in request.input
            if item.type == "reasoning" and (digest := response_reasoning_digest(item.model_dump()))
        }
    if not digests:
        return
    if len(digests) > handler.max_reasoning_fingerprints:
        raise GatewayPolicyError(
            "reasoning_history_limit",
            limit=LimitDetail(
                stage="admission",
                reason="reasoning_fingerprints",
                measured=len(digests),
                maximum=handler.max_reasoning_fingerprints,
                unit="items",
                exact=True,
            ),
        )
    if handler.request_context is None:
        raise GatewayPolicyError("reasoning_context_missing")
    await handler.client.reasoning_receipts(
        reasoning_scope(handler.request_context),
        sorted(digests),
        record=False,
        correlation_id=handler.correlation_id,
        policy_version=policy_version,
    )
