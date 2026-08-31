"""Derive opaque policy-session keys from trusted destination context."""

from __future__ import annotations

import hashlib

from agentgateway_extproc.models.destination import DestinationPolicy
from agentgateway_extproc.models.engine import EngineMcpRequest, EngineRequest

_SESSION_KEY_VERSION = "neurwerk-pii-session-v2"


def make_session_key(
    policy: DestinationPolicy,
    request: EngineRequest,
    *,
    conversation_id: str | None = None,
    mcp_session_id: str | None = None,
    request_nonce: bytes | None = None,
) -> str:
    """Hash length-framed trusted identity, destination, and session components."""
    if isinstance(request, EngineMcpRequest):
        if policy.destination_kind != "mcp":
            raise ValueError("MCP request requires MCP destination metadata")  # noqa: TRY003
        session_kind = "session" if mcp_session_id is not None else "request"
        if mcp_session_id is not None:
            reference = mcp_session_id
        elif request_nonce is not None:
            reference = request_nonce.hex()
        else:
            raise ValueError("request-scoped MCP session requires a nonce")  # noqa: TRY003
        destination_id = policy.destination_id
    else:
        if policy.destination_kind != "model":
            raise ValueError("model request requires model destination metadata")  # noqa: TRY003
        if conversation_id is not None:
            session_kind = "conversation"
            reference = conversation_id[:256]
        elif request_nonce is not None:
            session_kind = "request"
            reference = request_nonce.hex()
        else:
            raise ValueError("model request requires a conversation or request nonce")  # noqa: TRY003
        destination_id = request.model
    encoded = _length_frame(
        (
            ("version", _SESSION_KEY_VERSION.encode()),
            ("principal", policy.principal_id.encode()),
            ("destination_kind", policy.destination_kind.encode()),
            ("destination_id", destination_id.encode()),
            ("session_kind", session_kind.encode()),
            ("reference", reference.encode()),
        )
    )
    return hashlib.sha256(encoded).hexdigest()


def _length_frame(parts: tuple[tuple[str, bytes], ...]) -> bytes:
    encoded = bytearray()
    for label, value in parts:
        label_bytes = label.encode("ascii")
        encoded.extend(len(label_bytes).to_bytes(2, "big"))
        encoded.extend(label_bytes)
        encoded.extend(len(value).to_bytes(4, "big"))
        encoded.extend(value)
    return bytes(encoded)
