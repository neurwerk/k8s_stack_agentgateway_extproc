"""Closed native-operation recipes and credential-free header projections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

from agentgateway_extproc.gen import ext_proc_pb2, shared_envoy_pb2
from agentgateway_extproc.models.admission import AdmissionEnvelope
from agentgateway_extproc.models.exceptions import GatewayPolicyError
from agentgateway_extproc.models.types import PROVIDER_CREDENTIAL_HEADERS, REQUEST_HEADERS


@dataclass(frozen=True)
class ModelProfile:
    """Keep parsing and credential admission in one reviewed operation recipe."""

    request_kind: Literal["chat", "responses", "anthropic", "google"]
    path: str
    credential_headers: frozenset[str] = frozenset({"authorization"})
    forward_proxy_path: str | None = None


_PROFILES = {
    "openai-chat-completions-v1": ModelProfile("chat", "/v1/chat/completions"),
    "openai-responses-v1": ModelProfile("responses", "/v1/responses"),
    "openai-codex-responses-v1": ModelProfile(
        "responses", "/v1/responses", forward_proxy_path="/backend-api/codex/responses"
    ),
    "anthropic-messages-v1": ModelProfile("anthropic", "/v1/messages", frozenset({"x-api-key"})),
    "google-generate-content-v1": ModelProfile("google", "", frozenset({"x-goog-api-key"})),
}
_RETAINED_HEADERS = frozenset(
    {
        ":method",
        ":path",
        "content-type",
        "content-encoding",
        "content-length",
        "x-session-id",
        "x-conversation-id",
        "anthropic-version",
        "accept",
        "mcp-protocol-version",
        "mcp-session-id",
        "mcp-method",
        "mcp-name",
        "last-event-id",
    }
)


def model_profile(admission: AdmissionEnvelope) -> ModelProfile:
    """Reject known but unimplemented profiles and operations explicitly."""
    context = admission.request_context
    profile = _PROFILES.get(context.protocol_profile)
    operations = (
        {"generateContent", "streamGenerateContent"}
        if profile is not None and profile.request_kind == "google"
        else {"create"}
    )
    if (
        profile is None
        or context.destination_kind != "model"
        or context.operation_id not in operations
    ):
        raise GatewayPolicyError("unsupported_feature")
    return profile


def google_model_path(path: str, operation: str) -> str:
    """Admit only the documented REST text operations, with no credential query parameters."""
    suffix = r"\?alt=sse" if operation == "streamGenerateContent" else ""
    matched = re.fullmatch(r"/v1beta/models/([A-Za-z0-9_./:-]+):" + operation + suffix, path)
    if matched is None:
        raise GatewayPolicyError("invalid_request")
    return cast(str, matched.group(1))


def operation_path(path: str, admission: AdmissionEnvelope) -> str:
    """Normalize only the verified API mount for validation, never an upstream mutation."""
    context = admission.request_context
    prefix = context.api_path_prefix or f"/bindings/{context.binding_id}"
    if context.entry_mode == "api":
        if not path.startswith(prefix + "/"):
            raise GatewayPolicyError("invalid_request")
        return path[len(prefix) :]
    return path


def header_value(header: shared_envoy_pb2.HeaderValue) -> str:
    """Read a transient header without retaining its protobuf event."""
    return header.raw_value.decode(errors="replace") if header.raw_value else header.value


def request_headers(  # noqa: C901
    request: ext_proc_pb2.ProcessingRequest, admission: AdmissionEnvelope
) -> tuple[dict[str, str], list[str]]:
    """Validate transport while keeping credential values out of retained state."""
    items = request.request_headers.headers.headers
    names = [item.key.lower() for item in items]
    if any(item.raw_value and item.value for item in items) or (
        admission.request_context.destination_kind == "model"
        and any(names.count(name) > 1 for name in _RETAINED_HEADERS | REQUEST_HEADERS)
    ):
        raise GatewayPolicyError("invalid_request")
    headers: dict[str, str] = {
        item.key.lower(): header_value(item)
        for item in items
        if item.key.lower() in _RETAINED_HEADERS
    }
    if admission.request_context.destination_kind == "mcp":
        admission.resolve_mcp()
        # The MCP validator rejects mirrored parameter headers when inspection is
        # enabled. Retain their presence, never their uninspected values.
        headers.update({name: "" for name in names if name.startswith("mcp-param-")})
        return headers, [name for name in names if name in REQUEST_HEADERS]
    profile = model_profile(admission)
    path = operation_path(headers.get(":path", ""), admission)
    expected_path = (
        profile.forward_proxy_path
        if admission.request_context.entry_mode == "forward_proxy" and profile.forward_proxy_path
        else profile.path
    )
    if (
        headers.get(":method") != "POST"
        or (profile.request_kind != "google" and path != expected_path)
        or re.fullmatch(
            r'application/json(?:\s*;\s*charset\s*=\s*(?:utf-8|"utf-8"))?',
            headers.get("content-type", "").strip(),
            re.IGNORECASE,
        )
        is None
        or headers.get("content-encoding", "identity").strip().lower() not in {"", "identity"}
        or (
            "content-length" in headers
            and re.fullmatch(r"[0-9]{1,20}", headers["content-length"]) is None
        )
    ):
        raise GatewayPolicyError("invalid_request")
    if profile.request_kind == "google":
        google_model_path(path, admission.request_context.operation_id)
    if profile.request_kind == "anthropic" and (
        headers.get("anthropic-version") != "2023-06-01" or "anthropic-beta" in names
    ):
        raise GatewayPolicyError("unsupported_feature")
    source = admission.request_context.provider_credential_source
    supplied = set(names) & PROVIDER_CREDENTIAL_HEADERS
    preserved: frozenset[str] = frozenset()
    if source == "override":
        if not supplied:
            raise GatewayPolicyError("provider_credential_required")
        if supplied != profile.credential_headers:
            raise GatewayPolicyError("invalid_provider_credential")
        for item in items:
            if (
                item.key.lower() in supplied
                and re.fullmatch(
                    r"(?i:Bearer) [A-Za-z0-9._~+/=-]{1,16384}"
                    if item.key.lower() == "authorization"
                    else r"[A-Za-z0-9._~+/=-]{1,16384}",
                    header_value(item),
                )
                is None
            ):
                raise GatewayPolicyError("invalid_provider_credential")
        preserved = profile.credential_headers
    elif source == "none" and supplied:
        raise GatewayPolicyError("invalid_provider_credential")
    return headers, [name for name in names if name in REQUEST_HEADERS and name not in preserved]
