"""Shared protobuf and engine fixtures."""

from __future__ import annotations

import json

import httpx
import pytest
from neurwerk_request_segments import extract_request, parse_request

from agentgateway_extproc.config.settings import EngineSettings
from agentgateway_extproc.gen import ext_proc_pb2, shared_envoy_pb2
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.models.destination import DESTINATION_POLICY_NAMESPACE

REVERSIBLE_TOKEN = "<REV_PERSON_0123456789abcdef_fedcba9876543210>"
MODEL_POLICY: dict[str, object] = {
    "contract_version": 1,
    "destination_kind": "model",
    "principal_id": "principal-1",
    "models": {"test": True},
}


def segment_reply(reply: dict[str, object]) -> dict[str, object]:
    """Serialize internal pipeline fixtures as the actual v2 segment contract."""
    result = {**reply, "api_version": "v2"}
    request = result.pop("request", None)
    result["segments"] = (
        [segment.model_dump() for segment in extract_request(parse_request(request)).segments]
        if request is not None
        else None
    )
    return result


def mcp_policy(destination_id: str = "brave", *, pii_enabled: bool = True) -> dict[str, object]:
    """Build trusted metadata for one exact PII-enabled MCP route."""
    return {
        "contract_version": 1,
        "destination_kind": "mcp",
        "principal_id": "principal-1",
        "destination_id": destination_id,
        "pii_enabled": pii_enabled,
    }


@pytest.fixture
def engine_reply() -> dict[str, object]:
    """Return a valid sanitized engine response."""
    return {
        "api_version": "v1",
        "decision": "apply_actions",
        "entities": ["PERSON"],
        "entity_counts": {"PERSON": 1},
        "applied_actions": ["reversible_replace"],
        "remote_allowed": True,
        "route_class": None,
        "request": {
            "model": "test",
            "messages": [{"role": "user", "content": REVERSIBLE_TOKEN}],
        },
        "analysis": {
            "source": "current_request",
            "scan_performed": True,
            "duration_ms": 3200,
            "overlap_count": 0,
            "overlap_resolution": "strictest_action",
            "policy_version": "test",
            "text_leaf_count": 1,
            "cached_decision_applied": False,
        },
        "notices": {"request": ["Request protected"], "response": ["Protected"]},
        "report": {
            "rows": [
                {
                    "entity_type": "PERSON",
                    "action": "reversible_replace",
                    "detected_count": 1,
                    "transformed_count": 1,
                    "unique_transformed_count": 1,
                }
            ],
        },
        "reversal": {REVERSIBLE_TOKEN: "Jane Doe"},
    }


@pytest.fixture
def engine_client(engine_reply: dict[str, object]) -> EngineClient:
    """Build a mocked HTTP engine client."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=segment_reply(engine_reply), request=request)

    return EngineClient(
        EngineSettings(base_url="https://pii-engine.test"),
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://pii-engine.test"
        ),
    )


def header_request(
    headers: dict[str, str] | None = None,
    end_of_stream: bool = False,
    policy: dict[str, object] | None = MODEL_POLICY,
) -> ext_proc_pb2.ProcessingRequest:
    """Build an Envoy request headers message."""
    supplied = dict(headers or {})
    if policy is not None:
        context = policy.get("request_context", policy)
        if isinstance(context, dict) and context.get("destination_kind") == "model":
            binding = context.get("binding_id", "test-binding")
            operation = (
                "responses"
                if context.get("protocol_profile") == "openai-responses-v1"
                else "chat/completions"
            )
            supplied = {
                ":method": "POST",
                ":path": f"/bindings/{binding}/v1/{operation}",
                "content-type": "application/json",
            } | supplied
            if policy.get("contract_version") != 5 and supplied[":path"].startswith("/v1/"):
                supplied[":path"] = "/bindings/test-binding" + supplied[":path"]
    values = [
        shared_envoy_pb2.HeaderValue(key=key, raw_value=value.encode())
        for key, value in supplied.items()
    ]
    return add_policy(
        ext_proc_pb2.ProcessingRequest(
            request_headers=ext_proc_pb2.HttpHeaders(
                headers=ext_proc_pb2.HeaderMap(headers=values), end_of_stream=end_of_stream
            )
        ),
        policy,
    )


def body_request(
    body: bytes,
    end_of_stream: bool = True,
    policy: dict[str, object] | None = MODEL_POLICY,
) -> ext_proc_pb2.ProcessingRequest:
    """Build an Envoy request body message."""
    return add_policy(
        ext_proc_pb2.ProcessingRequest(
            request_body=ext_proc_pb2.HttpBody(body=body, end_of_stream=end_of_stream)
        ),
        policy,
    )


def response_headers(
    content_type: str,
    encoding: str | None = None,
    status: int = 200,
    extra_headers: dict[str, str] | None = None,
    policy: dict[str, object] | None = MODEL_POLICY,
    end_of_stream: bool = False,
) -> ext_proc_pb2.ProcessingRequest:
    """Build an Envoy response headers message."""
    values = [
        shared_envoy_pb2.HeaderValue(key=":status", value=str(status)),
        shared_envoy_pb2.HeaderValue(key="content-type", raw_value=content_type.encode()),
    ]
    if encoding:
        values.append(
            shared_envoy_pb2.HeaderValue(key="content-encoding", raw_value=encoding.encode())
        )
    values.extend(
        shared_envoy_pb2.HeaderValue(key=key, raw_value=value.encode())
        for key, value in (extra_headers or {}).items()
    )
    return add_policy(
        ext_proc_pb2.ProcessingRequest(
            response_headers=ext_proc_pb2.HttpHeaders(
                headers=ext_proc_pb2.HeaderMap(headers=values),
                end_of_stream=end_of_stream,
            )
        ),
        policy,
    )


def response_body(
    body: bytes,
    end_of_stream: bool = True,
    policy: dict[str, object] | None = MODEL_POLICY,
) -> ext_proc_pb2.ProcessingRequest:
    """Build an Envoy response body message."""
    return add_policy(
        ext_proc_pb2.ProcessingRequest(
            response_body=ext_proc_pb2.HttpBody(body=body, end_of_stream=end_of_stream)
        ),
        policy,
    )


def add_policy(
    request: ext_proc_pb2.ProcessingRequest,
    policy: dict[str, object] | None = MODEL_POLICY,
) -> ext_proc_pb2.ProcessingRequest:
    """Attach the current admission envelope around the selected policy fixture."""
    if policy is not None:
        envelope = policy
        if policy.get("contract_version") != 5 and policy.get("destination_kind") in {
            "model",
            "mcp",
        }:
            mcp = policy["destination_kind"] == "mcp"
            models = policy.get("models", {})
            models = models if isinstance(models, dict) else {}
            permissions = ["llm:invoke"]
            if mcp:
                permissions.append(f"mcp:{policy.get('destination_id')}:invoke")
            else:
                permissions.extend(f"model:{name}:invoke" for name in models)
                permissions.extend(["provider:test:company-use", "model:local:invoke"])
            context = {
                "request_id": "0123456789abcdef0123456789abcdef",
                "principal_id": policy.get("principal_id"),
                "credential_context_version": policy.get("credential_context_version"),
                "credential_id": policy.get("credential_id"),
                "credential_kind": policy.get("credential_kind"),
                "entry_mode": "api",
                "destination_kind": policy["destination_kind"],
                "provider_id": None if mcp else "test",
                "binding_id": f"mcp-{policy.get('destination_id')}" if mcp else "test-binding",
                "protocol_profile": "mcp-streamable-http-v1"
                if mcp
                else "openai-chat-completions-v1",
                "operation_id": "exchange" if mcp else "create",
                "provider_credential_source": "none" if mcp else "company",
                "policy_revision": "test-policy",
                "permissions": list(dict.fromkeys(permissions)),
            }
            envelope = {
                "contract_version": 5,
                "request_context": context,
                "destination_policy": policy,
                "model_bindings": {}
                if mcp
                else {
                    name: {
                        "upstream_model": name,
                        "required_permissions": ["llm:invoke", f"model:{name}:invoke"],
                        "content_tracing_enabled": True,
                        "allowed_reroutes": ["local"],
                    }
                    for name in models
                },
                "reroute_targets": {}
                if mcp
                else {
                    "local": {
                        "binding_id": "local-binding",
                        "upstream_model": "local",
                        "required_permissions": ["llm:invoke", "model:local:invoke"],
                        "content_tracing_enabled": True,
                    }
                },
            }
        request.metadata_context.filter_metadata[DESTINATION_POLICY_NAMESPACE].update(envelope)
    return request


def mcp_headers(
    *,
    method: str = "POST",
    destination_id: str = "brave",
    session_id: str | None = None,
) -> dict[str, str]:
    """Build strict Streamable HTTP request headers for MCP tests."""
    headers = {
        ":method": method,
        ":path": f"/bindings/mcp-{destination_id}/mcp",
        "mcp-protocol-version": "2025-11-25",
    }
    if method == "POST":
        headers.update(
            {
                "content-type": "application/json; charset=utf-8",
                "accept": "application/json, text/event-stream",
            }
        )
    elif method == "GET":
        headers["accept"] = "text/event-stream"
    if session_id is not None:
        headers["mcp-session-id"] = session_id
    return headers


def request_json() -> bytes:
    """Build a valid OpenAI-compatible request."""
    return json.dumps(
        {"model": "test", "messages": [{"role": "user", "content": "Jane Doe"}]}
    ).encode()
