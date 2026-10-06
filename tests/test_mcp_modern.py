"""Protect the modern MCP inspection boundary without changing legacy behavior."""

import base64
import json

import pytest

from agentgateway_extproc.lib.pipeline.mcp import (
    MCP_MODERN_VERSION,
    McpProtocolError,
    parse_mcp_message,
    validate_mcp_headers,
)
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.destination import McpDestinationPolicy

from .conftest import (
    REVERSIBLE_TOKEN,
    body_request,
    header_request,
    mcp_headers,
    mcp_policy,
    response_body,
    response_headers,
)


def modern_request(method="tools/call", **params):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": MCP_MODERN_VERSION,
                "io.modelcontextprotocol/clientCapabilities": {},
            },
            **params,
        },
    }


def modern_headers(method="tools/call", name="search"):
    return mcp_headers() | {
        "mcp-protocol-version": MCP_MODERN_VERSION,
        "mcp-method": method,
        "mcp-name": name,
    }


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["params"]["_meta"].pop("io.modelcontextprotocol/clientCapabilities"),
        lambda p: p["params"]["_meta"].update(
            {"io.modelcontextprotocol/protocolVersion": "2025-11-25"}
        ),
        lambda p: p.update(method="tools/list"),
        lambda p: p["params"].update(name="different"),
        lambda p: p["params"].update(inputResponses={}),
        lambda p: p["params"].update(requestState="opaque"),
    ],
)
def test_modern_requests_reject_mismatches_and_uninspected_inputs(change):
    payload = modern_request(name="search", arguments={"query": "safe"})
    change(payload)
    headers = validate_mcp_headers(
        modern_headers(), McpDestinationPolicy.model_validate(mcp_policy())
    )
    with pytest.raises(McpProtocolError):
        parse_mcp_message(json.dumps(payload).encode(), headers)


def test_modern_resource_name_header_decodes_utf8():
    uri = "file:///example/世界"
    encoded = base64.b64encode(uri.encode()).decode()
    headers = validate_mcp_headers(
        modern_headers("resources/read", f"=?base64?{encoded}?="),
        McpDestinationPolicy.model_validate(mcp_policy()),
    )
    context = parse_mcp_message(
        json.dumps(modern_request("resources/read", uri=uri)).encode(), headers
    )
    assert context.protocol_version == MCP_MODERN_VERSION


@pytest.mark.parametrize("pii_enabled", [True, False])
@pytest.mark.parametrize("version", ["2025-11-25", MCP_MODERN_VERSION])
async def test_mirrored_parameter_headers_cannot_bypass_pii(engine_client, pii_enabled, version):
    policy = mcp_policy(pii_enabled=pii_enabled)
    handler = StreamHandler(engine_client)
    response = await handler.handle(
        header_request(
            modern_headers() | {"mcp-protocol-version": version, "mcp-param-query": "Jane Doe"},
            policy=policy,
        )
    )
    assert response is not None
    assert response.WhichOneof("response") == (
        "immediate_response" if pii_enabled else "request_headers"
    )


@pytest.mark.parametrize("sse", [False, True])
async def test_modern_tool_call_masks_arguments_and_reverses_result(
    engine_client, engine_reply, sse
):
    payload = modern_request(name="search", arguments={"query": "Jane Doe"})
    engine_reply["request"] = {
        **payload,
        "params": {**payload["params"], "arguments": {"query": REVERSIBLE_TOKEN}},
    }
    engine_reply["notices"] = {"request": [], "response": []}
    policy = mcp_policy()
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(modern_headers(), policy=policy))
    response = await handler.handle(body_request(json.dumps(payload).encode(), policy=policy))
    assert response is not None
    forwarded = json.loads(response.request_body.response.body_mutation.body)
    assert forwarded["params"]["arguments"] == {"query": REVERSIBLE_TOKEN}
    assert forwarded["params"]["_meta"] == payload["params"]["_meta"]
    result = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "resultType": "complete",
            "content": [{"type": "text", "text": REVERSIBLE_TOKEN}],
            "structuredContent": [REVERSIBLE_TOKEN],
        },
    }
    media = "text/event-stream" if sse else "application/json"
    await handler.handle(response_headers(media, policy=policy))
    body = json.dumps(result).encode()
    if sse:
        body = b"event: message\ndata: " + body + b"\n\n"
    response = await handler.handle(response_body(body, policy=policy))
    assert response is not None
    body = response.response_body.response.body_mutation.streamed_response.body
    if sse:
        body = next(line[6:] for line in body.splitlines() if line.startswith(b"data: "))
    restored = json.loads(body)
    assert restored["result"]["content"][0]["text"] == "Jane Doe"
    assert restored["result"]["structuredContent"] == ["Jane Doe"]


@pytest.mark.parametrize("pii_enabled", [True, False])
async def test_modern_negotiation_error_keeps_safe_versions_not_diagnostics(
    engine_client, pii_enabled
):
    policy = mcp_policy(pii_enabled=pii_enabled)
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(modern_headers("server/discover"), policy=policy))
    await handler.handle(
        body_request(json.dumps(modern_request("server/discover")).encode(), policy=policy)
    )
    await handler.handle(
        response_headers(
            "application/json",
            status=400,
            policy=policy,
            extra_headers={"www-authenticate": "private challenge", "x-debug": "private"},
        )
    )
    error = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32022,
            "message": "private upstream diagnostic",
            "data": {"supported": ["2025-11-25", "private-version"], "secret": "private"},
        },
    }
    response = await handler.handle(response_body(json.dumps(error).encode(), policy=policy))
    assert response is not None
    safe = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    assert safe["error"] == {
        "code": -32022,
        "message": "Unsupported MCP protocol version",
        "data": {"supported": ["2025-11-25"]},
    }
    headers = handler.pop_pending_response_headers()
    assert headers is not None
    removed = headers.response_headers.response.header_mutation.remove_headers
    assert "www-authenticate" in removed
    assert "x-debug" in removed


@pytest.mark.parametrize("result", [{}, {"resultType": "input_required", "requestState": "x"}])
async def test_modern_results_reject_missing_type_and_multi_round_trip(engine_client, result):
    policy = mcp_policy(pii_enabled=False)
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(modern_headers("server/discover"), policy=policy))
    await handler.handle(
        body_request(json.dumps(modern_request("server/discover")).encode(), policy=policy)
    )
    await handler.handle(response_headers("application/json", policy=policy))
    with pytest.raises(McpProtocolError):
        await handler.handle(
            response_body(
                json.dumps({"jsonrpc": "2.0", "id": 1, "result": result}).encode(), policy=policy
            )
        )
