"""Test trusted destination dispatch and the narrowed MCP transport contract."""

from __future__ import annotations

import gzip
import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.lib.pipeline.mcp import (
    MCP_PROTOCOL_VERSION,
    McpHeaderContext,
    McpProtocolError,
    parse_mcp_message,
)
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.lib.session import make_session_key
from agentgateway_extproc.metrics import dispatcher_total
from agentgateway_extproc.models.destination import (
    McpDestinationPolicy,
    ModelDestinationPolicy,
)
from agentgateway_extproc.models.engine import EngineMcpRequest
from agentgateway_extproc.models.exceptions import (
    InvalidEngineReplyError,
    InvalidReversalError,
    TrustedMetadataError,
)

from .conftest import (
    MODEL_POLICY,
    REVERSIBLE_TOKEN,
    add_policy,
    body_request,
    header_request,
    mcp_headers,
    mcp_policy,
    request_json,
    response_body,
    response_headers,
)


async def test_missing_or_changing_metadata_is_a_platform_503(engine_client) -> None:
    servicer = ExtProcServicer(engine_client)

    async def missing():
        yield header_request(policy=None)

    responses = [response async for response in servicer.Process(missing(), object())]
    assert responses[-1].immediate_response.status.code == 503
    assert responses[-1].immediate_response.body == '{"error":"internal processing error"}'

    changed = {**MODEL_POLICY, "principal_id": "principal-2"}

    async def inconsistent():
        yield header_request()
        yield body_request(request_json(), policy=changed)

    responses = [response async for response in servicer.Process(inconsistent(), object())]
    assert responses[-1].immediate_response.status.code == 503


async def test_metadata_less_empty_response_eos_uses_locked_destination(engine_client) -> None:
    """AgentGateway may omit metadata only on its empty final response callback."""
    before = dispatcher_total.labels(outcome="metadata_eos_compat")._value.get()
    servicer = ExtProcServicer(engine_client)
    payload = b'{"choices":[{"index":0,"message":{"content":"answer"}}]}'

    async def requests():
        yield header_request()
        yield body_request(request_json())
        yield response_headers("application/json")
        yield response_body(payload, end_of_stream=False)
        yield response_body(b"", policy=None)

    responses = [response async for response in servicer.Process(requests(), object())]

    assert [response.WhichOneof("response") for response in responses] == [
        "request_headers",
        "request_body",
        "response_headers",
        "response_body",
    ]
    transformed = json.loads(
        responses[-1].response_body.response.body_mutation.streamed_response.body
    )
    assert transformed["choices"][0]["message"]["content"].startswith("answer")
    after = dispatcher_total.labels(outcome="metadata_eos_compat")._value.get()
    assert after - before == 1


@pytest.mark.parametrize(
    "processing_request",
    [
        response_body(b"not-empty", policy=None),
        response_body(b"", end_of_stream=False, policy=None),
        response_body(b"", policy={}),
        response_body(b"", policy={**MODEL_POLICY, "principal_id": "principal-2"}),
    ],
    ids=["non-empty", "not-final", "malformed", "changed"],
)
async def test_metadata_eos_compat_rejects_nearby_invalid_callbacks(
    engine_client, processing_request
) -> None:
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))
    await handler.handle(response_headers("application/json"))

    with pytest.raises(TrustedMetadataError):
        await handler.handle(processing_request)


async def test_metadata_eos_compat_requires_accepted_response_headers(engine_client) -> None:
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))

    with pytest.raises(TrustedMetadataError):
        await handler.handle(response_body(b"", policy=None))


async def test_metadata_eos_compat_rejects_callback_after_completed_response(
    engine_client,
) -> None:
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))
    await handler.handle(response_headers("application/json"))
    completed = await handler.handle(
        response_body(b'{"choices":[{"message":{"content":"answer"}}]}')
    )
    assert completed is not None

    with pytest.raises(TrustedMetadataError):
        await handler.handle(response_body(b"", policy=None))


@pytest.mark.parametrize(
    "policy",
    [
        {**MODEL_POLICY, "contract_version": 2},
        {**MODEL_POLICY, "principal_id": ""},
        {**MODEL_POLICY, "models": {"test": "false"}},
        {**mcp_policy(), "pii_enabled": "false"},
        {**mcp_policy(), "destination_id": "Bad_ID"},
        {**MODEL_POLICY, "content_tracing_enabled": True},
    ],
)
async def test_malformed_metadata_fails_closed(engine_client, policy) -> None:
    servicer = ExtProcServicer(engine_client)

    async def requests():
        yield header_request(policy=policy)

    responses = [response async for response in servicer.Process(requests(), object())]
    assert responses[-1].immediate_response.status.code == 503


async def test_exact_disabled_model_bypasses_engine_with_noop_response_callbacks(
    engine_client,
) -> None:
    policy = {**MODEL_POLICY, "models": {"test": False, "other": True}}
    original = b'{ "model" : "test", "messages" : [ {"role":"user","content":"raw"} ] }'
    handler = StreamHandler(engine_client)
    with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
        await handler.handle(header_request(policy=policy))
        response = await handler.handle(body_request(original, policy=policy))

    assert response is not None
    assert not response.request_body.response.body_mutation.ByteSize()
    assert not response.HasField("mode_override")
    assert not handler.reversal_map
    assert handler.request_stats is None
    assert handler.presidio_code is None
    assert not handler.notice_messages
    analyze.assert_not_awaited()

    headers = await handler.handle(response_headers("application/json", policy=policy))
    body = await handler.handle(response_body(b'{"value":"unchanged"}', policy=policy))
    trailers = await handler.handle(
        add_policy(
            ext_proc_pb2.ProcessingRequest(
                response_trailers=ext_proc_pb2.HttpTrailers(
                    trailers={"headers": [{"key": "digest", "value": "sha-256=stale"}]}
                )
            ),
            policy,
        )
    )
    assert headers is not None and headers.HasField("response_headers")
    assert body is not None and not body.response_body.response.body_mutation.ByteSize()
    assert trailers is not None
    assert not trailers.response_trailers.header_mutation.remove_headers


@pytest.mark.parametrize(
    "body",
    [
        b'{"model":"unknown","messages":[{"role":"user","content":"x"}]}',
        b'{"model":"test","model":"other","messages":[{"role":"user","content":"x"}]}',
        b'{"model":"test","messages":[{"role":"user","content":"x"}],"stream":1}',
        b'{"model":"test","messages":[{"role":"user","content":"x"}],"temperature":NaN}',
    ],
)
async def test_invalid_model_selection_is_a_400_before_engine(engine_client, body) -> None:
    policy = {**MODEL_POLICY, "models": {"test": False, "other": True}}
    handler = StreamHandler(engine_client)
    with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
        await handler.handle(header_request(policy=policy))
        response = await handler.handle(body_request(body, policy=policy))
    assert response is not None
    assert response.immediate_response.status.code == 400
    analyze.assert_not_awaited()


async def test_all_disabled_catalog_sets_safe_header_phase_mode_override(engine_client) -> None:
    policy = {**MODEL_POLICY, "models": {"test": False}}
    response = await StreamHandler(engine_client).handle(header_request(policy=policy))
    assert response is not None
    assert response.mode_override.response_header_mode == response.mode_override.SKIP
    assert response.mode_override.response_body_mode == response.mode_override.NONE


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        (mcp_headers() | {":path": "/mcp/brave/extra"}, b"{}"),
        (mcp_headers() | {"mcp-protocol-version": "2025-03-26"}, b"{}"),
        (mcp_headers() | {"content-encoding": "gzip"}, gzip.compress(b"{}")),
        (mcp_headers(), b"[]"),
        (
            mcp_headers(),
            b'{"jsonrpc":"2.0","id":1,"method":"tasks/get","params":{}}',
        ),
        (
            mcp_headers(),
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
            b'"params":{"name":"search","arguments":[]}}',
        ),
        (
            mcp_headers(),
            b'{"jsonrpc":"2.0","id":1,"method":"resources/list","params":{"cursor":1e9999}}',
        ),
        (mcp_headers() | {"last-event-id": "cursor-1"}, b"{}"),
        (
            mcp_headers() | {"accept": "application/json;q=0, text/event-stream;q=1"},
            b"{}",
        ),
        (
            mcp_headers() | {"accept": "application/json;q=1, text/event-stream;q=0"},
            b"{}",
        ),
        (
            mcp_headers() | {"accept": "*/*;q=1, application/json;q=0, text/event-stream;q=1"},
            b"{}",
        ),
        (
            mcp_headers(method="GET") | {"accept": "text/event-stream;q=0"},
            b"{}",
        ),
        (
            mcp_headers() | {"accept": "application/json;q=1.1, text/event-stream"},
            b"{}",
        ),
    ],
)
async def test_mcp_protocol_violations_are_400_before_engine(engine_client, headers, body) -> None:
    policy = mcp_policy()
    handler = StreamHandler(engine_client)
    with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
        first = await handler.handle(header_request(headers, policy=policy))
        response = (
            first
            if first is not None and first.HasField("immediate_response")
            else await handler.handle(body_request(body, policy=policy))
        )
    assert response is not None
    assert response.immediate_response.status.code == 400
    analyze.assert_not_awaited()


async def test_mcp_accepts_positive_quality_values(engine_client) -> None:
    policy = mcp_policy()
    headers = mcp_headers() | {
        "accept": "application/json;profile=test;q=0.001, text/event-stream;q=1.000"
    }
    handler = StreamHandler(engine_client)
    response = await handler.handle(header_request(headers, policy=policy))
    assert response is not None and not response.HasField("immediate_response")


@pytest.mark.parametrize(
    "payload",
    [
        {
            "jsonrpc": "2.0",
            "id": -9_007_199_254_740_991,
            "method": "tools/call",
            "params": {"name": "a" * 128},
        },
        {
            "jsonrpc": "2.0",
            "id": 9_007_199_254_740_991,
            "method": "tools/call",
            "params": {
                "name": "admin.tools-list_2",
                "arguments": {
                    **{str(index): index for index in range(255)},
                    "nested": [None] * 256,
                },
                "_meta": {
                    **{str(index): index for index in range(63)},
                    "k" * 256: {str(index): index for index in range(256)},
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": "x" * 256,
            "method": "tools/call",
            "params": {"name": "lookup", "arguments": {"number": 1.5, "flag": False}},
        },
    ],
    ids=["minimum-integer-and-omitted-optionals", "collection-limits", "maximum-string-id"],
)
def test_mcp_parser_accepts_exact_engine_schema_boundaries(payload) -> None:
    expected = EngineMcpRequest.model_validate(payload)
    context = parse_mcp_message(
        json.dumps(payload).encode(),
        McpHeaderContext("POST", MCP_PROTOCOL_VERSION, None),
    )
    actual = context.engine_request
    assert actual is not None
    assert actual == expected
    assert actual.model_dump(mode="json", by_alias=True, exclude_none=True) == payload


@pytest.mark.parametrize(
    "payload",
    [
        {"jsonrpc": "2.0", "id": True, "method": "tools/call", "params": {"name": "x"}},
        {"jsonrpc": "2.0", "id": 1.0, "method": "tools/call", "params": {"name": "x"}},
        {"jsonrpc": "2.0", "id": "", "method": "tools/call", "params": {"name": "x"}},
        {
            "jsonrpc": "2.0",
            "id": "x" * 257,
            "method": "tools/call",
            "params": {"name": "x"},
        },
        {
            "jsonrpc": "2.0",
            "id": 9_007_199_254_740_992,
            "method": "tools/call",
            "params": {"name": "x"},
        },
        {
            "jsonrpc": "2.0",
            "id": -9_007_199_254_740_992,
            "method": "tools/call",
            "params": {"name": "x"},
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "x" * 129},
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "bad/name"},
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "bad:name"},
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lookup", "arguments": None},
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lookup", "arguments": {str(index): index for index in range(257)}},
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lookup", "arguments": {"nested": [None] * 257}},
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "lookup",
                "arguments": {"nested": {str(index): index for index in range(257)}},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lookup", "_meta": None},
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lookup", "_meta": {str(index): index for index in range(65)}},
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lookup", "_meta": {"k" * 257: "value"}},
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lookup", "arguments": {"number": float("nan")}},
        },
    ],
    ids=[
        "boolean-id",
        "float-id",
        "empty-id",
        "long-id",
        "integer-id-above",
        "integer-id-below",
        "long-name",
        "slash-name",
        "colon-name",
        "null-arguments",
        "arguments-entries",
        "nested-list-entries",
        "nested-dict-entries",
        "null-meta",
        "meta-entries",
        "meta-key-length",
        "non-finite-number",
    ],
)
async def test_mcp_parser_rejects_first_engine_schema_violations_before_engine(
    engine_client, payload
) -> None:
    with pytest.raises(ValidationError):
        EngineMcpRequest.model_validate(payload)
    with pytest.raises(McpProtocolError):
        parse_mcp_message(
            json.dumps(payload).encode(),
            McpHeaderContext("POST", MCP_PROTOCOL_VERSION, None),
        )

    policy = mcp_policy()
    handler = StreamHandler(engine_client)
    with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
        await handler.handle(header_request(mcp_headers(), policy=policy))
        response = await handler.handle(body_request(json.dumps(payload).encode(), policy=policy))
    assert response is not None
    assert response.immediate_response.status.code == 400
    analyze.assert_not_awaited()


@pytest.mark.parametrize("pii_enabled", [True, False])
async def test_mcp_lifecycle_and_no_text_calls_preserve_request_bytes(
    engine_client, pii_enabled: bool
) -> None:
    policy = mcp_policy(pii_enabled=pii_enabled)
    requests = [
        b'{ "jsonrpc":"2.0", "id":1, "method":"resources/list", "params":{} }',
        b'{ "jsonrpc":"2.0", "id":2, "method":"tools/call", '
        b'"params":{"name":"lookup","arguments":{"limit":2,"active":true}} }',
    ]
    with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
        for body in requests:
            handler = StreamHandler(engine_client)
            await handler.handle(header_request(mcp_headers(), policy=policy))
            response = await handler.handle(body_request(body, policy=policy))
            assert response is not None
            assert not response.request_body.response.body_mutation.ByteSize()
    analyze.assert_not_awaited()


async def test_mcp_pii_disabled_text_call_is_protocol_only(engine_client) -> None:
    policy = mcp_policy(pii_enabled=False)
    original = (
        b'{ "jsonrpc":"2.0", "id":"call-1", "method":"tools/call", '
        b'"params":{"name":"search","arguments":{"query":"' + REVERSIBLE_TOKEN.encode() + b'"}} }'
    )
    handler = StreamHandler(engine_client)
    with (
        patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze,
        patch("agentgateway_extproc.lib.pipeline.request.make_session_key") as session_key,
    ):
        await handler.handle(header_request(mcp_headers(), policy=policy))
        request = await handler.handle(body_request(original, policy=policy))

    assert request is not None
    assert not request.request_body.response.body_mutation.ByteSize()
    assert not request.request_body.response.header_mutation.set_headers
    assert handler.mcp_context is not None
    assert handler.mcp_context.engine_request is None
    assert handler.request_nonce is None
    assert not handler.reversal_map
    assert handler.request_stats is None
    assert handler.presidio_code is None
    assert not handler.notice_messages
    analyze.assert_not_awaited()
    session_key.assert_not_called()

    upstream = (
        b'{ "jsonrpc":"2.0", "id":"call-1", "result":{"value":"'
        + REVERSIBLE_TOKEN.encode()
        + b'"} }'
    )
    assert await handler.handle(response_headers("application/json", policy=policy)) is None
    response = await handler.handle(response_body(upstream, policy=policy))
    assert response is not None
    assert response.response_body.response.body_mutation.streamed_response.body == upstream
    headers = handler.pop_pending_response_headers()
    assert headers is not None
    assert not headers.response_headers.response.header_mutation.set_headers
    assert list(headers.response_headers.response.header_mutation.remove_headers) == [
        "x-presidio-code"
    ]


@pytest.mark.parametrize(
    ("body", "request_id"),
    [
        (
            b'{ "jsonrpc" : "2.0", "id" : "sampling-1", "result" : '
            b'{"model":"test","content":{"type":"text","text":"'
            + REVERSIBLE_TOKEN.encode()
            + b'"}} }',
            "sampling-1",
        ),
        (
            b'{"jsonrpc":"2.0","id":7,"error":'
            b'{"code":-32603,"message":"sampling failed","data":{"retryable":false}}}',
            7,
        ),
    ],
    ids=["sampling-result", "error"],
)
async def test_mcp_client_response_is_preserved_without_engine_and_acknowledged(
    engine_client, body: bytes, request_id: str | int
) -> None:
    policy = mcp_policy(pii_enabled=True)
    handler = StreamHandler(engine_client)
    with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
        await handler.handle(header_request(mcp_headers(), policy=policy))
        request = await handler.handle(body_request(body, policy=policy))

    assert request is not None
    assert not request.request_body.response.body_mutation.ByteSize()
    assert not request.request_body.response.header_mutation.set_headers
    assert handler.mcp_context is not None
    assert handler.mcp_context.method is None
    assert handler.mcp_context.request_id == request_id
    assert handler.mcp_context.engine_request is None
    assert not handler.request_body_chunks
    assert not handler.request_headers
    assert not handler.reversal_map
    analyze.assert_not_awaited()

    acknowledgement = await handler.handle(
        response_headers("application/json", status=202, policy=policy, end_of_stream=True)
    )
    assert acknowledgement is not None and acknowledgement.HasField("response_headers")


@pytest.mark.parametrize(
    "body",
    [
        b'[{"jsonrpc":"2.0","id":1,"result":{}}]',
        b'{"jsonrpc":"2.0","result":{}}',
        b'{"jsonrpc":"2.0","id":1}',
        b'{"jsonrpc":"2.0","id":1,"result":{},"error":{"code":-1,"message":"x"}}',
        b'{"jsonrpc":"2.0","id":1,"result":{},"extra":true}',
        b'{"jsonrpc":"2.0","id":null,"result":{}}',
        b'{"jsonrpc":"2.0","id":1,"error":{"code":-1}}',
        b'{"jsonrpc":"2.0","id":1,"method":"sampling/createMessage","result":{}}',
    ],
    ids=[
        "batch",
        "missing-id",
        "missing-result-or-error",
        "result-and-error",
        "extra-field",
        "null-id",
        "malformed-error",
        "method-present",
    ],
)
async def test_mcp_malformed_client_response_is_rejected_before_engine(
    engine_client, body: bytes
) -> None:
    policy = mcp_policy()
    handler = StreamHandler(engine_client)
    with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
        await handler.handle(header_request(mcp_headers(), policy=policy))
        response = await handler.handle(body_request(body, policy=policy))
    assert response is not None
    assert response.immediate_response.status.code == 400
    analyze.assert_not_awaited()


async def test_mcp_pii_disabled_gzip_response_preserves_encoded_bytes(engine_client) -> None:
    policy = mcp_policy(pii_enabled=False)
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": REVERSIBLE_TOKEN}},
    }
    upstream = json.dumps(
        {"jsonrpc": "2.0", "id": 3, "result": {"value": REVERSIBLE_TOKEN}},
        separators=(",", ":"),
    ).encode()
    encoded = gzip.compress(upstream)
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(mcp_headers(), policy=policy))
    await handler.handle(body_request(json.dumps(request).encode(), policy=policy))
    assert await handler.handle(response_headers("application/json", "gzip", policy=policy)) is None
    first = await handler.handle(
        response_body(encoded[: len(encoded) // 2], end_of_stream=False, policy=policy)
    )
    assert first is None
    response = await handler.handle(response_body(encoded[len(encoded) // 2 :], policy=policy))
    assert response is not None
    assert response.response_body.response.body_mutation.streamed_response.body == encoded
    headers = handler.pop_pending_response_headers()
    assert headers is not None
    removed = headers.response_headers.response.header_mutation.remove_headers
    assert "content-encoding" not in removed
    assert "content-length" not in removed
    trailers = await handler.handle(
        add_policy(
            ext_proc_pb2.ProcessingRequest(
                response_trailers=ext_proc_pb2.HttpTrailers(
                    trailers={"headers": [{"key": "digest", "value": "sha-256=current"}]}
                )
            ),
            policy,
        )
    )
    assert trailers is not None
    assert not trailers.response_trailers.header_mutation.remove_headers


@pytest.mark.parametrize(
    "mutate",
    [
        lambda encoded: encoded[:-1],
        lambda encoded: encoded + b"trailing",
        lambda encoded: encoded + gzip.compress(b"{}"),
    ],
    ids=["truncated", "trailing-data", "concatenated-member"],
)
async def test_mcp_pii_disabled_gzip_is_not_forwarded_before_single_member_validation(
    engine_client, mutate
) -> None:
    policy = mcp_policy(pii_enabled=False)
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": "safe"}},
    }
    upstream = json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}}).encode()
    encoded = mutate(gzip.compress(upstream))
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(mcp_headers(), policy=policy))
    await handler.handle(body_request(json.dumps(request).encode(), policy=policy))
    await handler.handle(response_headers("application/json", "gzip", policy=policy))
    with pytest.raises(ValueError, match="gzip"):
        await handler.handle(response_body(encoded, policy=policy))
    assert handler.response_emitted_bytes == 0


async def test_mcp_text_call_sends_narrowed_engine_request_once(
    engine_client, engine_reply
) -> None:
    policy = mcp_policy()
    original = {
        "jsonrpc": "2.0",
        "id": "call-1",
        "method": "tools/call",
        "params": {
            "name": "search",
            "arguments": {"query": "Jane Doe", "nested": [1, {"city": "Jane Doe"}]},
            "_meta": {"progressToken": "safe"},
        },
    }
    engine_reply["entity_counts"] = {"PERSON": 2}
    engine_reply["report"]["rows"][0].update(
        {"detected_count": 2, "transformed_count": 2, "unique_transformed_count": 1}
    )
    engine_reply["request"] = {
        **original,
        "params": {
            **original["params"],
            "arguments": {"query": REVERSIBLE_TOKEN, "nested": [1, {"city": REVERSIBLE_TOKEN}]},
        },
    }
    engine_reply["notices"] = {"request": [], "response": []}
    handler = StreamHandler(engine_client)
    with patch.object(
        engine_client,
        "analyze_request",
        wraps=engine_client.analyze_request,
    ) as analyze:
        await handler.handle(header_request(mcp_headers(session_id="session-1"), policy=policy))
        response = await handler.handle(body_request(json.dumps(original).encode(), policy=policy))
    assert response is not None
    analyze.assert_awaited_once()
    call = analyze.await_args
    assert call is not None
    sent = call.args[0]
    assert isinstance(sent, EngineMcpRequest)
    assert sent.model_dump(by_alias=True) == original
    forwarded = json.loads(response.request_body.response.body_mutation.body)
    assert forwarded["params"]["name"] == "search"
    assert forwarded["params"]["_meta"] == {"progressToken": "safe"}
    assert [
        item.header.key for item in response.request_body.response.header_mutation.set_headers
    ] == ["content-length"]


async def test_mcp_block_is_exact_json_rpc_200(engine_client, engine_reply) -> None:
    policy = mcp_policy()
    original = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": "Jane Doe"}},
    }
    engine_reply.update(
        {
            "decision": "block",
            "applied_actions": ["block"],
            "remote_allowed": False,
            "route_class": None,
            "request": None,
            "reversal": {},
        }
    )
    engine_reply["report"]["rows"][0].update(
        {"action": "block", "transformed_count": 0, "unique_transformed_count": 0}
    )
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(mcp_headers(), policy=policy))
    response = await handler.handle(body_request(json.dumps(original).encode(), policy=policy))
    assert response is not None
    assert response.immediate_response.status.code == 200
    assert response.immediate_response.body == (
        '{"jsonrpc":"2.0","id":7,"error":{"code":-32000,'
        '"message":"Request blocked by data policy"}}'
    )
    assert response.immediate_response.headers.set_headers[0].header.value == "application/json"


async def test_mcp_engine_reroute_reply_is_invalid(engine_client, engine_reply) -> None:
    policy = mcp_policy()
    original = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": "Jane Doe"}},
    }
    engine_reply.update(
        {
            "decision": "reroute",
            "applied_actions": ["reroute"],
            "remote_allowed": False,
            "route_class": "local",
            "request": original,
            "reversal": {},
        }
    )
    engine_reply["report"]["rows"][0].update(
        {"action": "reroute", "transformed_count": 0, "unique_transformed_count": 0}
    )
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(mcp_headers(), policy=policy))
    with pytest.raises(InvalidEngineReplyError):
        await handler.handle(body_request(json.dumps(original).encode(), policy=policy))


async def test_mcp_response_reverses_only_authorized_result_locations(
    engine_client, engine_reply
) -> None:
    handler, policy = await _analyzed_mcp_handler(engine_client, engine_reply)
    await handler.handle(response_headers("application/json", policy=policy))
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {"type": "text", "text": REVERSIBLE_TOKEN},
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///safe",
                        "mimeType": "text/plain",
                        "text": REVERSIBLE_TOKEN,
                    },
                },
            ],
            "structuredContent": {"nested": [REVERSIBLE_TOKEN]},
        },
    }
    response = await handler.handle(response_body(json.dumps(payload).encode(), policy=policy))
    assert response is not None
    transformed = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    assert transformed["result"]["content"][0]["text"] == "Jane Doe"
    assert transformed["result"]["content"][1]["resource"]["text"] == "Jane Doe"
    assert transformed["result"]["structuredContent"]["nested"] == ["Jane Doe"]
    assert transformed["result"]["content"][1]["resource"]["uri"] == "file:///safe"


@pytest.mark.parametrize(
    "payload",
    [
        {"jsonrpc": "2.0", "id": 2, "result": {}},
        {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": REVERSIBLE_TOKEN}},
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "resource", "resource": {"uri": REVERSIBLE_TOKEN, "text": "safe"}}
                ]
            },
        },
        {"jsonrpc": "2.0", "id": 1, "result": {"name": REVERSIBLE_TOKEN}},
    ],
)
async def test_mcp_response_mismatch_or_unauthorized_placeholder_fails_closed(
    engine_client, engine_reply, payload
) -> None:
    handler, policy = await _analyzed_mcp_handler(engine_client, engine_reply)
    await handler.handle(response_headers("application/json", policy=policy))
    with pytest.raises((McpProtocolError, InvalidReversalError)):
        await handler.handle(response_body(json.dumps(payload).encode(), policy=policy))


async def test_mcp_request_scoped_sse_reversal_and_resumption_rejection(
    engine_client, engine_reply
) -> None:
    handler, policy = await _analyzed_mcp_handler(engine_client, engine_reply)
    await handler.handle(response_headers("text/event-stream", policy=policy))
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": REVERSIBLE_TOKEN}]},
    }
    first = await handler.handle(
        response_body(
            f"data: {json.dumps(payload)}\n".encode(),
            end_of_stream=False,
            policy=policy,
        )
    )
    final = await handler.handle(response_body(b"\n", policy=policy))
    assert first is not None and final is not None
    assert b"Jane Doe" in final.response_body.response.body_mutation.streamed_response.body

    resumed, policy = await _analyzed_mcp_handler(engine_client, engine_reply)
    await resumed.handle(response_headers("text/event-stream", policy=policy))
    with pytest.raises(McpProtocolError, match="resumable"):
        await resumed.handle(
            response_body(
                f"id: cursor-1\ndata: {json.dumps(payload)}\n\n".encode(),
                policy=policy,
            )
        )


async def test_mcp_sse_allows_server_messages_before_one_final_response(
    engine_client, engine_reply
) -> None:
    handler, policy = await _analyzed_mcp_handler(engine_client, engine_reply)
    await handler.handle(response_headers("text/event-stream", policy=policy))
    server_request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "server-1",
            "method": "sampling/createMessage",
            "params": {"messages": [{"role": "user", "content": "safe"}]},
        },
        separators=(",", ":"),
    )
    notification = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {"level": "info", "data": "safe"},
        },
        separators=(",", ":"),
    )
    final = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": REVERSIBLE_TOKEN}]},
        },
        separators=(",", ":"),
    )
    stream = (
        f"event:message\r\ndata:{server_request}\r\n\r\n"
        f"data:{notification}\r\n\r\n"
        f"event: message\r\ndata:{final}\r\n\r\n"
    ).encode()
    response = await handler.handle(response_body(stream, policy=policy))
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body
    unchanged_prefix = (
        f"event:message\r\ndata:{server_request}\r\n\r\ndata:{notification}\r\n\r\n"
    ).encode()
    assert output.startswith(unchanged_prefix)
    assert b"event: message\r\ndata:" in output
    assert b"Jane Doe" in output
    assert output.endswith(b"\r\n\r\n")


@pytest.mark.parametrize(
    "events",
    [
        [
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 1, "result": {}},
        ],
        [
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "method": "notifications/message", "params": {}},
        ],
        [
            {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {"data": "safe"},
            }
        ],
    ],
    ids=["extra-final", "message-after-final", "missing-final"],
)
async def test_mcp_sse_requires_exactly_one_terminal_response(
    engine_client, engine_reply, events
) -> None:
    handler, policy = await _analyzed_mcp_handler(engine_client, engine_reply)
    await handler.handle(response_headers("text/event-stream", policy=policy))
    stream = "".join(f"data:{json.dumps(event)}\n\n" for event in events).encode()
    with pytest.raises(McpProtocolError):
        await handler.handle(response_body(stream, policy=policy))


async def test_mcp_sse_server_message_cannot_use_authorized_request_placeholder(
    engine_client, engine_reply
) -> None:
    handler, policy = await _analyzed_mcp_handler(engine_client, engine_reply)
    await handler.handle(response_headers("text/event-stream", policy=policy))
    message = {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {"data": REVERSIBLE_TOKEN},
    }
    with pytest.raises(InvalidReversalError, match="unauthorized placeholder"):
        await handler.handle(
            response_body(f"data:{json.dumps(message)}\n\n".encode(), policy=policy)
        )


@pytest.mark.parametrize(
    "metadata_line",
    [
        f": {REVERSIBLE_TOKEN}",
        f"x-extension: {REVERSIBLE_TOKEN}",
        f"{REVERSIBLE_TOKEN}: safe",
    ],
    ids=["comment", "extension-value", "extension-name"],
)
async def test_mcp_sse_metadata_cannot_carry_placeholders(
    engine_client, engine_reply, metadata_line: str
) -> None:
    handler, policy = await _analyzed_mcp_handler(engine_client, engine_reply)
    await handler.handle(response_headers("text/event-stream", policy=policy))
    with pytest.raises(InvalidReversalError, match="unauthorized placeholder"):
        await handler.handle(response_body(f"{metadata_line}\n\n".encode(), policy=policy))


async def test_mcp_pii_disabled_sse_is_byte_identical(engine_client) -> None:
    policy = mcp_policy(pii_enabled=False)
    request = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": "Jane Doe"}},
    }
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(mcp_headers(), policy=policy))
    await handler.handle(body_request(json.dumps(request).encode(), policy=policy))
    await handler.handle(response_headers("text/event-stream", policy=policy))
    server_message = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {"data": REVERSIBLE_TOKEN},
        },
        separators=(",", ":"),
    )
    final = json.dumps(
        {"jsonrpc": "2.0", "id": 9, "result": {"value": REVERSIBLE_TOKEN}},
        separators=(",", ":"),
    )
    stream = (
        f"\r\n\nevent:message\r\ndata:{server_message}\r\n\r\n\r\ndata:{final}\r\n\r\n\n"
    ).encode()
    response = await handler.handle(response_body(stream, policy=policy))
    assert response is not None
    assert response.response_body.response.body_mutation.streamed_response.body == stream


async def test_mcp_notification_empty_json_202_is_acknowledged(engine_client) -> None:
    policy = mcp_policy()
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }
    servicer = ExtProcServicer(engine_client)

    async def requests():
        yield header_request(mcp_headers(), policy=policy)
        yield body_request(json.dumps(notification).encode(), policy=policy)
        yield response_headers("application/json", status=202, policy=policy, end_of_stream=True)

    responses = [response async for response in servicer.Process(requests(), object())]
    assert [response.WhichOneof("response") for response in responses][-1] == "response_headers"


@pytest.mark.parametrize("status", [200, 202])
async def test_mcp_delete_rejects_non_204_lifecycle_status(engine_client, status: int) -> None:
    policy = mcp_policy()
    handler = StreamHandler(engine_client)
    await handler.handle(
        header_request(mcp_headers(method="DELETE"), end_of_stream=True, policy=policy)
    )
    with pytest.raises(McpProtocolError, match="empty HTTP 204"):
        await handler.handle(
            response_headers("application/json", status=status, policy=policy, end_of_stream=True)
        )


async def test_mcp_delete_empty_204_is_acknowledged(engine_client) -> None:
    policy = mcp_policy()
    servicer = ExtProcServicer(engine_client)

    async def requests():
        yield header_request(mcp_headers(method="DELETE"), end_of_stream=True, policy=policy)
        yield response_headers("application/json", status=204, policy=policy, end_of_stream=True)

    responses = [response async for response in servicer.Process(requests(), object())]
    assert [response.WhichOneof("response") for response in responses] == [
        "request_headers",
        "response_headers",
    ]


@pytest.mark.parametrize(
    ("extra_headers", "end_of_stream"),
    [
        ({"content-length": "1"}, True),
        ({"transfer-encoding": "chunked"}, True),
        ({}, False),
    ],
    ids=["content-length", "chunked", "body-callback-promised"],
)
async def test_mcp_delete_204_rejects_promised_body(
    engine_client, extra_headers: dict[str, str], end_of_stream: bool
) -> None:
    policy = mcp_policy()
    handler = StreamHandler(engine_client)
    await handler.handle(
        header_request(mcp_headers(method="DELETE"), end_of_stream=True, policy=policy)
    )
    with pytest.raises(McpProtocolError, match="cannot contain a body"):
        await handler.handle(
            response_headers(
                "application/json",
                status=204,
                extra_headers=extra_headers,
                policy=policy,
                end_of_stream=end_of_stream,
            )
        )


async def test_mcp_delete_204_rejects_body_after_header_only_ack(engine_client) -> None:
    policy = mcp_policy()
    handler = StreamHandler(engine_client)
    await handler.handle(
        header_request(mcp_headers(method="DELETE"), end_of_stream=True, policy=policy)
    )
    acknowledgement = await handler.handle(
        response_headers("application/json", status=204, policy=policy, end_of_stream=True)
    )
    assert acknowledgement is not None and acknowledgement.HasField("response_headers")
    with pytest.raises(McpProtocolError, match="cannot contain a body"):
        await handler.handle(response_body(b"unexpected", policy=policy))


@pytest.mark.parametrize(
    "payload",
    [
        {
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "id": 1, "method": "resources/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "safe"}},
        },
    ],
    ids=["initialize", "lifecycle-request-id", "tools-call"],
)
@pytest.mark.parametrize("end_of_stream", [False, True], ids=["with-body", "empty"])
async def test_mcp_request_id_methods_reject_202(
    engine_client, payload, end_of_stream: bool
) -> None:
    policy = mcp_policy(pii_enabled=False)
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(mcp_headers(), policy=policy))
    await handler.handle(body_request(json.dumps(payload).encode(), policy=policy))
    with pytest.raises(McpProtocolError, match="HTTP 200"):
        await handler.handle(
            response_headers(
                "application/json",
                status=202,
                policy=policy,
                end_of_stream=end_of_stream,
            )
        )


async def test_mcp_notification_202_rejects_promised_body(engine_client) -> None:
    policy = mcp_policy()
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(mcp_headers(), policy=policy))
    await handler.handle(body_request(json.dumps(notification).encode(), policy=policy))
    with pytest.raises(McpProtocolError, match="cannot contain a body"):
        await handler.handle(response_headers("application/json", status=202, policy=policy))


@pytest.mark.parametrize(
    "payload",
    [
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        {"jsonrpc": "2.0", "id": "sampling-1", "result": {"content": "safe"}},
    ],
    ids=["notification", "client-response"],
)
@pytest.mark.parametrize("status", [200, 204])
async def test_mcp_notification_and_client_response_require_http_202(
    engine_client, payload, status: int
) -> None:
    policy = mcp_policy()
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(mcp_headers(), policy=policy))
    await handler.handle(body_request(json.dumps(payload).encode(), policy=policy))
    with pytest.raises(McpProtocolError, match="empty HTTP 202"):
        await handler.handle(
            response_headers(
                "application/json",
                status=status,
                policy=policy,
                end_of_stream=True,
            )
        )


async def test_mcp_client_response_202_rejects_promised_body(engine_client) -> None:
    policy = mcp_policy()
    response = {"jsonrpc": "2.0", "id": "sampling-1", "result": {"content": "safe"}}
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(mcp_headers(), policy=policy))
    await handler.handle(body_request(json.dumps(response).encode(), policy=policy))
    with pytest.raises(McpProtocolError, match="cannot contain a body"):
        await handler.handle(response_headers("application/json", status=202, policy=policy))


@pytest.mark.parametrize("pii_enabled", [True, False])
async def test_mcp_initialization_requires_exact_backend_protocol(
    engine_client, pii_enabled: bool
) -> None:
    policy = mcp_policy(pii_enabled=pii_enabled)
    initialize = {
        "jsonrpc": "2.0",
        "id": "init-1",
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(mcp_headers(), policy=policy))
    request = await handler.handle(body_request(json.dumps(initialize).encode(), policy=policy))
    assert request is not None and not request.request_body.response.body_mutation.ByteSize()
    await handler.handle(response_headers("application/json", policy=policy))
    response = {
        "jsonrpc": "2.0",
        "id": "init-1",
        "result": {"protocolVersion": "2025-03-26", "capabilities": {}},
    }
    with pytest.raises(McpProtocolError, match="unsupported MCP version"):
        await handler.handle(response_body(json.dumps(response).encode(), policy=policy))


async def test_mcp_gzip_json_response_is_bounded_and_reversed(engine_client, engine_reply) -> None:
    handler, policy = await _analyzed_mcp_handler(engine_client, engine_reply)
    headers = await handler.handle(response_headers("application/json", "gzip", policy=policy))
    assert headers is None
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": REVERSIBLE_TOKEN}]},
    }
    response = await handler.handle(
        response_body(gzip.compress(json.dumps(payload).encode()), policy=policy)
    )
    assert response is not None
    transformed = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    assert transformed["result"]["content"][0]["text"] == "Jane Doe"


async def test_header_only_mcp_get_and_response_events_pass_without_engine(engine_client) -> None:
    policy = mcp_policy()
    handler = StreamHandler(engine_client)
    with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
        request = await handler.handle(
            header_request(mcp_headers(method="GET"), end_of_stream=True, policy=policy)
        )
        assert request is not None and request.HasField("request_headers")
        await handler.handle(response_headers("text/event-stream", policy=policy))
        body = (
            b'data: {"jsonrpc":"2.0","method":"notifications/message","params":{}}\n\n'
            b'data: {"jsonrpc":"2.0","method":"notifications/message","params":{}}\n\n'
        )
        response = await handler.handle(response_body(body, policy=policy))
    assert response is not None
    analyze.assert_not_awaited()


async def test_mcp_buffered_body_and_trailers_receive_one_matching_response_each(
    engine_client,
) -> None:
    policy = mcp_policy()
    body = b'{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}'
    servicer = ExtProcServicer(engine_client)

    async def requests():
        yield header_request(mcp_headers(), policy=policy)
        yield body_request(body, end_of_stream=False, policy=policy)
        yield add_policy(
            ext_proc_pb2.ProcessingRequest(
                request_trailers=ext_proc_pb2.HttpTrailers(trailers=ext_proc_pb2.HeaderMap())
            ),
            policy,
        )

    responses = [response async for response in servicer.Process(requests(), object())]
    kinds = [response.WhichOneof("response") for response in responses]
    assert kinds == ["request_headers", "request_body", "request_trailers"]
    assert not responses[1].request_body.response.body_mutation.ByteSize()


def test_session_keys_scope_principal_destination_and_mcp_session() -> None:
    request = EngineMcpRequest(
        jsonrpc="2.0",
        id=1,
        method="tools/call",
        params={"name": "search", "arguments": {"query": "safe"}},
    )
    brave = McpDestinationPolicy(
        contract_version=1,
        destination_kind="mcp",
        principal_id="principal-1",
        destination_id="brave",
        pii_enabled=True,
    )
    other_destination = brave.model_copy(update={"destination_id": "other"})
    other_principal = brave.model_copy(update={"principal_id": "principal-2"})
    key = make_session_key(brave, request, mcp_session_id="session-1")
    assert key == make_session_key(brave, request, mcp_session_id="session-1")
    assert key != make_session_key(brave, request, mcp_session_id="session-2")
    assert key != make_session_key(other_destination, request, mcp_session_id="session-1")
    assert key != make_session_key(other_principal, request, mcp_session_id="session-1")
    assert make_session_key(brave, request, request_nonce=b"a" * 32) != make_session_key(
        brave,
        request,
        request_nonce=b"b" * 32,
    )

    model = ModelDestinationPolicy(
        contract_version=1,
        destination_kind="model",
        principal_id="principal-1",
        models={"test": True},
    )
    with pytest.raises(ValueError, match="MCP destination"):
        make_session_key(model, request, mcp_session_id="session-1")


async def _analyzed_mcp_handler(engine_client, engine_reply):
    policy = mcp_policy()
    original = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": "Jane Doe"}},
    }
    engine_reply["request"] = {
        **original,
        "params": {"name": "search", "arguments": {"query": REVERSIBLE_TOKEN}},
    }
    engine_reply["notices"] = {"request": [], "response": []}
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(mcp_headers(), policy=policy))
    await handler.handle(body_request(json.dumps(original).encode(), policy=policy))
    return handler, policy
