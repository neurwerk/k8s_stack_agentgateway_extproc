"""Native extraction, routing and streaming boundaries with synthetic credentials."""

import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from google.protobuf.json_format import MessageToDict
from neurwerk_request_segments import extract_request, parse_request

from agentgateway_extproc.config.settings import EngineSettings, Settings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.pipeline.sse import SseDecoder
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.exceptions import GatewayPolicyError, TrustedMetadataError

from .conftest import (
    REVERSIBLE_TOKEN,
    body_request,
    header_request,
    response_body,
    response_headers,
)
from .test_fail_closed import FakeAioContext, StreamAbortedError

FIXTURES = Path(__file__).parent / "fixtures"
CREDENTIAL = "provider-credential-sentinel"


@pytest.fixture(params=["anthropic", "google"])
def native(request):
    kind = request.param
    admission = json.loads((FIXTURES / "admission-v5.json").read_text())
    context = admission["request_context"]
    context.update(
        protocol_profile="anthropic-messages-v1"
        if kind == "anthropic"
        else "google-generate-content-v1",
        operation_id="create" if kind == "anthropic" else "streamGenerateContent",
        provider_id=kind,
        binding_id=f"{kind}-api",
    )
    headers = {
        ":method": "POST",
        ":path": "/v1/messages"
        if kind == "anthropic"
        else "/v1beta/models/provider-chat:streamGenerateContent?alt=sse",
        "content-type": "application/json",
        "x-api-key" if kind == "anthropic" else "x-goog-api-key": CREDENTIAL,
    }
    if kind == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    return kind, admission, headers, json.loads((FIXTURES / f"{kind}-request.json").read_text())


@pytest.fixture
async def native_engine(engine_reply):
    calls = []

    def analyze(request):
        wire = json.loads(request.content)
        calls.append(wire)
        assert wire["api_version"] == "v2" and wire["request_kind"] == "chat"
        assert "model" not in wire and "messages" not in wire and "contents" not in wire
        assert CREDENTIAL not in str(request.headers) + request.content.decode()
        segments = wire["segments"]
        private_value = engine_reply["reversal"][REVERSIBLE_TOKEN]
        count = sum(segment["text"].count(private_value) for segment in segments)
        reply = deepcopy(engine_reply)
        reply.pop("request")
        reply.update(
            api_version="v2",
            route_class="general",
            segments=[
                {**segment, "text": segment["text"].replace(private_value, REVERSIBLE_TOKEN)}
                for segment in segments
            ],
        )
        reply["entity_counts"] = {"PERSON": count}
        reply["analysis"]["text_leaf_count"] = len(segments)
        reply["report"]["rows"][0].update(
            detected_count=count, transformed_count=count, unique_transformed_count=1
        )
        return httpx.Response(200, json=reply)

    async with httpx.AsyncClient(transport=httpx.MockTransport(analyze)) as http:
        yield EngineClient(EngineSettings(base_url="https://engine.example.test"), http), calls


def encoded(value):
    return json.dumps(value, ensure_ascii=False).encode()


def json_response(kind, text=REVERSIBLE_TOKEN):
    if kind == "anthropic":
        return {
            "id": "message-1",
            "type": "message",
            "role": "assistant",
            "model": "provider-chat",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "tool_use",
                    "id": "call-2",
                    "name": "lookup",
                    "input": {"person": text, "count": 1},
                },
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
    return {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": text},
                        {
                            "functionCall": {
                                "id": "call-2",
                                "name": "lookup",
                                "args": {"person": text, "count": 1},
                            }
                        },
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 10,
            "totalTokenCount": 30,
        },
        "modelVersion": "provider-chat",
    }


def stream_events(kind, text=REVERSIBLE_TOKEN):
    if kind == "google":
        return [
            {
                "candidates": [
                    {"index": 0, "content": {"role": "model", "parts": [{"text": fragment}]}}
                ]
            }
            for fragment in [text[:12], text[12:]]
        ] + [
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {
                            "role": "model",
                            "parts": [
                                {"functionCall": {"name": "lookup", "args": {"person": text}}}
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 20,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 30,
                },
            }
        ]
    message = json_response(kind)
    message.update(content=[], stop_reason=None)
    tool_json = json.dumps({"person": text, "quoted": 'a "quote"\nand newline'})
    return [
        {"type": "message_start", "message": message},
        {"type": "ping"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        *[
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": fragment},
            }
            for fragment in [text[:12], text[12:]]
        ],
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "call-2", "name": "lookup", "input": {}},
        },
        *[
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": fragment},
            }
            for fragment in [tool_json[:30], tool_json[30:]]
        ],
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 10},
        },
        {"type": "message_stop"},
    ]


def sse_bytes(kind, events):
    return "".join(
        (f"event: {event['type']}\n" if kind == "anthropic" else "")
        + f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        for event in events
    ).encode()


def test_native_segments_roundtrip_and_controls(native):
    kind, _, _, payload = native
    original = parse_request(payload, kind=kind)
    extracted = extract_request(original)
    assert extracted.protocol_kind == kind and extracted.request_kind == "chat"
    assert extracted.rebuild(extracted.segments).model_dump(exclude_unset=True) == payload
    assert len(extracted.segments) == 6
    transformed = extracted.rebuild(
        [
            segment.model_copy(update={"text": segment.text.replace("Jane Doe", REVERSIBLE_TOKEN)})
            for segment in extracted.segments
        ]
    )
    assert extract_request(transformed).control_shape() == extracted.control_shape()
    assert "Jane Doe" not in transformed.model_dump_json()
    assert "lookup" in transformed.model_dump_json() and "call-1" in transformed.model_dump_json()
    with pytest.raises(ValueError):
        extracted.rebuild(extracted.segments[:-1])


@pytest.mark.parametrize("pii", [False, True])
@pytest.mark.parametrize("alias", [False, True])
async def test_native_inspection_routing_and_metadata(native, native_engine, pii, alias):
    kind, admission, headers, payload = native
    client, calls = native_engine
    admission["destination_policy"]["models"]["example-chat"] = pii
    model = "provider-chat" if alias else "example-chat"
    if kind == "anthropic":
        payload["model"] = model
    else:
        headers[":path"] = headers[":path"].replace("provider-chat", model)
    handler = StreamHandler(client)
    head = await handler.handle(header_request(headers, policy=admission))
    assert head is not None
    credential_name = "x-api-key" if kind == "anthropic" else "x-goog-api-key"
    assert credential_name not in head.request_headers.response.header_mutation.remove_headers
    reply = await handler.handle(body_request(encoded(payload), policy=admission))
    assert reply is not None and reply.HasField("request_body")
    forwarded = json.loads(reply.request_body.response.body_mutation.body)
    assert forwarded["model"] == "example-chat"
    assert ("Jane Doe" in str(forwarded)) is not pii
    assert len(calls) == int(pii)
    assert ("messages" in forwarded) is (kind == "anthropic")
    assert ("contents" in forwarded) is (kind == "google")
    metadata = MessageToDict(reply.dynamic_metadata)["neurwerk.request_context"]
    assert (
        metadata["destination_id"] == "example-chat"
        and metadata["resolved_model"] == "provider-chat"
    )
    assert metadata["requested_model"] == model
    assert metadata["usage_completeness"] == metadata["billing_classification"] == "unknown"
    assert CREDENTIAL not in str(reply) + repr(vars(handler))
    assert not any(
        h.header.key == ":path" for h in reply.request_body.response.header_mutation.set_headers
    )


@pytest.mark.parametrize("pii", [False, True])
async def test_native_json_restores_only_text_and_tool_values(native, native_engine, pii):
    kind, admission, headers, payload = native
    client, _ = native_engine
    if kind == "anthropic":
        payload["stream"] = False
    else:
        admission["request_context"]["operation_id"] = "generateContent"
        headers[":path"] = "/v1beta/models/provider-chat:generateContent"
    admission["destination_policy"]["models"]["example-chat"] = pii
    handler = StreamHandler(client)
    await handler.handle(header_request(headers, policy=admission))
    await handler.handle(body_request(encoded(payload), policy=admission))
    await handler.handle(response_headers("application/json", policy=admission))
    reply = await handler.handle(response_body(encoded(json_response(kind)), policy=admission))
    assert reply is not None
    restored = json.loads(reply.response_body.response.body_mutation.streamed_response.body)
    assert restored == json_response(kind, "Jane Doe" if pii else REVERSIBLE_TOKEN)
    assert not reply.dynamic_metadata.fields
    assert handler.reversal_map == {}


@pytest.mark.parametrize("chunk_size", [1, 4096])
async def test_native_sse_split_tokens_tools_usage_and_empty_eos(native, native_engine, chunk_size):
    kind, admission, headers, payload = native
    client, _ = native_engine
    handler = StreamHandler(client)
    await handler.handle(header_request(headers, policy=admission))
    await handler.handle(body_request(encoded(payload), policy=admission))
    await handler.handle(response_headers("text/event-stream", policy=admission))
    incoming = sse_bytes(kind, stream_events(kind))
    output = b""
    for position in range(0, len(incoming), chunk_size):
        reply = await handler.handle(
            response_body(
                incoming[position : position + chunk_size], end_of_stream=False, policy=admission
            )
        )
        if reply is not None:
            output += reply.response_body.response.body_mutation.streamed_response.body
    reply = await handler.handle(response_body(b"", policy=None))
    assert reply is not None
    output += reply.response_body.response.body_mutation.streamed_response.body
    events = [json.loads(event.data) for event in SseDecoder().feed(output.decode(), final=True)]
    assert "Jane Doe" in output.decode() and REVERSIBLE_TOKEN not in output.decode()
    if kind == "anthropic":
        tool = next(
            event for event in events if event.get("delta", {}).get("type") == "input_json_delta"
        )
        assert json.loads(tool["delta"]["partial_json"]) == {
            "person": "Jane Doe",
            "quoted": 'a "quote"\nand newline',
        }
        assert events[-1]["type"] == "message_stop"
        assert events[-2]["usage"]["output_tokens"] == 10
    else:
        assert events[-1]["usageMetadata"]["totalTokenCount"] == 30
        assert events[-1]["candidates"][0]["content"]["parts"][0]["functionCall"]["args"] == {
            "person": "Jane Doe"
        }
    assert handler.reversal_map == {}


@pytest.mark.parametrize(
    "failure", ["permission", "unknown_model", "credential", "feature", "content", "depth", "bytes"]
)
async def test_native_request_failures_precede_analysis(native, native_engine, failure):  # noqa: C901
    kind, admission, headers, payload = native
    client, calls = native_engine
    settings = Settings(max_request_bytes=1024) if failure == "bytes" else Settings()
    if failure == "permission":
        admission["request_context"]["permissions"] = []
    elif failure == "unknown_model":
        if kind == "anthropic":
            payload["model"] = "outside-binding"
        else:
            headers[":path"] = headers[":path"].replace("provider-chat", "outside-binding")
    elif failure == "credential":
        headers["authorization"] = f"Bearer {CREDENTIAL}"
    elif failure == "feature":
        payload["thinking" if kind == "anthropic" else "cachedContent"] = CREDENTIAL
    elif failure == "content":
        if kind == "anthropic":
            payload["messages"][0]["content"] = [{"type": "image", "source": {"data": CREDENTIAL}}]
        else:
            payload["contents"][0]["parts"] = [{"inlineData": {"data": CREDENTIAL}}]
    elif failure == "depth":
        nested = "Jane Doe"
        for _ in range(40):
            nested = {"nested": nested}
        if kind == "anthropic":
            payload["messages"][1]["content"][0]["input"] = nested
        else:
            payload["contents"][1]["parts"][0]["functionCall"]["args"] = nested
    else:
        if kind == "anthropic":
            payload["messages"][0]["content"] = "a" * 2000
        else:
            payload["contents"][0]["parts"][0]["text"] = "a" * 2000

    async def requests():
        yield header_request(headers, policy=admission)
        yield body_request(encoded(payload), policy=admission)

    replies = [
        reply async for reply in ExtProcServicer(client, settings).Process(requests(), object())
    ]
    assert replies[-1].HasField("immediate_response")
    assert replies[-1].immediate_response.status.code in {400, 403, 413}
    assert calls == []
    assert CREDENTIAL not in str(replies)


@pytest.mark.parametrize(
    "failure", ["error", "unknown_event", "truncated", "protocol_token", "unknown_token", "limit"]
)
async def test_native_stream_errors_abort_without_leaking_credentials(
    native, native_engine, caplog, failure
):
    kind, admission, headers, payload = native
    client, _ = native_engine
    events = stream_events(kind)
    settings = Settings(max_response_bytes=1024) if failure == "limit" else Settings()
    if failure == "error":
        events = [{"type": "error", "error": {"message": CREDENTIAL}}]
    elif failure == "unknown_event":
        events = [{"type": "unqualified_content", "data": CREDENTIAL}]
    elif failure == "truncated":
        events = events[:-1]
    elif failure == "protocol_token":
        if kind == "anthropic":
            events[0]["message"]["id"] = REVERSIBLE_TOKEN
        else:
            events[0]["responseId"] = REVERSIBLE_TOKEN
    elif failure == "unknown_token":
        events = stream_events(kind, "<REV_PERSON_aaaaaaaaaaaaaaaa_bbbbbbbbbbbbbbbb>")
    else:
        events = stream_events(kind, "a" * 2000)
    context = FakeAioContext()
    replies = []

    async def requests():
        yield header_request(headers, policy=admission)
        yield body_request(encoded(payload), policy=admission)
        yield response_headers("text/event-stream", policy=admission)
        yield response_body(sse_bytes(kind, events), policy=admission)

    with pytest.raises(StreamAbortedError):
        async for reply in ExtProcServicer(client, settings).Process(requests(), context):
            replies.append(reply)
    assert CREDENTIAL not in str(replies) + caplog.text + context.details


async def test_metadata_is_required_except_wholly_absent_empty_terminal_callback(
    native, native_engine
):
    _, admission, headers, payload = native
    client, _ = native_engine
    handler = StreamHandler(client)
    with pytest.raises(TrustedMetadataError):
        await handler.handle(header_request(headers, policy=None))
    await handler.handle(header_request(headers, policy=admission))
    with pytest.raises(TrustedMetadataError):
        await handler.handle(body_request(encoded(payload), policy=None))
    await handler.handle(body_request(encoded(payload), policy=admission))
    await handler.handle(response_headers("text/event-stream", policy=admission))
    partial = response_body(b"", policy=None)
    partial.metadata_context.filter_metadata["neurwerk.request_context"].update(
        {"request_id": "other"}
    )
    with pytest.raises(TrustedMetadataError):
        await handler.handle(partial)
    with pytest.raises(TrustedMetadataError):
        await handler.handle(response_body(b"data: {}\n\n", policy=None))
    changed = deepcopy(admission)
    changed["request_context"]["binding_id"] = "different-binding"
    with pytest.raises(TrustedMetadataError):
        await handler.handle(response_body(b"", policy=changed))


async def test_native_tool_restoration_preserves_json_escaping(native, native_engine, engine_reply):
    kind, admission, headers, payload = native
    client, _ = native_engine
    private_value = 'Jane "Doe"\n'
    engine_reply["reversal"][REVERSIBLE_TOKEN] = private_value
    payload = json.loads(json.dumps(payload).replace("Jane Doe", json.dumps(private_value)[1:-1]))
    handler = StreamHandler(client)
    await handler.handle(header_request(headers, policy=admission))
    await handler.handle(body_request(encoded(payload), policy=admission))
    await handler.handle(response_headers("text/event-stream", policy=admission))
    reply = await handler.handle(
        response_body(sse_bytes(kind, stream_events(kind)), policy=admission)
    )
    assert reply is not None
    events = [
        json.loads(event.data)
        for event in SseDecoder().feed(
            reply.response_body.response.body_mutation.streamed_response.body.decode(), final=True
        )
    ]
    if kind == "anthropic":
        call = next(
            event for event in events if event.get("delta", {}).get("type") == "input_json_delta"
        )
        assert json.loads(call["delta"]["partial_json"])["person"] == private_value
    else:
        assert (
            events[-1]["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["person"]
            == private_value
        )


@pytest.mark.parametrize(
    "failure", ["transport", "hosted_tool", "signature", "wrong_response_format"]
)
async def test_unqualified_native_features_and_transport_fail_closed(
    native, native_engine, failure
):
    kind, admission, headers, payload = native
    client, calls = native_engine
    if failure == "transport":
        if kind == "anthropic":
            headers["anthropic-beta"] = "unqualified-feature"
        else:
            headers[":path"] += "&key=" + CREDENTIAL
    elif failure == "hosted_tool":
        payload["tools"] = (
            [{"type": "web_search_20250305", "name": "web_search"}]
            if kind == "anthropic"
            else [{"googleSearch": {}}]
        )
    elif failure == "signature":
        if kind == "anthropic":
            payload["messages"][1]["content"] = [
                {"type": "thinking", "thinking": "Jane Doe", "signature": CREDENTIAL}
            ]
        else:
            payload["contents"][1]["parts"][0]["thoughtSignature"] = CREDENTIAL

    async def requests():
        yield header_request(headers, policy=admission)
        yield body_request(encoded(payload), policy=admission)
        yield response_headers("application/json", policy=admission)

    replies = [reply async for reply in ExtProcServicer(client).Process(requests(), object())]
    assert replies[-1].HasField("immediate_response")
    assert len(calls) == int(failure == "wrong_response_format")
    assert CREDENTIAL not in str(replies)


def test_native_extraction_does_not_accept_control_mutations(native):
    kind, _, _, payload = native
    original = extract_request(parse_request(payload, kind=kind))
    mutated = deepcopy(payload)
    if kind == "anthropic":
        mutated["messages"][1]["content"][0]["name"] = "different-tool"
    else:
        mutated["contents"][1]["parts"][0]["functionCall"]["name"] = "different-tool"
    assert (
        extract_request(parse_request(mutated, kind=kind)).control_shape()
        != original.control_shape()
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_native_refusal_without_text_is_preserved(native, native_engine, stream):
    kind, admission, headers, payload = native
    client, _ = native_engine
    if kind == "anthropic":
        payload["stream"] = stream
        data = json_response(kind)
        data.update(content=[], stop_reason="refusal")
        start = {**data, "stop_reason": None}
        events = [
            {"type": "message_start", "message": start},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "refusal"},
                "usage": {"output_tokens": 0},
            },
            {"type": "message_stop"},
        ]
    else:
        if not stream:
            admission["request_context"]["operation_id"] = "generateContent"
            headers[":path"] = "/v1beta/models/provider-chat:generateContent"
        data = {"candidates": [{"index": 0, "content": None, "finishReason": "SAFETY"}]}
        events = [data]
    handler = StreamHandler(client)
    await handler.handle(header_request(headers, policy=admission))
    await handler.handle(body_request(encoded(payload), policy=admission))
    await handler.handle(
        response_headers("text/event-stream" if stream else "application/json", policy=admission)
    )
    reply = await handler.handle(
        response_body(sse_bytes(kind, events) if stream else encoded(data), policy=admission)
    )
    assert reply is not None and reply.HasField("response_body")
    body = reply.response_body.response.body_mutation.streamed_response.body
    if stream:
        assert [
            json.loads(event.data) for event in SseDecoder().feed(body.decode(), final=True)
        ] == events
    else:
        assert json.loads(body) == data


@pytest.mark.parametrize(
    "mode,prefix_binding,accepted",
    [
        ("api", "verified", True),
        ("api", "other", False),
        ("forward_proxy", "verified", False),
    ],
)
async def test_native_api_mount_keeps_upstream_path_rewrite_with_gateway(
    native, native_engine, mode, prefix_binding, accepted
):
    _, admission, headers, payload = native
    client, calls = native_engine
    context = admission["request_context"]
    context["entry_mode"] = mode
    binding = context["binding_id"] if prefix_binding == "verified" else "another-binding"
    headers[":path"] = f"/bindings/{binding}" + headers[":path"]
    handler = StreamHandler(client)
    if not accepted:
        with pytest.raises(GatewayPolicyError):
            await handler.handle(header_request(headers, policy=admission))
        assert calls == []
        assert handler.request_headers == {}
        return
    reply = await handler.handle(header_request(headers, policy=admission))
    assert reply is not None and not reply.request_headers.response.header_mutation.set_headers
    assert handler.request_headers[":path"] == headers[":path"]
    reply = await handler.handle(body_request(encoded(payload), policy=admission))
    assert reply is not None and reply.HasField("request_body")
    assert json.loads(reply.request_body.response.body_mutation.body)["model"] == "example-chat"
    assert not any(
        h.header.key == ":path" for h in reply.request_body.response.header_mutation.set_headers
    )
    assert len(calls) == 1
