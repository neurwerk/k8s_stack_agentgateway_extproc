"""V5 closed response schemas and protocol completion at the real gRPC boundary."""

import json
import logging
from copy import deepcopy
from unittest.mock import patch

import grpc
import httpx
import pytest

from agentgateway_extproc.config.settings import EngineSettings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.pipeline.diagnostics import diagnostic_json
from agentgateway_extproc.lib.pipeline.sse import SseDecoder
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler

from .conftest import (
    REVERSIBLE_TOKEN,
    add_policy,
    body_request,
    header_request,
    response_body,
    response_headers,
)
from .openai_fixtures import event_bytes, response_payload, stream_events
from .test_admission import admission as admission
from .test_admission import body, headers
from .test_fail_closed import FakeAioContext, StreamAbortedError
from .test_reasoning import reasoning_engine as reasoning_engine

SENTINEL = "synthetic-sentinel"


def request_messages(admission, kind, *, stream):
    admission["request_context"]["protocol_profile"] = (
        "openai-responses-v1" if kind == "responses" else "openai-chat-completions-v1"
    )
    path = "/v1/responses" if kind == "responses" else "/v1/chat/completions"
    path = f"/bindings/{admission['request_context']['binding_id']}" + path
    return [
        header_request(headers(**{":path": path}), policy=admission),
        body_request(body(responses=kind == "responses", stream=stream), policy=admission),
    ]


async def collect(engine_client, messages, replies, context):
    async def requests():
        for message in messages:
            yield message

    async for reply in ExtProcServicer(engine_client).Process(requests(), context):
        replies.append(reply)


async def test_buffered_chat_preserves_reasoning_values(admission, engine_client):
    reasoning = {
        "reasoning": "Think about " + REVERSIBLE_TOKEN,
        "reasoning_content": "Alias " + REVERSIBLE_TOKEN,
        "reasoning_details": [{"type": "reasoning.text", "text": "Thought", "index": 0}],
    }
    payload = response_payload("chat")
    payload["choices"][0]["message"].update(reasoning)
    replies = []
    await collect(
        engine_client,
        [
            *request_messages(admission, "chat", stream=False),
            response_headers("application/json", policy=admission),
            response_body(json.dumps(payload).encode(), policy=admission),
        ],
        replies,
        FakeAioContext(),
    )
    assert replies[-1].HasField("response_body")
    value = json.loads(replies[-1].response_body.response.body_mutation.streamed_response.body)
    message = value["choices"][0]["message"]
    assert {key: message[key] for key in reasoning} == reasoning
    assert "Jane Doe" in message["content"]


async def test_streaming_chat_preserves_reasoning_deltas(admission, engine_client):
    events = stream_events("chat", tools=True)
    reasoning = {
        "reasoning": "Thought " + REVERSIBLE_TOKEN,
        "reasoning_content": "Alias thought",
        "reasoning_details": [{"type": "reasoning.text", "text": "Thought", "index": 0}],
    }
    events[1]["choices"][0]["delta"].update(reasoning)
    replies = []
    await collect(
        engine_client,
        [
            *request_messages(admission, "chat", stream=True),
            response_headers("text/event-stream", policy=admission),
            response_body(
                b"".join(event_bytes("chat", event) for event in events), policy=admission
            ),
        ],
        replies,
        FakeAioContext(),
    )
    emitted = b"".join(
        reply.response_body.response.body_mutation.streamed_response.body for reply in replies
    )
    frames = SseDecoder().feed(emitted.decode(), final=True)
    assert frames[-1].data == "[DONE]"
    deltas = [
        choice["delta"] for frame in frames[:-1] for choice in json.loads(frame.data)["choices"]
    ]
    actual = next(delta for delta in deltas if "reasoning_details" in delta)
    assert {key: actual[key] for key in reasoning} == reasoning
    assert "Jane Doe" in "".join(delta.get("content") or "" for delta in deltas)


@pytest.mark.parametrize(
    "field", ["reasoning", "reasoning_content", "reasoning_details", "thinking_blocks"]
)
async def test_assistant_reasoning_replay_inspects_client_modified_text(
    admission, engine_reply, field
):
    history = {"role": "assistant", "content": "Earlier answer", field: "Jane Doe"}
    if field == "reasoning_details":
        history[field] = [{"type": "reasoning.text", "text": "Jane Doe", "index": 0}]
    elif field == "thinking_blocks":
        history[field] = [{"type": "thinking", "thinking": "Jane Doe"}]
    payload = {
        "model": "example-chat",
        "messages": [history, {"role": "user", "content": "Continue"}],
    }
    seen = []

    def analyze(request):
        wire = json.loads(request.content)
        seen.extend(segment["text"] for segment in wire["segments"])
        reply = deepcopy(engine_reply)
        reply.pop("request")
        reply["api_version"] = "v2"
        reply["analysis"]["text_leaf_count"] = len(wire["segments"])
        reply["segments"] = [
            {**segment, "text": segment["text"].replace("Jane Doe", REVERSIBLE_TOKEN)}
            for segment in wire["segments"]
        ]
        return httpx.Response(200, json=reply)

    async with httpx.AsyncClient(transport=httpx.MockTransport(analyze)) as http:
        client = EngineClient(EngineSettings(base_url="https://engine.example.test"), http)
        replies = []
        await collect(
            client,
            [
                header_request(headers(), policy=admission),
                body_request(json.dumps(payload).encode(), policy=admission),
            ],
            replies,
            FakeAioContext(),
        )
    assert "Jane Doe" in seen
    assert replies[-1].HasField("request_body")
    forwarded = json.loads(replies[-1].request_body.response.body_mutation.body)["messages"][1]
    assert forwarded[field] == (
        [{"type": "reasoning.text", "text": REVERSIBLE_TOKEN, "index": 0}]
        if field == "reasoning_details"
        else [{"type": "thinking", "thinking": REVERSIBLE_TOKEN}]
        if field == "thinking_blocks"
        else REVERSIBLE_TOKEN
    )


@pytest.mark.parametrize("pii", [False, True])
@pytest.mark.parametrize(
    "case", ["unknown_json", "unknown_event", "chat_reasoning", "chat_reasoning_sse"]
)
async def test_review_repros_reject_before_transformation(
    admission, engine_client, caplog, case, pii
):
    admission["destination_policy"]["models"]["example-chat"] = pii
    stream = case in {"unknown_event", "chat_reasoning_sse"}
    kind = "responses" if case == "unknown_event" else "chat"
    if case == "unknown_json":
        payload = {"unqualified_feature": {"raw": SENTINEL}}
    elif case == "unknown_event":
        payload = {"type": "response.unqualified_content", "raw": SENTINEL, "sequence_number": 0}
    elif case == "chat_reasoning":
        payload = response_payload("chat", "safe text")
        payload["choices"][0]["message"]["reasoning_details"] = [
            {"type": "reasoning.text", "unqualified": SENTINEL}
        ]
    else:
        payload = stream_events("chat", "safe text")[0]
        payload["choices"][0]["delta"]["reasoning_details"] = [
            {"type": "reasoning.text", "unqualified": SENTINEL}
        ]
    messages = request_messages(admission, kind, stream=stream)
    messages += [
        response_headers("text/event-stream" if stream else "application/json", policy=admission),
        response_body(
            event_bytes(kind, payload) if stream else json.dumps(payload).encode(), policy=admission
        ),
    ]
    replies, context = [], FakeAioContext()
    if stream:
        with pytest.raises(StreamAbortedError):
            await collect(engine_client, messages, replies, context)
        assert context.code == grpc.StatusCode.INTERNAL
    else:
        await collect(engine_client, messages, replies, context)
        assert replies[-1].immediate_response.status.code == 503
    assert not any(reply.HasField("response_body") for reply in replies)
    assert SENTINEL not in str(replies) + caplog.text + context.details


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("error_debug", [False, True])
async def test_debug_failure_reports_exact_errors_and_full_payloads(
    admission, engine_client, engine_reply, monkeypatch, caplog, debug, stream, error_debug
):
    monkeypatch.setenv("EXTPROC_LOG_LEVEL", "DEBUG" if debug else "INFO")
    monkeypatch.setenv("ERROR_LOG_LEVEL", "debug" if error_debug else "info")
    caplog.set_level(logging.DEBUG, logger="agentgateway_extproc.controllers.grpc_servicer")
    secrets = {
        "api_key": "synthetic-api-secret",
        "cookie": "synthetic-cookie-secret",
        "recovery_phrase": "synthetic recovery material",
        "private_key": "synthetic-private-key",
        "secret_access_key": "synthetic-secret-access-key",
    }
    signed = {
        "type": "reasoning",
        "id": "rs-private",
        "summary": [{"type": "summary_text", "text": "signed-summary-sentinel"}],
        "encrypted_content": "opaque-continuation-sentinel",
    }
    payload = response_payload("responses", "ordinary response text", tools=True)
    payload["output"][0]["phase"] = "invalid"
    payload["output"][0]["content"][0]["unqualified"] = "ordinary extra text"
    payload["output"][1]["arguments"] = json.dumps(
        {
            "ordinary": "ordinary nested argument",
            **secrets,
            "url": "https://synthetic-user:synthetic-password@example.test/path",
            "signed_url": "https://example.test/path?token=synthetic-url-token",
            "sas_url": "https://example.test/path?sig=synthetic-url-signature",
            "nested": json.dumps({"ordinary": "ordinary encoded argument", "carrier": signed}),
            "partial": '{"password":"partial-credential-sentinel"',
            "Authorization": "Bearer credential-sentinel",
            "echo": "credential-sentinel",
            "chat": {
                "content": "ordinary signed-carrier sibling",
                "reasoning": "signed-chat-content",
                "reasoning_signature": "signed-chat-signature",
            },
            "signed": {
                "type": "thinking",
                "signature": "signature-sentinel",
                "thinking": "signed-thinking-sentinel",
            },
        }
    )
    payload["output"].append(signed)
    payload["tools"] = [{"type": "function", "name": "lookup", "output_schema": {"type": "object"}}]
    payload["instructions"] = "a-" * 40_000 + "response-trailing-sentinel"
    failure = (
        {"type": "response.completed", "sequence_number": 1, "response": payload}
        if stream
        else payload
    )
    messages = request_messages(admission, "responses", stream=stream)
    request_payload = json.loads(body(responses=True, stream=stream))
    request_payload["input"] = (
        "Jane Doe " + "a-" * 40_000 + json.dumps(secrets) + "request-trailing-sentinel"
    )
    engine_reply["request"]["messages"][0]["content"] = request_payload["input"].replace(
        "Jane Doe", REVERSIBLE_TOKEN
    )
    request_headers = headers(
        **{":path": f"/bindings/{admission['request_context']['binding_id']}/v1/responses"}
    )
    del request_headers["authorization"]  # The request fixture strips platform credentials.
    messages[1] = body_request(json.dumps(request_payload).encode(), policy=admission)
    messages.append(
        response_headers("text/event-stream" if stream else "application/json", policy=admission)
    )
    if stream:
        initial = stream_events("responses")[0]
        initial["response"]["instructions"] = "earlier-event-not-retained"
        messages.append(
            response_body(event_bytes("responses", initial), end_of_stream=False, policy=admission)
        )
    messages.append(
        response_body(
            event_bytes("responses", failure) if stream else json.dumps(failure).encode(),
            policy=admission,
        )
    )
    replies, context = [], FakeAioContext()
    with patch(
        "agentgateway_extproc.lib.pipeline.request.diagnostic_json", wraps=diagnostic_json
    ) as capture:
        if stream:
            with pytest.raises(StreamAbortedError):
                await collect(engine_client, messages, replies, context)
            assert context.code == grpc.StatusCode.INTERNAL
            assert context.details.startswith("response processing failed closed; request_id=")
            assert ("traceback" in context.details) is error_debug
            assert not any(reply.HasField("immediate_response") for reply in replies)
        else:
            await collect(engine_client, messages, replies, context)
            assert replies[-1].immediate_response.status.code == 503
        assert capture.called is debug
    forbidden = [
        *secrets.values(),
        "credential-sentinel",
        "synthetic-user",
        "synthetic-password",
        "synthetic-url-token",
        "synthetic-url-signature",
        "signed-summary-sentinel",
        "opaque-continuation-sentinel",
        "signature-sentinel",
        "signed-thinking-sentinel",
        "partial-credential-sentinel",
        "signed-chat-content",
        "signed-chat-signature",
    ]
    client_replies = (
        str([reply for reply in replies if reply.WhichOneof("response") != "request_body"])
        + context.details
    )
    assert all(value not in client_replies for value in forbidden)
    assert "response-trailing-sentinel" not in client_replies
    assert "request-trailing-sentinel" not in client_replies
    records = [
        record for record in caplog.records if record.msg.startswith("ext_proc failure diagnostics")
    ]
    assert len(records) == int(debug)
    if not debug:
        assert all(value not in caplog.text for value in forbidden)
        assert "ordinary response text" not in caplog.text and "Jane Doe" not in caplog.text
        assert "response-trailing-sentinel" not in caplog.text
        assert "request-trailing-sentinel" not in caplog.text
        return
    details = json.loads(records[0].args[2])
    assert json.loads(records[0].args[3]) == {
        "headers": request_headers,
        "body": request_payload,
    }
    assert details["response"] == (
        {"event": "response.completed", "data": failure} if stream else failure
    )
    prefix = ["response.completed", "response"] if stream else []
    errors = {(tuple(error["path"]), error["code"]) for error in details["validation"]["errors"]}
    assert ((*prefix, "output", 0, "message", "phase"), "literal_error") in errors
    assert (
        (*prefix, "output", 0, "message", "content", 0, "output_text", "unqualified"),
        "extra_forbidden",
    ) in errors
    assert ((*prefix, "tools", 0, "output_schema"), "none_required") in errors
    assert all(error["message"] for error in details["validation"]["errors"])
    assert "ordinary response text" in caplog.text and "ordinary nested argument" in caplog.text
    assert "ordinary encoded argument" in caplog.text
    assert "ordinary signed-carrier sibling" in caplog.text
    assert "earlier-event-not-retained" not in caplog.text
    if stream:
        assert details["response"]["event"] == "response.completed"


@pytest.mark.parametrize(
    "case", ["chat_nested_field", "chat_tool_field", "responses_part", "responses_reasoning"]
)
async def test_v5_closes_nested_output_shapes(admission, engine_client, case):
    kind = "chat" if case.startswith("chat") else "responses"
    payload = response_payload(kind, tools=True)
    if case == "chat_nested_field":
        payload["choices"][0]["unqualified_feature"] = SENTINEL
    elif case == "chat_tool_field":
        payload["choices"][0]["message"]["tool_calls"][0]["function"]["unqualified_feature"] = (
            SENTINEL
        )
    elif case == "responses_part":
        payload["output"][0]["content"][0]["encrypted_content"] = SENTINEL
    else:
        payload["output"] = [
            {"id": "rs-example", "type": "reasoning", "summary": [], "unqualified": SENTINEL}
        ]
    messages = [
        *request_messages(admission, kind, stream=False),
        response_headers("application/json", policy=admission),
        response_body(json.dumps(payload).encode(), policy=admission),
    ]
    replies = []
    await collect(engine_client, messages, replies, FakeAioContext())
    assert replies[-1].immediate_response.status.code == 503
    assert SENTINEL not in str(replies)


@pytest.mark.parametrize(
    ("kind", "stream", "empty_terminal"),
    [
        ("chat", False, False),
        ("chat", True, False),
        ("responses", False, False),
        ("responses", True, False),
        ("responses", True, True),
    ],
)
@pytest.mark.parametrize("pii", [False, True])
async def test_qualified_v5_text_tools_and_usage_roundtrip(
    admission, reasoning_engine, kind, pii, stream, empty_terminal
):
    engine_client, _ = reasoning_engine
    admission["destination_policy"]["models"]["example-chat"] = pii
    handler = StreamHandler(engine_client)
    for message in request_messages(admission, kind, stream=stream):
        await handler.handle(message)
    # This test compares provider content; notice qualification is covered independently.
    handler.response_notice_allowed = False
    await handler.handle(
        response_headers("text/event-stream" if stream else "application/json", policy=admission)
    )
    incoming = (
        b"".join(
            event_bytes(kind, event)
            for event in stream_events(
                kind, tools=True, reasoning=True, empty_terminal=empty_terminal
            )
        )
        if stream
        else json.dumps(response_payload(kind, tools=True, reasoning=True)).encode()
    )
    emitted = b""
    for position in range(0, len(incoming), 37):
        reply = await handler.handle(
            response_body(incoming[position : position + 37], end_of_stream=False, policy=admission)
        )
        if reply is not None:
            emitted += reply.response_body.response.body_mutation.streamed_response.body
    reply = await handler.handle(response_body(b"", policy=None))
    assert (
        reply is not None
        and reply.response_body.response.body_mutation.streamed_response.end_of_stream
    )
    emitted += reply.response_body.response.body_mutation.streamed_response.body
    text = "Jane Doe" if pii else REVERSIBLE_TOKEN
    if not stream:
        assert json.loads(emitted) == response_payload(kind, text, tools=True, reasoning=True)
    else:
        events = SseDecoder().feed(emitted.decode(), final=True)
        if kind == "chat":
            assert events[-1].data == "[DONE]"
            chunks = [json.loads(event.data) for event in events[:-1]]
            choices = [choice for chunk in chunks for choice in chunk["choices"]]
            assert "".join(choice["delta"].get("content", "") for choice in choices) == text
            arguments = "".join(
                tool["function"].get("arguments", "")
                for choice in choices
                for tool in choice["delta"].get("tool_calls", [])
            )
            assert json.loads(arguments) == {"person": text, "count": 1}
            assert chunks[-1]["usage"]["total_tokens"] == 30
        else:
            terminal = json.loads(events[-1].data)
            assert terminal["type"] == "response.completed"
            expected = response_payload(kind, text, tools=True, reasoning=True)
            completed_items = [
                json.loads(event.data)["item"]
                for event in events
                if json.loads(event.data)["type"] == "response.output_item.done"
            ]
            assert completed_items == expected["output"]
            if empty_terminal:
                expected["output"] = []
            assert terminal["response"] == expected
    assert handler.reversal_map == {}


@pytest.mark.parametrize("kind", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
async def test_v5_response_format_must_match_admitted_stream_flag(
    admission, engine_client, kind, stream
):
    messages = request_messages(admission, kind, stream=stream)
    messages.append(
        response_headers("application/json" if stream else "text/event-stream", policy=admission)
    )
    replies, context = [], FakeAioContext()
    await collect(engine_client, messages, replies, context)
    assert replies[-1].immediate_response.status.code == 503
    assert not any(reply.HasField("response_headers") for reply in replies)
    assert context.code is None


@pytest.mark.parametrize("kind", ["chat", "responses"])
@pytest.mark.parametrize("ending", ["body", "iterator", "trailers"])
async def test_v5_truncated_stream_aborts_after_commit(admission, engine_client, kind, ending):
    events = stream_events(kind, tools=True)
    if kind == "chat":
        events = [events[1]]  # A lone Chat delta is not a complete choice or stream.
    else:
        index = next(
            index
            for index, event in enumerate(events)
            if event["type"] == "response.function_call_arguments.delta"
        )
        events = events[: index + 1]
        events[-1]["delta"] = '{"incomplete":'
    messages = [
        *request_messages(admission, kind, stream=True),
        response_headers("text/event-stream", policy=admission),
        response_body(
            b"".join(event_bytes(kind, event) for event in events),
            end_of_stream=False,
            policy=admission,
        ),
    ]
    if ending == "body":
        messages.append(response_body(b"", policy=None))
    elif ending == "trailers":
        messages.append(add_policy(ext_proc_pb2.ProcessingRequest(response_trailers={}), admission))
    replies, context = [], FakeAioContext()
    with pytest.raises(StreamAbortedError):
        await collect(engine_client, messages, replies, context)
    assert context.code == grpc.StatusCode.INTERNAL
    assert any(reply.HasField("response_body") for reply in replies)
    assert not any(
        reply.response_body.response.body_mutation.streamed_response.end_of_stream
        for reply in replies
    )
    assert not any(
        reply.HasField("immediate_response") or reply.HasField("response_trailers")
        for reply in replies
    )


@pytest.mark.parametrize(
    "case",
    [
        "chat_missing_finish",
        "chat_open_arguments",
        "responses_missing_args_done",
        "responses_open_arguments",
        "responses_missing_item_done",
        "responses_missing_part_done",
        "responses_changed_snapshot",
        "responses_after_terminal",
    ],
)
@pytest.mark.parametrize("pii", [False, True])
async def test_v5_stream_requires_closed_consistent_tool_lifecycle(
    admission, engine_client, case, pii
):
    admission["destination_policy"]["models"]["example-chat"] = pii
    kind = "chat" if case.startswith("chat") else "responses"
    events = stream_events(kind, tools=True, empty_terminal=case != "responses_changed_snapshot")
    if case == "chat_missing_finish":
        events = [*events[:-3], "[DONE]"]
    elif case == "chat_open_arguments":
        events[3]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = (
            '{"incomplete":'
        )
        del events[4]
    elif case in {"responses_missing_args_done", "responses_missing_part_done"}:
        missing_type = {
            "responses_missing_args_done": "response.function_call_arguments.done",
            "responses_missing_part_done": "response.content_part.done",
        }[case]
        events = [event for event in events if event["type"] != missing_type]
    elif case == "responses_open_arguments":
        delta_seen = False
        for event in events:
            if event["type"] == "response.function_call_arguments.delta":
                event["delta"] = "" if delta_seen else '{"incomplete":'
                delta_seen = True
            elif event["type"] == "response.function_call_arguments.done":
                event["arguments"] = '{"incomplete":'
    elif case == "responses_missing_item_done":
        events = [
            event
            for event in events
            if not (event["type"] == "response.output_item.done" and event["output_index"] == 1)
        ]
    elif case == "responses_changed_snapshot":
        events[-1]["response"]["output"][1]["arguments"] = '{"different":true}'
    else:
        events.append(deepcopy(events[-1]))
        events[-1]["sequence_number"] += 1
    messages = [
        *request_messages(admission, kind, stream=True),
        response_headers("text/event-stream", policy=admission),
        response_body(b"".join(event_bytes(kind, event) for event in events), policy=admission),
    ]
    replies, context = [], FakeAioContext()
    with pytest.raises(StreamAbortedError):
        await collect(engine_client, messages, replies, context)
    assert context.code == grpc.StatusCode.INTERNAL
    assert not any(reply.HasField("response_body") for reply in replies)


def refusal_parts(value):
    if isinstance(value, dict):
        if value.get("type") in {
            "output_text",
            "response.output_text.delta",
            "response.output_text.done",
        }:
            value["type"] = value["type"].replace("output_text", "refusal")
            if "text" in value:
                value["refusal"] = value.pop("text")
            value.pop("logprobs", None)
            value.pop("annotations", None)
        for nested in value.values():
            refusal_parts(nested)
    elif isinstance(value, list):
        for nested in value:
            refusal_parts(nested)


@pytest.mark.parametrize("kind", ["chat", "responses"])
async def test_v5_refusals_are_restored_instead_of_becoming_opaque_streams(
    admission, engine_client, kind
):
    events = stream_events(kind)
    if kind == "chat":
        for event in events[:-1]:
            for choice in event["choices"]:
                if "content" in choice["delta"]:
                    choice["delta"]["refusal"] = choice["delta"].pop("content")
    else:
        refusal_parts(events)
    messages = [
        *request_messages(admission, kind, stream=True),
        response_headers("text/event-stream", policy=admission),
        response_body(b"".join(event_bytes(kind, event) for event in events), policy=admission),
    ]
    replies = []
    await collect(engine_client, messages, replies, FakeAioContext())
    assert replies[-1].HasField("response_body")
    emitted = replies[-1].response_body.response.body_mutation.streamed_response.body
    assert b"Jane Doe" in emitted and REVERSIBLE_TOKEN.encode() not in emitted
