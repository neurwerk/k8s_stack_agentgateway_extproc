"""V5 closed response schemas and protocol completion at the real gRPC boundary."""

import json
from copy import deepcopy

import grpc
import pytest

from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.gen import ext_proc_pb2
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

SENTINEL = "synthetic-sentinel"


def request_messages(admission, kind, *, stream):
    admission["request_context"]["protocol_profile"] = (
        "openai-responses-v1" if kind == "responses" else "openai-chat-completions-v1"
    )
    path = "/v1/responses" if kind == "responses" else "/v1/chat/completions"
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
        payload["choices"][0]["message"]["reasoning_content"] = REVERSIBLE_TOKEN + SENTINEL
    else:
        payload = stream_events("chat", "safe text")[0]
        payload["choices"][0]["delta"]["reasoning_content"] = REVERSIBLE_TOKEN + SENTINEL
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
            {"id": "rs-example", "type": "reasoning", "summary": [], "encrypted_content": SENTINEL}
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


@pytest.mark.parametrize("kind", ["chat", "responses"])
@pytest.mark.parametrize("pii", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_qualified_v5_text_tools_and_usage_roundtrip(
    admission, engine_client, kind, pii, stream
):
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
        b"".join(event_bytes(kind, event) for event in stream_events(kind, tools=True))
        if stream
        else json.dumps(response_payload(kind, tools=True)).encode()
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
        assert json.loads(emitted) == response_payload(kind, text, tools=True)
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
            assert terminal["response"] == response_payload(kind, text, tools=True)
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
    events = stream_events(kind, tools=True)
    if case == "chat_missing_finish":
        events = [*events[:-3], "[DONE]"]
    elif case == "chat_open_arguments":
        events[3]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = (
            '{"incomplete":'
        )
        del events[4]
    elif case == "responses_missing_args_done":
        events = [
            event for event in events if event["type"] != "response.function_call_arguments.done"
        ]
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
