"""Reasoning crosses real admission, segment HTTP and response completion boundaries."""

import json
from copy import deepcopy
from typing import Any

import httpx
import pytest

from agentgateway_extproc.config.settings import EngineSettings, Settings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.pipeline.sse import SseDecoder

from .conftest import (
    REVERSIBLE_TOKEN,
    body_request,
    header_request,
    response_body,
    response_headers,
)
from .openai_fixtures import event_bytes, reasoning_value, response_payload, stream_events
from .test_admission import admission as admission
from .test_admission import body, headers
from .test_fail_closed import FakeAioContext, StreamAbortedError


@pytest.fixture
async def reasoning_engine(engine_reply):
    state: dict[str, Any] = {"values": set(), "segments": [], "receipts": [], "unavailable": False}

    def transport(request):
        wire = json.loads(request.content)
        if request.url.path.endswith("reasoning-receipts"):
            state["receipts"].append(wire)
            if state["unavailable"]:
                return httpx.Response(503)
            keys = {(wire["scope"], digest) for digest in wire["digests"]}
            if request.method == "PUT":
                state["values"].update(keys)
            elif not keys <= state["values"]:
                return httpx.Response(403)
            return httpx.Response(204)
        state["segments"].extend(wire["segments"])
        reply = deepcopy(engine_reply)
        reply.pop("request")
        reply["api_version"] = "v2"
        reply["analysis"]["text_leaf_count"] = len(wire["segments"])
        count = sum(segment["text"].count("Jane Doe") for segment in wire["segments"])
        reply["segments"] = [
            {**segment, "text": segment["text"].replace("Jane Doe", REVERSIBLE_TOKEN)}
            for segment in wire["segments"]
        ]
        if count:
            reply["entity_counts"] = {"PERSON": count}
            reply["report"]["rows"][0].update(detected_count=count, transformed_count=count)
        else:
            reply.update(
                decision="pass",
                entities=[],
                entity_counts={},
                applied_actions=[],
                report={"rows": []},
                reversal={},
            )
        if state.get("reroute"):
            reply.update(decision="reroute", route_class="local", remote_allowed=False)
            reply["applied_actions"].append("reroute")
            reply["entities"].append("SENSITIVE")
            reply["entity_counts"]["SENSITIVE"] = 1
            reply["report"]["rows"].append(
                {
                    "entity_type": "SENSITIVE",
                    "action": "reroute",
                    "detected_count": 1,
                    "transformed_count": 0,
                    "unique_transformed_count": 0,
                }
            )
        return httpx.Response(200, json=reply)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
        yield EngineClient(EngineSettings(base_url="https://engine.example.test"), http), state


def request_messages(admission, kind, payload=None, *, stream=False):
    admission["request_context"]["protocol_profile"] = (
        "openai-responses-v1" if kind == "responses" else "openai-chat-completions-v1"
    )
    path = "responses" if kind == "responses" else "chat/completions"
    path = f"/bindings/{admission['request_context']['binding_id']}/v1/{path}"
    return [
        header_request(headers(**{":path": path}), policy=admission),
        body_request(
            json.dumps(payload).encode()
            if payload
            else body(responses=kind == "responses", stream=stream),
            policy=admission,
        ),
    ]


async def exchange(client, messages, replies=None, settings=None):
    replies = [] if replies is None else replies

    async def requests():
        for message in messages:
            yield message

    async for reply in ExtProcServicer(client, settings).Process(requests(), FakeAioContext()):
        replies.append(reply)
    return replies


def emitted_reasoning(kind, stream, replies):  # noqa: C901
    emitted = b"".join(
        reply.response_body.response.body_mutation.streamed_response.body for reply in replies
    )
    if not stream:
        payload = json.loads(emitted)
        return (
            payload["output"][0]
            if kind == "responses"
            else {key: payload["choices"][0]["message"][key] for key in reasoning_value(kind)}
        )
    frames = SseDecoder().feed(emitted.decode(), final=True)
    if kind == "responses":
        return json.loads(frames[-1].data)["response"]["output"][0]
    assert frames[-1].data == "[DONE]"
    # Assemble the client's received deltas, including late signatures.
    value = {}
    for frame in frames[:-1]:
        for choice in json.loads(frame.data)["choices"]:
            for key, fragment in choice["delta"].items():
                if key not in reasoning_value(kind):
                    continue
                if isinstance(fragment, str):
                    value[key] = value.get(key, "") + fragment
                else:
                    blocks = value.setdefault(key, [])
                    for delta in fragment:
                        index = delta["index"]
                        if index == len(blocks):
                            blocks.append({"index": index, "type": delta["type"]})
                        for field, text in delta.items():
                            if field in {"id", "format"}:
                                blocks[index][field] = text
                            elif field not in {"index", "type"}:
                                blocks[index][field] = blocks[index].get(field, "") + text
    return value


def replay_payload(kind, reasoning):
    if kind == "chat":
        return {
            "model": "example-chat",
            "messages": [
                {"role": "assistant", "content": "Earlier answer", **reasoning},
                {"role": "user", "content": "Continue"},
            ],
        }
    return {"model": "example-chat", "input": [reasoning, {"role": "user", "content": "Continue"}]}


def native_reasoning_events(kind):
    events = stream_events(kind, tools=True, reasoning=True)
    if kind == "responses":
        events[1]["item"]["encrypted_content"] = reasoning_value(kind)["encrypted_content"]
        events[1]["item"]["content"] = []
        events.insert(
            1,
            {
                "type": "response.metadata",
                "response_id": events[0]["response"]["id"],
                "metadata": {
                    "openai_chatgpt_moderation_metadata": {
                        "prompt": {
                            "omnimod": {
                                "protection_type": "synthetic",
                                "outputs": [
                                    {
                                        "results": [{"category_flags": {"A1": False}}],
                                        "is_blocked": False,
                                    }
                                ],
                            }
                        },
                        "generation": {},
                        "tool_call": {},
                        "tool_response": {},
                    }
                },
            },
        )
        for index, event in enumerate(events):
            event["sequence_number"] = index
    return events


@pytest.mark.parametrize("kind", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reroute", [False, True])
async def test_reasoning_survives_completed_exchange_and_verified_replay(
    admission, reasoning_engine, engine_reply, kind, stream, reroute, caplog
):
    client, state = reasoning_engine
    if reroute:
        state["reroute"] = True
        admission["request_context"]["provider_credential_source"] = "company"
        admission["request_context"]["permissions"].extend(
            ["provider:openai:company-use", "model:local:invoke"]
        )
        admission["model_bindings"]["example-chat"]["allowed_reroutes"] = ["local"]
        admission["reroute_targets"] = {
            "local": {
                "binding_id": "local-binding",
                "upstream_model": "local-provider-model",
                "required_permissions": ["llm:invoke", "model:local:invoke"],
                "content_tracing_enabled": False,
            }
        }
    incoming = (
        b"".join(event_bytes(kind, event) for event in native_reasoning_events(kind))
        if stream
        else json.dumps(response_payload(kind, tools=True, reasoning=True)).encode()
    )
    messages = [
        *request_messages(admission, kind, stream=stream),
        response_headers("text/event-stream" if stream else "application/json", policy=admission),
    ]
    messages.extend(
        response_body(incoming[index : index + 37], end_of_stream=False, policy=admission)
        for index in range(0, len(incoming), 37)
    )
    messages.append(response_body(b"", policy=None))
    replies = await exchange(client, messages)
    assert replies[-1].response_body.response.body_mutation.streamed_response.end_of_stream
    reasoning = emitted_reasoning(kind, stream, replies)
    assert reasoning == reasoning_value(kind)
    if kind == "responses" and stream:
        emitted = b"".join(
            reply.response_body.response.body_mutation.streamed_response.body for reply in replies
        )
        events = [
            json.loads(frame.data) for frame in SseDecoder().feed(emitted.decode(), final=True)
        ]
        assert events[1] == native_reasoning_events(kind)[1]
        assert events[2]["item"] == native_reasoning_events(kind)[2]["item"]
    assert state["values"]
    state["segments"].clear()
    admission["request_context"]["request_id"] = "next-request"
    # Responses summaries are editable even when encrypted state accompanies them.
    if kind == "responses":
        reasoning["summary"][0]["text"] = "Jane Doe"
        reasoning["content"][0]["text"] = "Jane Doe"
    else:
        reasoning["reasoning_details"][1]["summary"] = "Jane Doe"
    replies = await exchange(
        client, request_messages(admission, kind, replay_payload(kind, reasoning))
    )
    assert replies[-1].HasField("request_body")
    assert len(state["receipts"]) == 2
    assert all(
        receipt["policy_version"] == engine_reply["analysis"]["policy_version"]
        for receipt in state["receipts"]
    )
    forwarded = json.loads(replies[-1].request_body.response.body_mutation.body)
    if kind == "responses":
        assert forwarded["input"][0]["encrypted_content"] == reasoning["encrypted_content"]
        assert forwarded["input"][0]["summary"][0]["text"] == REVERSIBLE_TOKEN
        assert forwarded["input"][0]["content"][0]["text"] == REVERSIBLE_TOKEN
    else:
        message = forwarded["messages"][1]
        assert message["reasoning"] == reasoning["reasoning"]
        assert message["reasoning_content"] == reasoning["reasoning_content"]
        assert message["reasoning_signature"] == reasoning["reasoning_signature"]
        assert message["reasoning_details"][0] == reasoning["reasoning_details"][0]
        assert message["reasoning_details"][2] == reasoning["reasoning_details"][2]
        assert message["thinking_blocks"] == reasoning["thinking_blocks"]
        assert message["reasoning_details"][1]["summary"] == REVERSIBLE_TOKEN
    wire = json.dumps(state["segments"])
    retained = json.dumps(state["receipts"]) + repr(state["values"]) + caplog.text
    assert "Jane Doe" in wire
    for private in ("opaque-", "signature", "Signed thought", "Signed <REV_", "Alias thought"):
        assert private not in wire + retained


@pytest.mark.parametrize(
    "change", ["missing", "edited", "principal", "binding", "model", "policy", "outage"]
)
async def test_opaque_replay_cannot_borrow_or_forge_evidence(admission, reasoning_engine, change):
    client, state = reasoning_engine
    kind = "chat"
    replies = await exchange(
        client,
        [
            *request_messages(admission, kind),
            response_headers("application/json", policy=admission),
            response_body(
                json.dumps(response_payload(kind, reasoning=True)).encode(), policy=admission
            ),
        ],
    )
    reasoning = emitted_reasoning(kind, False, replies)
    state["segments"].clear()
    if change == "missing":
        state["values"].clear()
    elif change == "edited":
        reasoning["reasoning_details"][0]["text"] += " Jane Doe"
    elif change == "principal":
        admission["request_context"]["principal_id"] = "other-principal"
        admission["destination_policy"]["principal_id"] = "other-principal"
    elif change == "binding":
        admission["request_context"]["binding_id"] = "other-binding"
    elif change == "model":
        admission["model_bindings"]["example-chat"]["upstream_model"] = "other-model"
    elif change == "policy":
        admission["request_context"]["policy_revision"] = "other-policy"
    else:
        state["unavailable"] = True
    replies = await exchange(
        client, request_messages(admission, kind, replay_payload(kind, reasoning))
    )
    assert replies[-1].immediate_response.status.code == (503 if change == "outage" else 400)
    assert "signature" not in json.dumps(state["segments"])
    assert not any(reply.HasField("request_body") for reply in replies)


@pytest.mark.parametrize(
    "failure",
    [
        "truncated",
        "missing_item",
        "changed_snapshot",
        "late_reasoning",
        "changed_identity",
        "trailing_chat",
        "trailing_responses",
    ],
)
async def test_incomplete_reasoning_never_authorizes_replay(admission, reasoning_engine, failure):
    client, state = reasoning_engine
    kind = (
        "chat"
        if failure in {"late_reasoning", "changed_identity", "trailing_chat"}
        else "responses"
    )
    events = native_reasoning_events(kind)
    if failure == "truncated":
        events.pop()
    elif failure == "missing_item":
        events = [
            event
            for event in events
            if not (event["type"] == "response.output_item.done" and event["output_index"] == 0)
        ]
    elif failure == "changed_snapshot":
        events[-1]["response"]["output"][0]["encrypted_content"] += "changed"
    elif failure.startswith("trailing_"):
        events.append(deepcopy(events[-1]))
    elif failure == "changed_identity":
        for event in events[:-1]:
            for choice in event["choices"]:
                for detail in choice["delta"].get("reasoning_details", []):
                    if detail.get("signature"):
                        detail["id"] = "changed-identity"
    else:
        admission["request_context"]["provider_id"] = "openrouter"
        admission["request_context"]["permissions"].append("provider:openrouter:company-use")
        events[-2]["choices"] = [
            {"index": 0, "delta": {"reasoning": "after finish"}, "finish_reason": "tool_calls"}
        ]
    messages = [
        *request_messages(admission, kind, stream=True),
        response_headers("text/event-stream", policy=admission),
    ]
    messages.extend(
        response_body(event_bytes(kind, event), end_of_stream=False, policy=admission)
        for event in events
    )
    messages.append(response_body(b"", policy=None))
    with pytest.raises(StreamAbortedError):
        await exchange(client, messages)
    assert not state["values"]
    replies = await exchange(
        client, request_messages(admission, kind, replay_payload(kind, reasoning_value(kind)))
    )
    assert replies[-1].immediate_response.status.code == 400


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ((2, "item", "summary"), [{"type": "summary_text", "text": "premature text"}]),
        ((2, "item", "content"), [{"type": "reasoning_text", "text": "premature text"}]),
        ((1, "response_id"), "other-response"),
        ((1, "sequence_number"), 0),
        (
            (
                1,
                "metadata",
                "openai_chatgpt_moderation_metadata",
                "prompt",
                "omnimod",
                "outputs",
                0,
                "results",
                0,
                "category_flags",
                "A1",
            ),
            "false",
        ),
    ],
    ids=[
        "initial-summary",
        "initial-content",
        "metadata-identity",
        "metadata-sequence",
        "metadata-shape",
    ],
)
async def test_invalid_native_reasoning_events_never_authorize_replay(
    admission, reasoning_engine, path, replacement
):
    client, state = reasoning_engine
    events = native_reasoning_events("responses")
    target = events
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    messages = [
        *request_messages(admission, "responses", stream=True),
        response_headers("text/event-stream", policy=admission),
        *[
            response_body(event_bytes("responses", event), end_of_stream=False, policy=admission)
            for event in events
        ],
        response_body(b"", policy=None),
    ]
    with pytest.raises(StreamAbortedError):
        await exchange(client, messages)
    assert state["receipts"] == [] and not state["values"]
    replies = await exchange(
        client,
        request_messages(
            admission, "responses", replay_payload("responses", reasoning_value("responses"))
        ),
    )
    assert replies[-1].immediate_response.status.code == 400


async def test_final_output_limit_does_not_record_reasoning(admission, reasoning_engine):
    client, state = reasoning_engine
    incoming = json.dumps(response_payload("chat", reasoning=True)).encode()
    replies = await exchange(
        client,
        [
            *request_messages(admission, "chat"),
            response_headers("application/json", policy=admission),
            response_body(incoming, policy=admission),
        ],
        settings=Settings(max_response_bytes=len(incoming)),
    )
    assert replies[-1].immediate_response.status.code == 503
    assert state["receipts"] == [] and not state["values"]
