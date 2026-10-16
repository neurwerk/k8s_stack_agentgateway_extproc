"""Round-trip actual request masking through supported Responses reflections."""

import json
from copy import deepcopy

import httpx
import pytest
from pydantic import ValidationError

from agentgateway_extproc.config.settings import EngineSettings
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.pipeline.guard import GUARD_INSTRUCTION
from agentgateway_extproc.lib.pipeline.sse import SseDecoder
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.exceptions import InvalidReversalError

from .conftest import (
    REVERSIBLE_TOKEN,
    body_request,
    header_request,
    response_body,
    response_headers,
)
from .openai_fixtures import event_bytes, response_payload, stream_events
from .test_admission import admission as admission
from .test_admission import headers

REFLECTED = ("instructions", "tools", "text")
PRIVATE_TEXT = 'Jane "Doe"\n'
FOREIGN_TOKEN = "<REV_PERSON_aaaaaaaaaaaaaaaa_bbbbbbbbbbbbbbbb>"


@pytest.fixture
def reflected_request():
    schema = {
        "type": "object",
        "title": f"Record for {PRIVATE_TEXT}",
        "$defs": {
            "person": {
                "type": "string",
                "description": f"A person such as {PRIVATE_TEXT}",
                "enum": ["control-value"],
            }
        },
        "properties": {
            "person": {
                "$ref": "#/$defs/person",
                "description": f"Find {PRIVATE_TEXT}",
            },
        },
        "required": ["person"],
        "additionalProperties": False,
        "default": {"person": PRIVATE_TEXT, "count": 1, "enabled": False, "absent": None},
        "examples": [{"person": PRIVATE_TEXT}, [PRIVATE_TEXT, 1, False, None]],
        "x-contract": {"description": "uninspected extension control"},
    }
    return {
        "model": "example-chat",
        "input": "Please perform a lookup.",
        "instructions": f"Help {PRIVATE_TEXT}",
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": f"Look up {PRIVATE_TEXT}",
                "parameters": deepcopy(schema),
                "strict": False,
            },
            {"type": "function", "name": "optional", "parameters": None, "strict": None},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "lookup_result",
                "description": f"Result for {PRIVATE_TEXT}",
                "schema": schema,
                "strict": False,
            },
            "verbosity": "low",
        },
    }


@pytest.fixture
async def reflection_engine(engine_reply):
    calls = []

    def analyze(request):
        wire = json.loads(request.content)
        calls.append(wire)
        assert wire["api_version"] == "v2" and wire["request_kind"] == "responses"
        segments = wire["segments"]
        count = sum(segment["text"].count(PRIVATE_TEXT) for segment in segments)
        reply = deepcopy(engine_reply)
        reply.pop("request")
        reply.update(
            api_version="v2",
            segments=[
                {**segment, "text": segment["text"].replace(PRIVATE_TEXT, REVERSIBLE_TOKEN)}
                for segment in segments
            ],
            reversal={REVERSIBLE_TOKEN: PRIVATE_TEXT} if count else {},
            entity_counts={"PERSON": count} if count else {},
        )
        reply["analysis"]["text_leaf_count"] = len(segments)
        if count:
            reply["report"]["rows"][0].update(
                detected_count=count, transformed_count=count, unique_transformed_count=1
            )
        else:
            reply.update(
                decision="pass", entities=[], applied_actions=["pass"], report={"rows": []}
            )
        return httpx.Response(200, json=reply)

    async with httpx.AsyncClient(transport=httpx.MockTransport(analyze)) as http:
        yield EngineClient(EngineSettings(base_url="https://engine.example.test"), http), calls


async def inspect_request(client, admission, payload, *, stream):
    admission["request_context"]["protocol_profile"] = "openai-responses-v1"
    handler = StreamHandler(client)
    await handler.handle(
        header_request(
            headers(
                **{
                    ":path": "/bindings/openai-api/v1/responses",
                    "x-session-id": "example-conversation",
                }
            ),
            policy=admission,
        )
    )
    reply = await handler.handle(
        body_request(
            json.dumps({**payload, "stream": stream}).encode(),
            policy=admission,
        )
    )
    assert reply is not None and reply.HasField("request_body")
    return handler, json.loads(reply.request_body.response.body_mutation.body)


def provider_response(forwarded, *, stream, incomplete=False, phase_fields=None):
    fields = {field: deepcopy(forwarded[field]) for field in REFLECTED if field in forwarded}
    if fields.get("tools"):
        fields["tools"][0].setdefault("output_schema", None)
    if not stream:
        payload = {**response_payload("responses", '{"person":"safe"}'), **fields}
        payload["output"][0].update(phase_fields or {})
        return json.dumps(payload).encode()
    events = stream_events("responses", '{"person":"safe"}')
    events.insert(1, {**deepcopy(events[0]), "type": "response.in_progress"})
    if incomplete:
        events[-1]["type"] = "response.incomplete"
        events[-1]["response"].update(
            status="incomplete", incomplete_details={"reason": "max_output_tokens"}
        )
    for index, event in enumerate(events):
        event["sequence_number"] = index
        if event.get("item", {}).get("type") == "message":
            event["item"].update(phase_fields or {})
        if "response" in event:
            event["response"].update(deepcopy(fields))
            for item in event["response"]["output"]:
                if item["type"] == "message":
                    item.update(phase_fields or {})
    return b"".join(event_bytes("responses", event) for event in events)


@pytest.mark.parametrize("transport", ["json", "sse_completed", "sse_incomplete"])
@pytest.mark.parametrize(
    "phase_fields", [{}, {"phase": None}, {"phase": "commentary"}, {"phase": "final_answer"}]
)
async def test_actual_masked_reflections_restore_only_canonical_text(
    admission, reflection_engine, reflected_request, transport, phase_fields
):
    client, calls = reflection_engine
    stream = transport != "json"
    handler, forwarded = await inspect_request(client, admission, reflected_request, stream=stream)
    assert len(calls) == 1
    assert PRIVATE_TEXT not in str(forwarded)
    assert REVERSIBLE_TOKEN in forwarded["instructions"]
    assert REVERSIBLE_TOKEN in forwarded["tools"][0]["description"]
    assert (
        REVERSIBLE_TOKEN
        in forwarded["tools"][0]["parameters"]["properties"]["person"]["description"]
    )
    await handler.handle(
        response_headers("text/event-stream" if stream else "application/json", policy=admission)
    )
    incoming = provider_response(
        forwarded,
        stream=stream,
        incomplete=transport == "sse_incomplete",
        phase_fields=phase_fields,
    )
    output = b""
    for index in range(0, len(incoming), 113):
        reply = await handler.handle(
            response_body(incoming[index : index + 113], end_of_stream=False, policy=admission)
        )
        if reply is not None:
            output += reply.response_body.response.body_mutation.streamed_response.body
    reply = await handler.handle(response_body(b"", policy=None))
    assert (
        reply is not None
        and reply.response_body.response.body_mutation.streamed_response.end_of_stream
    )
    output += reply.response_body.response.body_mutation.streamed_response.body
    if stream:
        events = [
            json.loads(event.data) for event in SseDecoder().feed(output.decode(), final=True)
        ]
        snapshots = [event["response"] for event in events if "response" in event]
        assert len(snapshots) == 3
        assert events[-1]["type"] == (
            "response.incomplete" if transport == "sse_incomplete" else "response.completed"
        )
    else:
        snapshots = [json.loads(output)]
    for snapshot in snapshots:
        assert snapshot["instructions"] == "\n\n" + reflected_request["instructions"]
        assert snapshot["tools"] == [
            reflected_request["tools"][0] | {"output_schema": None},
            reflected_request["tools"][1],
        ]
        for item in snapshot["output"]:
            assert {key: item[key] for key in ("phase",) if key in item} == phase_fields
        assert snapshot["text"] == reflected_request["text"]
        assert snapshot["model"] == "provider-chat"
        assert snapshot["metadata"] == {}
    assert GUARD_INSTRUCTION.encode() not in output and REVERSIBLE_TOKEN.encode() not in output
    assert handler.reversal_map == {} and len(calls) == 1


@pytest.mark.parametrize(
    "phase_fields", [{}, {"phase": None}, {"phase": "commentary"}, {"phase": "final_answer"}]
)
async def test_assistant_phase_replay_preserves_controls_and_masks_text(
    admission, reflection_engine, phase_fields
):
    client, calls = reflection_engine
    message = {
        "type": "message",
        "id": "msg-replay",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": PRIVATE_TEXT}],
        **phase_fields,
    }
    handler, forwarded = await inspect_request(
        client, admission, {"model": "example-chat", "input": [message]}, stream=False
    )
    assert PRIVATE_TEXT in [segment["text"] for segment in calls[0]["segments"]]
    assert handler.request_stats.analysis.scan_performed
    assert forwarded["input"] == [
        message | {"content": [{"type": "output_text", "text": REVERSIBLE_TOKEN}]}
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "violation",
    [
        "foreign_token",
        "schema_key",
        "enum",
        "ref",
        "extension",
        "tool_name",
        "unknown_tool_field",
        "output_schema",
        "phase",
        "metadata",
    ],
)
async def test_reflections_reject_foreign_tokens_and_protocol_placeholders(
    admission, reflection_engine, reflected_request, stream, violation, caplog
):
    client, calls = reflection_engine
    handler, forwarded = await inspect_request(client, admission, reflected_request, stream=stream)
    schema = forwarded["tools"][0]["parameters"]
    if violation == "foreign_token":
        forwarded["instructions"] = FOREIGN_TOKEN
    elif violation == "schema_key":
        schema["properties"][REVERSIBLE_TOKEN] = {"type": "string"}
    elif violation == "enum":
        schema["$defs"]["person"]["enum"] = [REVERSIBLE_TOKEN]
    elif violation == "ref":
        schema["properties"]["person"]["$ref"] = REVERSIBLE_TOKEN
    elif violation == "extension":
        schema["x-contract"] = {"type": "output_text", "text": json.dumps(REVERSIBLE_TOKEN)}
    elif violation == "tool_name":
        forwarded["tools"][0]["name"] = REVERSIBLE_TOKEN
    elif violation == "unknown_tool_field":
        forwarded["tools"][0]["unqualified_feature"] = "synthetic-sentinel"
    elif violation == "output_schema":
        forwarded["tools"][0]["output_schema"] = {"type": "object"}
    incoming = provider_response(
        forwarded,
        stream=stream,
        phase_fields={"phase": "invalid"} if violation == "phase" else None,
    )
    if violation == "metadata":
        incoming = incoming.replace(
            b'"metadata": {}',
            json.dumps({"metadata": {"description": REVERSIBLE_TOKEN}}).encode()[1:-1],
        )
    await handler.handle(
        response_headers("text/event-stream" if stream else "application/json", policy=admission)
    )
    with pytest.raises((InvalidReversalError, ValidationError)):
        await handler.handle(response_body(incoming, policy=admission))
    assert len(calls) == 1
    assert PRIVATE_TEXT not in caplog.text and "synthetic-sentinel" not in caplog.text


async def test_reflection_reversal_does_not_cross_request_boundaries(
    admission, reflection_engine, reflected_request
):
    client, calls = reflection_engine
    first, forwarded = await inspect_request(client, admission, reflected_request, stream=False)
    await first.handle(response_headers("application/json", policy=admission))
    await first.handle(response_body(provider_response(forwarded, stream=False), policy=admission))
    assert first.reversal_map == {}
    admission["request_context"]["request_id"] = "another-request"
    second, _ = await inspect_request(
        client, admission, {"model": "example-chat", "input": "Hello"}, stream=False
    )
    assert second.reversal_map == {}
    await second.handle(response_headers("application/json", policy=admission))
    with pytest.raises(InvalidReversalError):
        await second.handle(
            response_body(provider_response(forwarded, stream=False), policy=admission)
        )
    assert len(calls) == 2
