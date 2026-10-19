"""Protect child text included in the next PII-enabled remote parent request."""

import json

import httpx

from agentgateway_extproc.config.settings import EngineSettings
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler

from .conftest import REVERSIBLE_TOKEN, body_request, header_request


async def test_next_pii_enabled_parent_request_masks_subagent_tool_result(engine_reply):
    child_result = "The document names Jane Doe as the contact."
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "subagent",
                    "arguments": json.dumps(
                        {"description": "Read the shared document", "subagent_type": "local-child"}
                    ),
                },
            }
        ],
    }
    tool_result = {"role": "tool", "tool_call_id": "call-1", "content": child_result}
    payload = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "Find the document contact"},
            assistant,
            tool_result,
        ],
    }
    calls = 0

    def analyze(request):
        nonlocal calls
        calls += 1
        assert request.method == "POST"
        assert request.url.path == "/v2/adapter/analyze-segments"
        wire = json.loads(request.content)
        assert wire["text_pii_enabled"] is True
        assert [segment["text"] for segment in wire["segments"]].count(child_result) == 1
        engine_reply.pop("request")
        engine_reply["api_version"] = "v2"
        engine_reply["analysis"]["text_leaf_count"] = len(wire["segments"])
        engine_reply["segments"] = [
            {**segment, "text": segment["text"].replace("Jane Doe", REVERSIBLE_TOKEN)}
            for segment in wire["segments"]
        ]
        return httpx.Response(200, json=engine_reply)

    async with httpx.AsyncClient(transport=httpx.MockTransport(analyze)) as http:
        client = EngineClient(EngineSettings(base_url="https://engine.example.test"), http)
        handler = StreamHandler(client)
        await handler.handle(header_request())
        response = await handler.handle(body_request(json.dumps(payload).encode()))

    assert calls == 1
    assert response is not None and response.WhichOneof("response") == "request_body"
    forwarded_body = response.request_body.response.body_mutation.body
    assert b"Jane Doe" not in forwarded_body
    forwarded = json.loads(forwarded_body)
    assert forwarded["model"] == "test"
    pair = [
        message for message in forwarded["messages"] if message["role"] in {"assistant", "tool"}
    ]
    assert len(pair) == 2
    assert pair[0] == assistant
    assert pair[1] == {
        **tool_result,
        "content": child_result.replace("Jane Doe", REVERSIBLE_TOKEN),
    }
