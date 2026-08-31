"""Test target-scoped detailed reports across supported response transports."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.engine import AnalysisMetadata, PIIReport
from agentgateway_extproc.models.types import RequestStats

from .conftest import REVERSIBLE_TOKEN, response_body, response_headers


async def test_chat_json_counts_only_the_selected_message_content(engine_client) -> None:
    handler = _handler(engine_client)
    await handler.handle(response_headers("application/json"))
    arguments = json.dumps({"person": REVERSIBLE_TOKEN})
    payload = {
        "choices": [
            {
                "index": 1,
                "message": {
                    "content": REVERSIBLE_TOKEN,
                    "tool_calls": [{"function": {"arguments": arguments}}],
                },
            },
            {"index": 0, "message": {"content": REVERSIBLE_TOKEN}},
        ]
    }

    response = await handler.handle(response_body(json.dumps(payload).encode()))

    assert response is not None
    output = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    choices = {choice["index"]: choice for choice in output["choices"]}
    assert choices[0]["message"]["content"].startswith("Jane Doe\n\n---\nPII Engine Notice")
    assert "| Person |" in choices[0]["message"]["content"]
    assert "| 1 restored |" in choices[0]["message"]["content"]
    assert choices[1]["message"]["content"] == "Jane Doe"
    assert json.loads(choices[1]["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "person": "Jane Doe"
    }


async def test_responses_json_counts_only_the_selected_output_text(engine_client) -> None:
    handler = _handler(engine_client, api_kind="responses")
    await handler.handle(response_headers("application/json"))
    arguments = json.dumps({"person": REVERSIBLE_TOKEN})
    payload = {
        "status": "completed",
        "output": [
            {"type": "function_call", "arguments": arguments},
            {
                "id": "msg_1",
                "type": "message",
                "content": [{"type": "output_text", "text": REVERSIBLE_TOKEN}],
            },
            {
                "id": "msg_2",
                "type": "message",
                "content": [{"type": "output_text", "text": REVERSIBLE_TOKEN}],
            },
        ],
    }

    response = await handler.handle(response_body(json.dumps(payload).encode()))

    assert response is not None
    output = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    assert json.loads(output["output"][0]["arguments"]) == {"person": "Jane Doe"}
    selected = output["output"][1]["content"][0]["text"]
    assert selected.startswith("Jane Doe\n\n---\nPII Engine Notice")
    assert "| 1 restored |" in selected
    assert output["output"][2]["content"][0]["text"] == "Jane Doe"


async def test_chat_sse_uses_first_content_bearing_choice_for_report_counts(engine_client) -> None:
    handler = _handler(engine_client)
    await handler.handle(response_headers("text/event-stream"))
    events = [
        {"choices": [{"index": 1, "delta": {"content": REVERSIBLE_TOKEN}}]},
        {"choices": [{"index": 0, "delta": {"content": REVERSIBLE_TOKEN}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [{"index": 1, "delta": {}, "finish_reason": "stop"}]},
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    stream += "data: [DONE]\n\n"

    response = await handler.handle(response_body(stream.encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    payloads = _json_sse_payloads(output)
    content: dict[int, str] = {0: "", 1: ""}
    for payload in payloads:
        for choice in payload.get("choices", []):
            content[choice["index"]] += choice.get("delta", {}).get("content", "")
    assert content[0] == "Jane Doe"
    assert content[1].startswith("Jane Doe\n\n---\nPII Engine Notice")
    assert "| 1 restored |" in content[1]
    notice_index = next(
        index
        for index, payload in enumerate(payloads)
        if "PII Engine Notice" in json.dumps(payload)
    )
    target_finish_index = next(
        index
        for index, payload in enumerate(payloads)
        if any(
            choice.get("index") == 1 and choice.get("finish_reason") == "stop"
            for choice in payload.get("choices", [])
        )
    )
    assert notice_index < target_finish_index


async def test_responses_sse_deduplicates_snapshots_and_scopes_multiple_items(
    engine_client,
) -> None:
    handler = _handler(engine_client, api_kind="responses")
    await handler.handle(response_headers("text/event-stream"))
    events = [
        _text_event("response.output_text.delta", "msg_1", 0, 0, "delta", 1),
        _text_event("response.output_text.done", "msg_1", 0, 0, "text", 2),
        {
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": REVERSIBLE_TOKEN},
            "sequence_number": 3,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": _message("msg_1"),
            "sequence_number": 4,
        },
        _text_event("response.output_text.delta", "msg_2", 1, 0, "delta", 5),
        _text_event("response.output_text.done", "msg_2", 1, 0, "text", 6),
        {
            "type": "response.completed",
            "response": {"status": "completed", "output": [_message("msg_1"), _message("msg_2")]},
            "sequence_number": 7,
        },
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    stream += "data: [DONE]\n\n"

    response = await handler.handle(response_body(stream.encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    payloads = _json_sse_payloads(output)
    notice_delta = next(
        payload
        for payload in payloads
        if payload.get("type") == "response.output_text.delta"
        and "PII Engine Notice" in payload.get("delta", "")
    )
    assert "| 1 restored |" in notice_delta["delta"]
    assert "2 restored" not in notice_delta["delta"]
    original_deltas = [
        payload["delta"]
        for payload in payloads
        if payload.get("type") == "response.output_text.delta"
        and "PII Engine Notice" not in payload.get("delta", "")
    ]
    assert original_deltas == ["Jane Doe", "Jane Doe"]
    completed = next(payload for payload in payloads if payload.get("type") == "response.completed")
    first_text = completed["response"]["output"][0]["content"][0]["text"]
    second_text = completed["response"]["output"][1]["content"][0]["text"]
    assert "PII Engine Notice" in first_text
    assert second_text == "Jane Doe"


async def test_responses_sse_supports_done_only_text(engine_client) -> None:
    handler = _handler(engine_client, api_kind="responses")
    await handler.handle(response_headers("text/event-stream"))
    events = [
        _text_event("response.output_text.done", "msg_1", 0, 0, "text", 1),
        {
            "type": "response.completed",
            "response": {"status": "completed", "output": [_message("msg_1")]},
            "sequence_number": 2,
        },
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    response = await handler.handle(response_body(stream.encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    payloads = _json_sse_payloads(output)
    notice = next(
        payload["delta"]
        for payload in payloads
        if payload.get("type") == "response.output_text.delta"
    )
    done = next(
        payload for payload in payloads if payload.get("type") == "response.output_text.done"
    )
    assert "| 1 restored |" in notice
    assert done["text"].startswith("Jane Doe\n\n---\nPII Engine Notice")


async def test_plain_text_rewriter_restores_split_tokens_and_counts_once(engine_client) -> None:
    handler = _handler(engine_client)
    await handler.handle(response_headers("text/plain"))
    split = len(REVERSIBLE_TOKEN) // 2

    first = await handler.handle(
        response_body(f"before {REVERSIBLE_TOKEN[:split]}".encode(), end_of_stream=False)
    )
    final = await handler.handle(
        response_body(f"{REVERSIBLE_TOKEN[split:]} after".encode(), end_of_stream=True)
    )

    assert first is not None
    assert final is not None
    output = (
        first.response_body.response.body_mutation.streamed_response.body
        + final.response_body.response.body_mutation.streamed_response.body
    ).decode()
    assert output.startswith("before Jane Doe after\n\n---\nPII Engine Notice")
    assert REVERSIBLE_TOKEN not in output
    assert "| 1 restored |" in output


@pytest.mark.parametrize("status", ["failed", "incomplete"])
async def test_responses_failure_after_text_done_never_emits_report(
    engine_client, status: str
) -> None:
    handler = _handler(engine_client, api_kind="responses")
    await handler.handle(response_headers("text/event-stream"))
    events = [
        _text_event("response.output_text.delta", "msg_1", 0, 0, "delta", 1),
        _text_event("response.output_text.done", "msg_1", 0, 0, "text", 2),
        {"type": f"response.{status}", "response": {"status": status}, "sequence_number": 3},
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    response = await handler.handle(response_body(stream.encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert "Jane Doe" in output
    assert "PII Engine Notice" not in output
    assert "| Entity |" not in output


async def test_incomplete_responses_json_reverses_but_suppresses_report(engine_client) -> None:
    handler = _handler(engine_client, api_kind="responses")
    await handler.handle(response_headers("application/json"))
    payload = {
        "status": "incomplete",
        "output": [_message("msg_1")],
    }

    response = await handler.handle(response_body(json.dumps(payload).encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert "Jane Doe" in output
    assert "PII Engine Notice" not in output


@pytest.mark.parametrize("api_kind", ["chat", "responses"])
async def test_tool_only_json_reverses_arguments_without_a_body_report(
    engine_client, api_kind: str
) -> None:
    handler = _handler(engine_client, api_kind=api_kind)
    await handler.handle(response_headers("application/json"))
    arguments = json.dumps({"person": REVERSIBLE_TOKEN})
    if api_kind == "chat":
        payload = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": None,
                        "tool_calls": [{"function": {"arguments": arguments}}],
                    },
                }
            ]
        }
    else:
        payload = {
            "status": "completed",
            "output": [{"type": "function_call", "arguments": arguments}],
        }

    response = await handler.handle(response_body(json.dumps(payload).encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert "Jane Doe" in output
    assert "PII Engine Notice" not in output
    assert "| Entity |" not in output


async def test_non_success_json_response_suppresses_detailed_report(engine_client) -> None:
    handler = _handler(engine_client)
    await handler.handle(response_headers("application/json", status=500))
    payload = {"choices": [{"index": 0, "message": {"content": REVERSIBLE_TOKEN}}]}

    response = await handler.handle(response_body(json.dumps(payload).encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert "Jane Doe" in output
    assert "PII Engine Notice" not in output


def _handler(engine_client, *, api_kind: str = "chat") -> StreamHandler:
    handler = StreamHandler(engine_client)
    handler.response_api_kind = api_kind
    handler.reversal_map = {REVERSIBLE_TOKEN: "Jane Doe"}
    handler.notice_messages = ["Protected"]
    handler.request_stats = RequestStats(
        report=PIIReport.model_validate(
            {
                "rows": [
                    {
                        "entity_type": "PERSON",
                        "action": "reversible_replace",
                        "detected_count": 1,
                        "transformed_count": 1,
                        "unique_transformed_count": 1,
                    }
                ],
            }
        ),
        analysis=AnalysisMetadata.model_validate(
            {
                "source": "current_request",
                "scan_performed": True,
                "duration_ms": 3200,
                "overlap_count": 0,
                "overlap_resolution": "strictest_action",
                "policy_version": "test",
                "text_leaf_count": 1,
                "cached_decision_applied": False,
            }
        ),
        decision="apply_actions",
        route_class="general",
    )
    return handler


def _text_event(
    event_type: str,
    item_id: str,
    output_index: int,
    content_index: int,
    field: str,
    sequence: int,
) -> dict[str, object]:
    return {
        "type": event_type,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        field: REVERSIBLE_TOKEN,
        "sequence_number": sequence,
    }


def _message(item_id: str) -> dict[str, object]:
    return {
        "id": item_id,
        "type": "message",
        "content": [{"type": "output_text", "text": REVERSIBLE_TOKEN}],
    }


def _json_sse_payloads(output: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in output.splitlines() if line.startswith("data: {")]
