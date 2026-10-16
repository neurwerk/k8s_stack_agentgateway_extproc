"""Small, complete native protocol fixtures for the v5 response boundary."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .conftest import REVERSIBLE_TOKEN


def reasoning_value(kind):
    if kind == "chat":
        return {
            "reasoning": "Model thought " + REVERSIBLE_TOKEN,
            "reasoning_content": "Alias thought",
            "reasoning_signature": "top-signature",
            "reasoning_details": [
                {
                    "type": "reasoning.text",
                    "text": "Signed " + REVERSIBLE_TOKEN,
                    "signature": "detail-signature",
                    "id": "reasoning-1",
                    "format": "anthropic-claude-v1",
                    "index": 0,
                },
                {"type": "reasoning.summary", "summary": "Readable summary", "index": 1},
                {"type": "reasoning.encrypted", "data": "opaque-detail", "index": 2},
            ],
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Signed thought",
                    "signature": "block-signature",
                    "index": 0,
                },
                {"type": "redacted_thinking", "data": "opaque-block", "index": 1},
            ],
        }
    return {
        "id": "rs-example",
        "type": "reasoning",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "Summary " + REVERSIBLE_TOKEN}],
        "content": [{"type": "reasoning_text", "text": "Reasoned carefully"}],
        "encrypted_content": "opaque-responses-state",
    }


def response_payload(kind, text=REVERSIBLE_TOKEN, *, tools=False, reasoning=False):
    payload = json.loads(
        (Path(__file__).parent / f"fixtures/openai-{kind}-response.json").read_text()
    )
    arguments = json.dumps({"person": text, "count": 1}, separators=(",", ":"))
    if kind == "chat":
        choice = payload["choices"][0]
        choice["message"]["content"] = text
        if tools:
            choice["finish_reason"] = "tool_calls"
            choice["message"]["tool_calls"] = [
                {
                    "id": "call-example",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": arguments},
                }
            ]
    else:
        payload["output"][0]["content"][0]["text"] = text
        if tools:
            payload["output"].append(
                {
                    "id": "fc-example",
                    "type": "function_call",
                    "call_id": "call-example",
                    "name": "lookup",
                    "arguments": arguments,
                    "status": "completed",
                }
            )
    if reasoning:
        if kind == "chat":
            payload["choices"][0]["message"].update(reasoning_value(kind))
        else:
            payload["output"].insert(0, reasoning_value(kind))
    return payload


def stream_events(  # noqa: C901
    kind, text=REVERSIBLE_TOKEN, *, tools=False, reasoning=False, empty_terminal=False
):
    payload = response_payload(kind, text, tools=tools)
    if kind == "chat":
        envelope = {key: value for key, value in payload.items() if key not in {"choices", "usage"}}
        envelope["object"] = "chat.completion.chunk"
        envelope["obfuscation"] = "aB3dE4"

        def chunk(delta, finish=None):
            return {**envelope, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        events = [chunk({"role": "assistant", "content": ""})]
        if reasoning:
            for name, value in reasoning_value(kind).items():
                if isinstance(value, str):
                    events.extend(chunk({name: fragment}) for fragment in (value[:4], value[4:]))
                else:
                    for block in value:
                        # Signatures deliberately arrive after the corresponding text.
                        for field in ("text", "summary", "thinking", "data", "signature"):
                            if field in block:
                                identity = {
                                    key: block[key]
                                    for key in ("type", "index", "id", "format")
                                    if key in block
                                }
                                events.extend(
                                    chunk({name: [{**identity, field: fragment}]})
                                    for fragment in (block[field][:4], block[field][4:])
                                )
        events += [chunk({"content": part}) for part in [text[:12], text[12:]]]
        if tools:
            tool = payload["choices"][0]["message"]["tool_calls"][0]
            arguments = tool["function"]["arguments"]
            events += [
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                **tool,
                                "function": {"name": "lookup", "arguments": arguments[:16]},
                            }
                        ]
                    }
                ),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": arguments[16:]}}]}),
            ]
        return [
            *events,
            chunk({}, "tool_calls" if tools else "stop"),
            {**envelope, "choices": [], "usage": payload["usage"]},
            "[DONE]",
        ]

    message = payload["output"][0]
    part = message["content"][0]
    coordinates = {"item_id": message["id"], "output_index": 0, "content_index": 0}
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "response": {**payload, "status": "in_progress", "output": [], "usage": None},
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**message, "status": "in_progress", "content": []},
        },
        {"type": "response.content_part.added", **coordinates, "part": {**part, "text": ""}},
        *[
            {"type": "response.output_text.delta", **coordinates, "delta": fragment}
            for fragment in [text[:12], text[12:]]
        ],
        {"type": "response.output_text.done", **coordinates, "text": text},
        {"type": "response.content_part.done", **coordinates, "part": part},
        {"type": "response.output_item.done", "output_index": 0, "item": message},
    ]
    if tools:
        tool = payload["output"][1]
        arguments = tool["arguments"]
        coordinates = {"item_id": tool["id"], "output_index": 1}
        events += [
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {**tool, "status": "in_progress", "arguments": ""},
            },
            *[
                {"type": "response.function_call_arguments.delta", **coordinates, "delta": fragment}
                for fragment in [arguments[:16], arguments[16:]]
            ],
            {
                "type": "response.function_call_arguments.done",
                **coordinates,
                "arguments": arguments,
            },
            {"type": "response.output_item.done", "output_index": 1, "item": tool},
        ]
    events.append({"type": "response.completed", "response": payload})
    if reasoning:
        item = reasoning_value(kind)
        for event in events:
            if isinstance(index := event.get("output_index"), int):
                event["output_index"] = index + 1
        payload["output"].insert(0, item)
        coordinates = {"item_id": item["id"], "output_index": 0, "summary_index": 0}
        summary = item["summary"][0]
        reasoning_events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "reasoning", "id": item["id"], "summary": []},
            },
            {
                "type": "response.reasoning_summary_part.added",
                **coordinates,
                "part": {**summary, "text": ""},
            },
            *[
                {"type": "response.reasoning_summary_text.delta", **coordinates, "delta": fragment}
                for fragment in (summary["text"][:15], summary["text"][15:])
            ],
            {
                "type": "response.reasoning_summary_text.done",
                **coordinates,
                "text": summary["text"],
            },
            {"type": "response.reasoning_summary_part.done", **coordinates, "part": summary},
            {
                "type": "response.reasoning_text.delta",
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "delta": item["content"][0]["text"],
            },
            {
                "type": "response.reasoning_text.done",
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "text": item["content"][0]["text"],
            },
            {"type": "response.output_item.done", "output_index": 0, "item": item},
        ]
        events[1:1] = reasoning_events
    if empty_terminal:
        # Native streaming summaries may omit already completed output items.
        events[-1]["response"] = {**payload, "output": []}
    return [
        deepcopy(event)
        | {"sequence_number": index}
        | (
            {"obfuscation": "aB3dE4"}
            if event["type"]
            in ("response.output_text.delta", "response.function_call_arguments.delta")
            else {}
        )
        for index, event in enumerate(events)
    ]


def event_bytes(kind, event):
    if isinstance(event, str):
        return f"data: {event}\n\n".encode()
    prefix = f"event: {event['type']}\n" if kind == "responses" else ""
    return (prefix + f"data: {json.dumps(event, ensure_ascii=False)}\n\n").encode()
