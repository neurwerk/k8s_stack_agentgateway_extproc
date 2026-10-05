"""Security and compatibility regressions for the canonical shared extraction."""

import json

import pytest
from neurwerk_request_segments import (
    CompatibilitySettings,
    ExtractionLimitError,
    TextSegment,
    UnsupportedFeatureError,
    extract_request,
    parse_request,
)
from pydantic import ValidationError

from agentgateway_extproc.config.settings import Settings
from agentgateway_extproc.lib.pipeline.guard import inject_guard_instruction
from agentgateway_extproc.lib.pipeline.request import _converted_request
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.engine import EngineChatRequest, EngineReply

from .conftest import body_request, header_request


@pytest.mark.parametrize("max_tokens", [None, 2_000_001])
def test_chat_roundtrip_preserves_controls_null_false_and_argument_encoding(max_tokens):
    payload = {
        "model": "test",
        "store": False,
        "max_tokens": max_tokens,
        "max_completion_tokens": 2_000_001,
        "parallel_tool_calls": False,
        "seed": 0,
        "reasoning_effort": "low",
        "metadata": {"trace": "opaque"},
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{ "person": "Jane", "nested": ["Doe", 2] }',
                        },
                    }
                ],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Find Jane",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "person": {
                                "type": "string",
                                "description": "Person name",
                                "enum": ["KEEP"],
                            }
                        },
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }
    extracted = extract_request(parse_request(payload))
    assert [s.text for s in extracted.segments] == ["Jane", "Doe", "Find Jane", "Person name"]
    assert extracted.diagnostic_path("s1") == (
        "messages",
        0,
        "tool_calls",
        0,
        "function",
        "arguments",
        "nested",
        0,
    )
    assert extracted.rebuild(extracted.segments).model_dump(exclude_unset=True) == payload
    changed = extracted.rebuild([TextSegment(id=s.id, text="MASK") for s in extracted.segments])
    wire = changed.model_dump(exclude_unset=True)
    assert wire["store"] is False and wire["max_tokens"] == max_tokens
    assert "stream" not in wire and "name" not in wire["messages"][0]
    assert json.loads(wire["messages"][0]["tool_calls"][0]["function"]["arguments"]) == {
        "person": "MASK",
        "nested": ["MASK", 2],
    }
    assert wire["tools"][0]["function"]["parameters"]["properties"]["person"]["enum"] == ["KEEP"]
    assert extracted.control_shape() == extract_request(changed).control_shape()


def test_responses_flat_tools_and_short_messages():
    payload = {
        "model": "test",
        "store": False,
        "max_output_tokens": 2_000_001,
        "input": [{"role": "user", "content": "Jane"}],
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "strict": None,
                "description": "Find Jane",
                "parameters": {"type": "object"},
            }
        ],
    }
    extracted = extract_request(parse_request(payload, kind="responses"))
    assert [s.text for s in extracted.segments] == ["Jane", "Find Jane"]
    assert extracted.rebuild(extracted.segments).model_dump(exclude_unset=True) == payload
    with pytest.raises((ValidationError, UnsupportedFeatureError)):
        parse_request(
            {**payload, "tools": [{"type": "function", "function": {"name": "lookup"}}]},
            kind="responses",
        )
    with pytest.raises((ValidationError, UnsupportedFeatureError)):
        parse_request(payload, kind="chat")


def test_mcp_analyzes_only_argument_values_and_preserves_metadata():
    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "lookup",
            "arguments": {"Jane": ["Doe", {"query": "mail"}]},
            "_meta": {"token": "KEEP"},
        },
    }
    extracted = extract_request(parse_request(payload))
    assert [s.text for s in extracted.segments] == ["Doe", "mail"]
    assert extracted.rebuild(extracted.segments).model_dump(exclude_unset=True) == payload
    omitted = {**payload, "params": {"name": "lookup"}}
    mcp = parse_request(omitted)
    assert mcp.model_dump(by_alias=True) == omitted
    fields = mcp.model_json_schema()["$defs"]["EngineMcpParams"]["properties"]
    assert all(
        fields[name]["type"] == "object" and "default" not in fields[name]
        for name in ("arguments", "_meta")
    )


@pytest.mark.parametrize(
    "replacement",
    [
        [],
        [TextSegment(id="s9", text="x")],
        [TextSegment(id="s0", text="x"), TextSegment(id="s0", text="y")],
    ],
)
def test_rebuild_rejects_missing_extra_and_wrong_id_segments(replacement):
    extracted = extract_request(parse_request({"model": "test", "input": "Jane"}))
    with pytest.raises(ValueError, match="identities"):
        extracted.rebuild(replacement)


async def test_reviewed_controls_are_scoped_typed_and_preserved_through_rebuild(
    engine_reply, engine_client
):
    controls = CompatibilitySettings.model_validate(
        {
            "controls": [
                {
                    "endpoint": "chat",
                    "location": "request",
                    "field": "vendor_mode",
                    "kinds": ["string"],
                    "enum": ["fast", "safe"],
                },
                {
                    "endpoint": "chat",
                    "location": "message",
                    "field": "vendor_flag",
                    "kinds": ["boolean"],
                },
            ]
        }
    )
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "Jane"}],
        "vendor_mode": "safe",
    }
    with pytest.raises(UnsupportedFeatureError):
        parse_request(payload)
    request = parse_request(payload, controls=controls)
    extracted = extract_request(request)
    assert (
        extracted.rebuild([TextSegment(id="s0", text="MASK")]).model_dump(exclude_unset=True)[
            "vendor_mode"
        ]
        == "safe"
    )
    with pytest.raises(ValidationError):
        parse_request({**payload, "vendor_mode": {"text": "Jane"}}, controls=controls)
    with pytest.raises(UnsupportedFeatureError):
        parse_request({"model": "test", "input": "Jane", "vendor_mode": "safe"}, controls=controls)
    document = {
        **payload,
        "max_tokens": None,
        "messages": [
            {
                "role": "user",
                "vendor_flag": False,
                "content": [{"type": "file", "file": {"file_data": "opaque"}}],
            }
        ],
    }
    original = parse_request(document, controls=controls)
    converted, _ = _converted_request(document, ["Jane"], {}, controls=original._controls)
    inspected = extract_request(converted)
    reconstructed = inspected.rebuild(inspected.segments)
    reply = EngineReply.model_validate({**engine_reply, "request": reconstructed})
    assert isinstance(reply.request, EngineChatRequest)
    guarded = inject_guard_instruction(reply.request)
    wire = guarded.model_dump(exclude_unset=True)
    assert wire["vendor_mode"] == "safe" and wire["max_tokens"] is None
    assert wire["messages"][1]["vendor_flag"] is False
    assert "stream" not in wire and "tools" not in wire
    handler = StreamHandler(engine_client, Settings(compatibility=controls))
    await handler.handle(header_request())
    response = await handler.handle(
        body_request(
            json.dumps(
                {
                    **payload,
                    "messages": [{"role": "user", "content": "Jane Doe", "vendor_flag": False}],
                }
            ).encode()
        )
    )
    assert response is not None and response.HasField("request_body")
    forwarded = json.loads(response.request_body.response.body_mutation.body)
    assert forwarded["vendor_mode"] == "safe"
    assert forwarded["messages"][1]["vendor_flag"] is False


@pytest.mark.parametrize(
    "rule",
    [
        {"field": name, "kinds": ["string"], "enum": ["fixed"]}
        for name in [
            "messages",
            "parameters",
            "content",
            "tools",
            "prediction",
            "reasoning_content",
            "store",
            "previous_response_id",
        ]
    ]
    + [
        {"field": "vendor_mode", "kinds": ["string"]},
        {"field": "vendor_mode", "kinds": ["number"], "enum": [float("inf")]},
        {"field": "vendor_mode", "kinds": ["boolean", "boolean"]},
    ],
)
def test_controls_cannot_override_content_structure_or_known_controls(rule):
    with pytest.raises(ValidationError):
        CompatibilitySettings.model_validate({"controls": [{"endpoint": "chat", **rule}]})


@pytest.mark.parametrize(
    "arguments",
    [
        '{"x":"Jane","x":"Doe"}',
        '{"x":NaN}',
        '{"x":1e999}',
        "not JSON",
        "[" * 2_000 + '"text"' + "]" * 2_000,
    ],
    ids=["duplicate-key", "nan", "infinity", "malformed", "deep-encoded-json"],
)
def test_encoded_arguments_fail_closed_without_collapsing_content(arguments):
    request = parse_request(
        {
            "model": "test",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": arguments,
                }
            ],
        }
    )
    with pytest.raises(ValueError) as rejected:
        extract_request(request)
    if arguments.startswith("[["):
        assert isinstance(rejected.value, ExtractionLimitError)
        assert (rejected.value.measured, rejected.value.maximum, rejected.value.exact) == (
            33,
            32,
            False,
        )
        with pytest.raises(ExtractionLimitError) as configured:
            extract_request(request, max_depth=4)
        assert (configured.value.measured, configured.value.maximum) == (5, 4)
        escaped = '\\"' + "[" * 100
        safe = parse_request(
            {
                "model": "test",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": json.dumps({"text": escaped}),
                    }
                ],
            }
        )
        assert extract_request(safe, max_depth=1).segments[0].text == escaped


def test_nested_schema_prose_not_identifiers_or_enums():
    schema = {
        "$defs": {"Jane": {"enum": ["Doe"], "description": "scan"}},
        "if": {"properties": {"name": {"title": "scan too"}}},
        "x-extension": {"description": "opaque"},
    }
    request = parse_request(
        {
            "model": "test",
            "input": "hello",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "output",
                    "schema": schema,
                    "strict": False,
                }
            },
        }
    )
    assert [s.text for s in extract_request(request).segments] == ["hello", "scan", "scan too"]

    for control in (
        {"tool_choice": {"type": "function", "function": {"name": "lookup", "prose": "hidden"}}},
        {"response_format": {"type": "json_object", "prose": "hidden"}},
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema, "prose": "hidden"},
            }
        },
    ):
        with pytest.raises(UnsupportedFeatureError):
            parse_request(
                {"model": "test", "messages": [{"role": "user", "content": "hello"}], **control}
            )
    with pytest.raises(UnsupportedFeatureError):
        parse_request({"model": "test", "input": "hello", "previous_response_id": "resp_1"})
    nullable = parse_request({"model": "test", "input": "hello", "previous_response_id": None})
    assert nullable.model_dump(exclude_unset=True)["previous_response_id"] is None
