"""Test ext_proc request and response transport behavior."""

from __future__ import annotations

import gzip
import json
from unittest.mock import patch

import httpx
import pytest

from agentgateway_extproc.config.settings import EngineSettings, Settings
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.pipeline.guard import GUARD_INSTRUCTION
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.exceptions import InvalidEngineReplyError, InvalidReversalError

from .conftest import (
    REVERSIBLE_TOKEN,
    body_request,
    header_request,
    mcp_headers,
    mcp_policy,
    request_json,
    response_body,
    response_headers,
)

SPECIAL_PLAINTEXT = 'quoted "path\\file" café\nnext line'
MAX_REQUEST_BYTES = 5_242_880
MAX_RESPONSE_BYTES = 10_485_760


def _request_body_of_size(size: int) -> bytes:
    prefix = b'{"model":"test","messages":[{"role":"user","content":"'
    suffix = b'"}]}'
    return prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix


async def test_request_mutation_and_stream_reversal(engine_client) -> None:
    """Engine request mutations are forwarded and response placeholders reverse."""
    handler = StreamHandler(engine_client)
    headers = await handler.handle(
        header_request({"x-route-class": "spoofed", "x-request-id": "safe"})
    )
    assert headers is not None
    assert "x-route-class" in list(headers.request_headers.response.header_mutation.remove_headers)
    request = await handler.handle(body_request(request_json()))
    assert request is not None
    assert request.request_body.response.body_mutation.body.startswith(b'{"model":"test"')
    forwarded = json.loads(request.request_body.response.body_mutation.body)
    assert forwarded["messages"][0] == {
        "role": "system",
        "content": GUARD_INSTRUCTION,
    }
    assert forwarded["messages"][1]["content"] == REVERSIBLE_TOKEN
    assert "Request protected" not in json.dumps(forwarded)
    assert "PII scan completed" not in json.dumps(forwarded)
    trusted = {
        item.header.key: item.header.value
        for item in request.request_body.response.header_mutation.set_headers
    }
    assert trusted["x-remote-allowed"] == "true"
    assert trusted["x-pii-entities"] == "PERSON"
    assert "x-route-class" not in trusted
    assert not handler.request_body_chunks
    assert not handler.request_headers
    assert handler.request_stats is not None
    assert handler.request_stats.analysis.source == "current_request"

    await handler.handle(response_headers("application/json"))
    response = await handler.handle(
        response_body(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    f"{GUARD_INSTRUCTION}{REVERSIBLE_TOKEN}{GUARD_INSTRUCTION}"
                                )
                            }
                        }
                    ]
                }
            ).encode()
        )
    )
    assert response is not None
    payload = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    assert "Jane Doe" in payload["choices"][0]["message"]["content"]
    assert GUARD_INSTRUCTION not in payload["choices"][0]["message"]["content"]
    assert "PII Engine Notice" in payload["choices"][0]["message"]["content"]


async def test_provider_request_omits_empty_tool_calls_from_chat_history(
    engine_client, engine_reply
) -> None:
    """Strict providers never receive empty tool-call arrays after analysis."""
    original = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "next"},
        ],
    }
    engine_reply.update(
        {
            "decision": "pass",
            "entities": [],
            "entity_counts": {},
            "applied_actions": [],
            "request": {
                "model": "test",
                "messages": [
                    {"role": "user", "content": "first", "tool_calls": []},
                    {"role": "assistant", "content": "reply", "tool_calls": []},
                    {"role": "user", "content": "next", "tool_calls": []},
                ],
            },
            "notices": {"request": [], "response": []},
            "report": {"rows": []},
            "reversal": {},
        }
    )
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())

    response = await handler.handle(body_request(json.dumps(original).encode()))

    assert response is not None and not response.HasField("immediate_response")
    forwarded = json.loads(response.request_body.response.body_mutation.body)
    assert [message["role"] for message in forwarded["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert all("tool_calls" not in message for message in forwarded["messages"])


@pytest.mark.parametrize("include_usage", [True, False])
async def test_chat_stream_options_survives_engine_and_provider_dispatch(
    engine_reply, include_usage: bool
) -> None:
    original = {
        "model": "test",
        "messages": [{"role": "user", "content": "Jane Doe"}],
        "stream": True,
        "stream_options": {"include_usage": include_usage},
    }
    engine_reply["request"].update(
        {"stream": True, "stream_options": {"include_usage": include_usage}}
    )
    captured: dict[str, object] = {}

    def transport(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=engine_reply, request=request)

    client = EngineClient(
        EngineSettings(base_url="https://pii-engine.test"),
        httpx.AsyncClient(
            transport=httpx.MockTransport(transport), base_url="https://pii-engine.test"
        ),
    )
    handler = StreamHandler(client)
    await handler.handle(header_request())

    response = await handler.handle(body_request(json.dumps(original).encode()))

    assert captured["stream_options"] == {"include_usage": include_usage}
    assert response is not None and not response.HasField("immediate_response")
    forwarded = json.loads(response.request_body.response.body_mutation.body)
    assert forwarded["stream_options"] == {"include_usage": include_usage}
    assert forwarded["messages"][0]["content"] == GUARD_INSTRUCTION


async def test_model_validation_log_contains_only_bounded_safe_metadata(
    engine_client, caplog
) -> None:
    rejected_name = "unknown_private_identifier_"
    rejected_value = "secret-payload-value"
    request_identifier = "request-sensitive-123"
    base = {
        "model": "test",
        "messages": [{"role": "user", "content": "private prompt payload"}],
        "user": request_identifier,
    }
    cases = [
        ({**base, rejected_name: rejected_value}, "top_level", 1),
        (
            {
                **base,
                "stream_options": {
                    "include_usage": True,
                    rejected_name: rejected_value,
                },
            },
            "stream_options",
            1,
        ),
        (
            {
                **base,
                "messages": [
                    {
                        **base["messages"][0],
                        **{f"{rejected_name}{index}": rejected_value for index in range(110)},
                    }
                ],
            },
            "messages",
            100,
        ),
    ]

    for payload, scope, count in cases:
        caplog.clear()
        handler = StreamHandler(engine_client)
        await handler.handle(header_request())

        response = await handler.handle(body_request(json.dumps(payload).encode()))

        assert response is not None and response.immediate_response.status.code == 400
        assert "protocol_failure" in handler._dispatch_outcomes
        assert caplog.messages == [
            "model request validation failed family=chat reason=extra_forbidden "
            f"scope={scope} count={count}"
        ]
        for private_value in (
            rejected_name,
            rejected_value,
            request_identifier,
            "private prompt payload",
        ):
            assert private_value not in caplog.text


async def test_headerless_identical_requests_use_distinct_engine_sessions(engine_client) -> None:
    """Separate requests cannot become linkable through identical prompt content."""
    first = StreamHandler(engine_client)
    second = StreamHandler(engine_client)
    with patch.object(
        engine_client,
        "analyze_request",
        wraps=engine_client.analyze_request,
    ) as analyze:
        await first.handle(header_request())
        await first.handle(body_request(request_json()))
        await second.handle(header_request())
        await second.handle(body_request(request_json()))

    session_keys = [call.args[1] for call in analyze.await_args_list]
    assert len(session_keys) == 2
    assert session_keys[0] != session_keys[1]


async def test_chunked_json_is_withheld_until_it_can_be_rewritten(engine_client) -> None:
    """Arbitrary JSON chunk boundaries preserve reversal and notice injection."""
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))
    await handler.handle(response_headers("application/json"))
    body = json.dumps({"choices": [{"message": {"content": REVERSIBLE_TOKEN}}]}).encode()
    first = await handler.handle(response_body(body[:20], end_of_stream=False))
    assert first is None
    final = await handler.handle(response_body(body[20:]))
    assert final is not None
    payload = json.loads(final.response_body.response.body_mutation.streamed_response.body)
    assert payload["choices"][0]["message"]["content"].startswith("Jane Doe")


async def test_json_reversal_preserves_json_escaping(engine_client) -> None:
    """Plaintext is inserted into decoded JSON values and serialized safely."""
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))
    handler.reversal_map = {REVERSIBLE_TOKEN: 'A "quoted" \\ value\nnext line'}
    handler.response_notice_allowed = False
    await handler.handle(response_headers("application/json"))
    body = json.dumps(
        {"choices": [{"index": 0, "message": {"content": REVERSIBLE_TOKEN}}]}
    ).encode()
    response = await handler.handle(response_body(body))
    assert response is not None
    payload = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    assert payload["choices"][0]["message"]["content"] == 'A "quoted" \\ value\nnext line'


async def test_buffered_json_rejects_placeholders_in_protocol_metadata(engine_client) -> None:
    """Authorized reversal material cannot escape through provider-controlled metadata."""
    handler = StreamHandler(engine_client)
    handler.reversal_map = {REVERSIBLE_TOKEN: "Jane Doe"}
    await handler.handle(response_headers("application/json"))
    body = json.dumps(
        {"id": REVERSIBLE_TOKEN, "choices": [{"message": {"content": "answer"}}]}
    ).encode()

    with pytest.raises(InvalidReversalError, match="protocol field"):
        await handler.handle(response_body(body))


async def test_plain_response_strips_a_guard_split_across_body_chunks(engine_client) -> None:
    """Plain output removes the protocol instruction before reversing placeholders."""
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))
    handler.response_notice_allowed = False
    await handler.handle(response_headers("text/plain"))
    split = len(GUARD_INSTRUCTION) // 2

    first = await handler.handle(
        response_body(GUARD_INSTRUCTION[:split].encode(), end_of_stream=False)
    )
    final = await handler.handle(
        response_body(f"{GUARD_INSTRUCTION[split:]}{REVERSIBLE_TOKEN}".encode())
    )

    assert first is not None
    assert final is not None
    output = (
        first.response_body.response.body_mutation.streamed_response.body
        + final.response_body.response.body_mutation.streamed_response.body
    ).decode()
    assert output == "Jane Doe"


async def test_chat_sse_strips_a_guard_split_across_provider_events(engine_client) -> None:
    """Semantic channel state removes an echo spanning separate Chat deltas."""
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))
    handler.response_notice_allowed = False
    await handler.handle(response_headers("text/event-stream"))
    split = len(GUARD_INSTRUCTION) // 2
    events = [
        {"choices": [{"index": 0, "delta": {"content": GUARD_INSTRUCTION[:split]}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": GUARD_INSTRUCTION[split:] + REVERSIBLE_TOKEN},
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"

    response = await handler.handle(response_body(body.encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    payloads = [_sse_data(line) for line in output.splitlines() if line.startswith("data: {")]
    content = "".join(
        choice["delta"].get("content", "") for payload in payloads for choice in payload["choices"]
    )
    assert content == "Jane Doe"
    assert GUARD_INSTRUCTION not in output


async def test_json_error_response_skips_notice_injection(engine_client) -> None:
    """Upstream errors remain visible when they have no model-output notice target."""
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))
    await handler.handle(response_headers("application/json", status=401))
    response = await handler.handle(response_body(b'{"error":{"message":"unauthorized"}}'))
    assert response is not None
    payload = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    assert payload == {"error": {"message": "unauthorized"}}


async def test_sse_reversal_can_span_provider_events(engine_client) -> None:
    """Only the candidate suffix is retained when one token spans SSE events."""
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))
    handler.reversal_map = {REVERSIBLE_TOKEN: 'A "quoted" value'}
    handler.response_notice_allowed = False
    await handler.handle(response_headers("text/event-stream"))
    split = len(REVERSIBLE_TOKEN) // 2
    first_event = json.dumps(
        {"choices": [{"index": 0, "delta": {"content": REVERSIBLE_TOKEN[:split]}}]}
    )
    second_event = json.dumps(
        {"choices": [{"index": 0, "delta": {"content": REVERSIBLE_TOKEN[split:]}}]}
    )
    first = await handler.handle(
        response_body(f"data: {first_event}\n\n".encode(), end_of_stream=False)
    )
    second = await handler.handle(
        response_body(f"data: {second_event}\n\ndata: [DONE]\n\n".encode())
    )
    assert first is not None
    assert second is not None
    output = (
        first.response_body.response.body_mutation.streamed_response.body
        + second.response_body.response.body_mutation.streamed_response.body
    ).decode()
    values = [json.loads(line[6:]) for line in output.splitlines() if line.startswith("data: {")]
    content = "".join(item["choices"][0]["delta"].get("content", "") for item in values)
    assert content == 'A "quoted" value'


async def test_chat_sse_marks_a_model_modified_placeholder_and_continues(engine_client) -> None:
    """Ordinary model text degrades safely instead of aborting an active chat stream."""
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))
    handler.reversal_map = {REVERSIBLE_TOKEN: "Jane Doe"}
    handler.response_notice_allowed = False
    await handler.handle(response_headers("text/event-stream"))
    altered = "<REV_PERSON_0123456789abcdef_...>"
    events = [
        {"choices": [{"index": 0, "delta": {"content": f"before {altered} after"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"

    response = await handler.handle(response_body(body.encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    payloads = [_sse_data(line) for line in output.splitlines() if line.startswith("data: {")]
    content = "".join(
        choice["delta"].get("content", "") for payload in payloads for choice in payload["choices"]
    )
    assert content.startswith("before <PII_INVALID_PERSON_")
    assert content.endswith("> after")
    assert altered not in content
    assert handler.reversal_misses == 1


async def test_chat_sse_emits_a_marker_for_an_incomplete_placeholder_at_eof(engine_client) -> None:
    handler = StreamHandler(engine_client)
    handler.reversal_map = {REVERSIBLE_TOKEN: "Jane Doe"}
    handler.response_notice_allowed = False
    await handler.handle(response_headers("text/event-stream"))
    events = [
        {"choices": [{"index": 0, "delta": {"content": "before <REV_PERSON_broken"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"

    response = await handler.handle(response_body(body.encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    payloads = [_sse_data(line) for line in output.splitlines() if line.startswith("data: {")]
    content = "".join(
        choice["delta"].get("content", "") for payload in payloads for choice in payload["choices"]
    )
    assert content.startswith("before <PII_INVALID_PERSON_")
    assert content.endswith(">")
    assert output.rstrip().endswith("data: [DONE]")


async def test_streaming_utf8_code_point_may_span_response_chunks(engine_client) -> None:
    """A multibyte code point split across body messages remains valid UTF-8."""
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))
    await handler.handle(response_headers("text/event-stream"))
    stream = "data: café\n\ndata: [DONE]\n\n".encode()
    split = stream.index(b"\xc3") + 1
    first = await handler.handle(response_body(stream[:split], end_of_stream=False))
    final = await handler.handle(response_body(stream[split:]))
    assert first is not None
    assert final is not None
    combined = (
        first.response_body.response.body_mutation.streamed_response.body
        + final.response_body.response.body_mutation.streamed_response.body
    )
    assert "café" in combined.decode()


async def test_sse_and_gzip_are_processed(engine_client) -> None:
    """Gzip SSE remains valid and notice precedes the terminal marker."""
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))
    await handler.handle(response_headers("text/event-stream", "gzip"))
    compressed = gzip.compress(b"data: hello\n\ndata: [DONE]\n\n")
    response = await handler.handle(response_body(compressed))
    assert response is not None
    text = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert text.index("PII Engine Notice") < text.index("data: [DONE]")
    assert not handler.reversal_map
    assert not handler.notice_messages


@pytest.mark.parametrize(
    "mutate",
    [
        lambda encoded: encoded[:-1],
        lambda encoded: encoded + b"trailing",
        lambda encoded: encoded + gzip.compress(b"data: second\n\n"),
    ],
    ids=["truncated", "trailing-data", "concatenated-member"],
)
async def test_gzip_requires_one_complete_member(engine_client, mutate) -> None:
    handler = StreamHandler(engine_client)
    await handler.handle(response_headers("text/event-stream", "gzip"))
    encoded = mutate(gzip.compress(b"data: hello\n\ndata: [DONE]\n\n"))
    with pytest.raises(ValueError, match="gzip"):
        await handler.handle(response_body(encoded))


async def test_gzip_decoding_is_bounded_before_allocation(engine_client) -> None:
    """Highly compressed output cannot expand beyond the configured decoded limit."""
    handler = StreamHandler(engine_client, Settings(max_response_bytes=1_024))
    await handler.handle(response_headers("text/event-stream", "gzip"))
    compressed = gzip.compress(b"x" * 1_025)
    try:
        await handler.handle(response_body(compressed))
    except ValueError as error:
        assert str(error) == "response body too large"
    else:
        raise AssertionError("oversized gzip response was accepted")


@pytest.mark.parametrize(
    "metadata_line",
    [f": {REVERSIBLE_TOKEN}", f"x-extension: {REVERSIBLE_TOKEN}"],
)
async def test_sse_metadata_placeholders_fail_closed(engine_client, metadata_line: str) -> None:
    handler = StreamHandler(engine_client)
    handler.reversal_map = {REVERSIBLE_TOKEN: "Jane Doe"}
    await handler.handle(response_headers("text/event-stream"))
    with pytest.raises(InvalidReversalError, match="SSE metadata"):
        await handler.handle(response_body(f"{metadata_line}\n\n".encode()))


async def test_sse_json_protocol_metadata_placeholders_fail_closed(engine_client) -> None:
    handler = StreamHandler(engine_client)
    handler.reversal_map = {REVERSIBLE_TOKEN: "Jane Doe"}
    await handler.handle(response_headers("text/event-stream"))
    event = {
        "id": REVERSIBLE_TOKEN,
        "choices": [{"index": 0, "delta": {"content": "answer"}}],
    }

    with pytest.raises(InvalidReversalError, match="protocol field"):
        await handler.handle(response_body(f"data: {json.dumps(event)}\n\n".encode()))


async def test_request_accepts_exact_five_mibibytes_in_buffered_phase(
    engine_client, engine_reply
) -> None:
    body = _request_body_of_size(MAX_REQUEST_BYTES)
    parsed = json.loads(body)
    engine_reply.update(
        {
            "decision": "pass",
            "entities": [],
            "entity_counts": {},
            "applied_actions": [],
            "request": parsed,
            "notices": {"request": [], "response": []},
            "report": {"rows": []},
            "reversal": {},
        }
    )
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())

    response = await handler.handle(body_request(body, end_of_stream=False))

    assert response is not None
    assert response.HasField("request_body")
    assert not response.HasField("immediate_response")


async def test_request_rejects_five_mibibytes_plus_one_in_buffered_phase(
    engine_client,
) -> None:
    body = _request_body_of_size(MAX_REQUEST_BYTES + 1)
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    response = await handler.handle(body_request(body, end_of_stream=False))

    assert response is not None
    assert response.immediate_response.status.code == 413


async def test_provider_input_accepts_exact_ten_mibibytes_across_chunks(engine_client) -> None:
    body = b'{"choices":[]}' + b" " * (MAX_RESPONSE_BYTES - len(b'{"choices":[]}'))
    handler = StreamHandler(engine_client)
    handler.notice_messages = []
    await handler.handle(response_headers("application/json"))

    first = await handler.handle(response_body(body[:6_000_000], end_of_stream=False))
    final = await handler.handle(response_body(body[6_000_000:]))

    assert first is None
    assert final is not None
    assert final.response_body.response.body_mutation.streamed_response.body == b'{"choices":[]}'


async def test_provider_input_rejects_ten_mibibytes_plus_one_across_chunks(
    engine_client,
) -> None:
    body = b"x" * (MAX_RESPONSE_BYTES + 1)
    handler = StreamHandler(engine_client)
    await handler.handle(response_headers("text/plain"))
    await handler.handle(response_body(body[:6_000_000], end_of_stream=False))

    with pytest.raises(ValueError, match="response body too large"):
        await handler.handle(response_body(body[6_000_000:]))


async def test_transformed_request_body_is_bounded_after_serialization(
    engine_client, engine_reply
) -> None:
    engine_reply.update(
        {
            "applied_actions": ["replace"],
            "request": {
                "model": "test",
                "messages": [{"role": "user", "content": "x" * 1_024}],
            },
            "notices": {"request": [], "response": []},
            "report": {
                "rows": [
                    {
                        "entity_type": "PERSON",
                        "action": "replace",
                        "detected_count": 1,
                        "transformed_count": 1,
                        "unique_transformed_count": 1,
                    }
                ]
            },
            "reversal": {},
        }
    )
    handler = StreamHandler(engine_client, Settings(max_transformed_request_bytes=1_024))
    await handler.handle(header_request())

    with pytest.raises(ValueError, match="transformed request body too large"):
        await handler.handle(body_request(request_json()))


async def test_engine_cannot_mutate_request_protocol_controls(engine_client, engine_reply) -> None:
    """A schema-valid reply cannot switch the requested model or request family."""
    engine_reply["request"]["model"] = "other-model"
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    with pytest.raises(InvalidEngineReplyError, match="protocol controls"):
        await handler.handle(body_request(request_json()))


async def test_reroute_reply_sets_only_trusted_routing_headers(engine_client, engine_reply) -> None:
    """A validated engine reroute becomes the sole local-routing instruction."""
    engine_reply.update(
        {
            "decision": "reroute",
            "remote_allowed": False,
            "route_class": "local-sensitive",
            "applied_actions": ["reroute"],
            "request": {
                "model": "test",
                "messages": [{"role": "user", "content": "Jane Doe"}],
            },
            "report": {
                "rows": [
                    {
                        "entity_type": "PERSON",
                        "action": "reroute",
                        "detected_count": 1,
                        "transformed_count": 0,
                        "unique_transformed_count": 0,
                    }
                ],
            },
            "reversal": {},
        }
    )
    handler = StreamHandler(engine_client)
    await handler.handle(header_request({"x-route-class": "client-spoof"}))
    response = await handler.handle(body_request(request_json()))
    assert response is not None
    trusted = {
        item.header.key: item.header.value
        for item in response.request_body.response.header_mutation.set_headers
    }
    assert trusted == {
        "content-length": str(len(response.request_body.response.body_mutation.body)),
        "x-pii-entities": "PERSON",
        "x-remote-allowed": "false",
        "x-route-class": "local-sensitive",
    }


async def test_block_reply_is_a_generic_403_without_a_report(engine_client, engine_reply) -> None:
    """Blocked requests terminate before any detailed report can reach the body."""
    engine_reply.update(
        {
            "decision": "block",
            "applied_actions": ["block"],
            "remote_allowed": False,
            "route_class": None,
            "request": None,
            "report": {
                "rows": [
                    {
                        "entity_type": "PERSON",
                        "action": "block",
                        "detected_count": 1,
                        "transformed_count": 0,
                        "unique_transformed_count": 0,
                    }
                ],
            },
            "reversal": {},
        }
    )
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())

    response = await handler.handle(body_request(request_json()))

    assert response is not None
    assert response.immediate_response.status.code == 403
    assert response.immediate_response.body == '{"error": "request blocked by policy"}'
    assert "PII Engine Notice" not in response.immediate_response.body
    assert "| Entity |" not in response.immediate_response.body


async def test_unscanned_current_safety_block_remains_fail_closed(
    engine_client, engine_reply
) -> None:
    engine_reply.update(
        {
            "decision": "block",
            "entities": [],
            "entity_counts": {},
            "applied_actions": ["block"],
            "remote_allowed": False,
            "route_class": None,
            "request": None,
            "report": {"rows": []},
            "safety_rule": "promptInjection",
            "reversal": {},
        }
    )
    engine_reply["analysis"].update({"scan_performed": False, "duration_ms": None})
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())

    response = await handler.handle(body_request(request_json()))

    assert response is not None
    assert response.immediate_response.status.code == 403


async def test_tool_only_model_request_receives_the_fixed_guard(
    engine_client, engine_reply
) -> None:
    """A PII-enabled tool-only model request still receives the fixed instruction."""
    tool_request = {
        "model": "test",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"id": 1}},
                    }
                ],
            }
        ],
    }
    engine_reply.update(
        {
            "decision": "pass",
            "entities": [],
            "entity_counts": {},
            "applied_actions": [],
            "request": tool_request,
            "report": {"rows": []},
            "reversal": {},
        }
    )
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())

    response = await handler.handle(body_request(json.dumps(tool_request).encode()))

    assert response is not None
    assert not response.HasField("immediate_response")
    forwarded = json.loads(response.request_body.response.body_mutation.body)
    assert forwarded["messages"][0]["content"] == GUARD_INSTRUCTION
    assert "content" not in forwarded["messages"][1]
    assert forwarded["messages"][1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert "Request protected" not in json.dumps(forwarded)


async def test_assistant_reasoning_bypasses_analysis_and_replays_unchanged(
    engine_client, engine_reply
) -> None:
    """Opaque assistant reasoning survives strict engine processing and guard insertion."""
    signature = {
        "provider": "opaque",
        "parts": [None, REVERSIBLE_TOKEN, GUARD_INSTRUCTION],
    }
    original = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "Jane Doe"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": None,
                "reasoning_signature": signature,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"id": 1}},
                    }
                ],
            },
            {"role": "user", "content": "continue"},
            {
                "role": "assistant",
                "content": "safe",
                "reasoning_content": {"parts": [REVERSIBLE_TOKEN, None]},
                "reasoning_signature": "second-opaque-signature",
            },
        ],
    }
    engine_reply["request"] = {
        "model": "test",
        "messages": [
            {"role": "user", "content": REVERSIBLE_TOKEN},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"id": 1}},
                    }
                ],
            },
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "safe"},
        ],
    }
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())

    response = await handler.handle(body_request(json.dumps(original).encode()))

    assert response is not None and not response.HasField("immediate_response")
    forwarded = json.loads(response.request_body.response.body_mutation.body)
    assert forwarded["messages"][0]["content"] == GUARD_INSTRUCTION
    assistant = forwarded["messages"][2]
    assert "content" not in assistant
    assert "reasoning_content" in assistant and assistant["reasoning_content"] is None
    assert assistant["reasoning_signature"] == signature
    second_assistant = forwarded["messages"][4]
    assert second_assistant["reasoning_content"] == {"parts": [REVERSIBLE_TOKEN, None]}
    assert second_assistant["reasoning_signature"] == "second-opaque-signature"


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "safe", "reasoning_content": "not allowed"},
        {"role": "assistant", "content": "safe", "reasoning": "not supported"},
    ],
    ids=["non-assistant", "unsupported-field"],
)
async def test_request_reasoning_contract_rejects_unsupported_locations(
    engine_client, message
) -> None:
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    body = json.dumps({"model": "test", "messages": [message]}).encode()

    response = await handler.handle(body_request(body))

    assert response is not None
    assert response.immediate_response.status.code == 400
    assert response.immediate_response.body == '{"error":"invalid model request"}'


async def test_mcp_mutates_only_arguments_and_reverses_only_result_text(
    engine_client, engine_reply
) -> None:
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
    policy = mcp_policy()
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(mcp_headers(), policy=policy))

    request = await handler.handle(body_request(json.dumps(original).encode(), policy=policy))

    assert request is not None
    forwarded = json.loads(request.request_body.response.body_mutation.body)
    assert forwarded["params"]["arguments"]["query"] == REVERSIBLE_TOKEN
    assert "Request protected" not in json.dumps(forwarded)
    assert handler.response_notice_allowed is False
    assert handler.notice_messages == []
    assert handler.presidio_code is None

    await handler.handle(response_headers("application/json", policy=policy))
    response = await handler.handle(
        response_body(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"{GUARD_INSTRUCTION}{REVERSIBLE_TOKEN}",
                            }
                        ]
                    },
                }
            ).encode(),
            policy=policy,
        )
    )
    assert response is not None
    payload = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    assert payload["result"]["content"][0]["text"] == f"{GUARD_INSTRUCTION}Jane Doe"
    assert "PII Engine Notice" not in json.dumps(payload)
    headers = handler.pop_pending_response_headers()
    assert headers is not None
    assert "x-presidio-code" not in _set_headers(headers)


async def test_reversal_plaintext_must_match_the_same_original_leaf(
    engine_client, engine_reply
) -> None:
    """A valid placeholder cannot recover unrelated plaintext from another leaf."""
    original = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "Jane Doe"},
            {"role": "user", "content": "unrelated"},
        ],
    }
    engine_reply["request"] = {
        **original,
        "messages": [
            {"role": "user", "content": "protected"},
            {"role": "user", "content": REVERSIBLE_TOKEN},
        ],
    }
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    with pytest.raises(InvalidEngineReplyError, match="original text leaf"):
        await handler.handle(body_request(json.dumps(original).encode()))


async def test_reversal_cannot_create_more_occurrences_than_the_source(
    engine_client, engine_reply
) -> None:
    """One plaintext occurrence cannot authorize duplicated recovery tokens."""
    engine_reply["request"]["messages"][0]["content"] = f"{REVERSIBLE_TOKEN} {REVERSIBLE_TOKEN}"
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    with pytest.raises(InvalidEngineReplyError, match="original text leaf"):
        await handler.handle(body_request(request_json()))


async def test_reversal_occurrences_must_match_the_detailed_report(
    engine_client, engine_reply
) -> None:
    """A reversible report cannot claim a transformation without its request token."""
    engine_reply["request"]["messages"][0]["content"] = "Jane Doe"
    engine_reply["reversal"] = {}
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())

    with pytest.raises(InvalidEngineReplyError, match="occurrence counts"):
        await handler.handle(body_request(request_json()))


async def test_preexisting_request_placeholder_is_rejected(engine_client) -> None:
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    body = json.dumps(
        {"model": "test", "messages": [{"role": "user", "content": REVERSIBLE_TOKEN}]}
    ).encode()

    with pytest.raises(InvalidEngineReplyError, match="already existed"):
        await handler.handle(body_request(body))


async def test_large_reversal_map_restores_exact_request_in_linear_pass(
    engine_client, engine_reply
) -> None:
    reversal = {
        f"<REV_PERSON_{index:016x}_{index + 1:016x}>": f"value-{index}" for index in range(618)
    }
    original = "|".join(reversal.values())
    transformed = "|".join(reversal)
    engine_reply.update(
        {
            "entity_counts": {"PERSON": 618},
            "request": {
                "model": "test",
                "messages": [{"role": "user", "content": transformed}],
            },
            "notices": {"request": [], "response": []},
            "report": {
                "rows": [
                    {
                        "entity_type": "PERSON",
                        "action": "reversible_replace",
                        "detected_count": 618,
                        "transformed_count": 618,
                        "unique_transformed_count": 618,
                    }
                ]
            },
            "reversal": reversal,
        }
    )
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    body = json.dumps(
        {"model": "test", "messages": [{"role": "user", "content": original}]}
    ).encode()

    response = await handler.handle(body_request(body))

    assert response is not None
    assert not response.HasField("immediate_response")
    assert len(handler.reversal_map) == 618


async def test_reversal_allows_other_transformations_in_the_same_leaf(
    engine_client, engine_reply
) -> None:
    original = "Jane Doe called 555-0100"
    engine_reply.update(
        {
            "entities": ["PERSON", "PHONE_NUMBER"],
            "entity_counts": {"PERSON": 1, "PHONE_NUMBER": 1},
            "applied_actions": ["mask", "reversible_replace"],
            "request": {
                "model": "test",
                "messages": [{"role": "user", "content": f"{REVERSIBLE_TOKEN} called ***-****"}],
            },
            "notices": {"request": [], "response": []},
            "report": {
                "rows": [
                    {
                        "entity_type": "PERSON",
                        "action": "reversible_replace",
                        "detected_count": 1,
                        "transformed_count": 1,
                        "unique_transformed_count": 1,
                    },
                    {
                        "entity_type": "PHONE_NUMBER",
                        "action": "mask",
                        "detected_count": 1,
                        "transformed_count": 1,
                        "unique_transformed_count": 1,
                    },
                ]
            },
        }
    )
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    body = json.dumps(
        {"model": "test", "messages": [{"role": "user", "content": original}]}
    ).encode()

    response = await handler.handle(body_request(body))

    assert response is not None
    assert not response.HasField("immediate_response")


async def test_engine_may_mutate_response_format_schema_prose(engine_client, engine_reply) -> None:
    """Response-format schema descriptions are model-visible rather than controls."""
    original = {
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string", "description": "Jane Doe"}},
                },
            },
        },
    }
    engine_reply["reversal"] = {}
    engine_reply["applied_actions"] = ["replace"]
    engine_reply["report"]["rows"][0].update(
        {
            "action": "replace",
            "transformed_count": 1,
            "unique_transformed_count": 1,
        }
    )
    engine_reply["request"] = json.loads(json.dumps(original))
    engine_reply["request"]["response_format"]["json_schema"]["schema"]["properties"]["value"][
        "description"
    ] = "Protected"
    engine_reply["notices"]["request"] = []
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    response = await handler.handle(body_request(json.dumps(original).encode()))
    assert response is not None
    forwarded = json.loads(response.request_body.response.body_mutation.body)
    assert (
        forwarded["response_format"]["json_schema"]["schema"]["properties"]["value"]["description"]
        == "Protected"
    )


async def test_responses_text_format_schema_prose_and_structured_output(
    engine_client, engine_reply
) -> None:
    """Responses text.format preserves controls, suppresses notices, and reverses JSON output."""
    original = {
        "model": "test",
        "input": "hello",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "description": "Jane Doe",
                "schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string", "description": "Jane Doe"}},
                },
                "strict": True,
            }
        },
    }
    transformed = json.loads(json.dumps(original))
    transformed["text"]["format"]["description"] = REVERSIBLE_TOKEN
    transformed["text"]["format"]["schema"]["properties"]["value"]["description"] = REVERSIBLE_TOKEN
    engine_reply["entity_counts"] = {"PERSON": 2}
    engine_reply["report"]["rows"][0].update(
        {
            "detected_count": 2,
            "transformed_count": 2,
            "unique_transformed_count": 1,
        }
    )
    engine_reply["request"] = transformed
    engine_reply["notices"]["request"] = []
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    request = await handler.handle(body_request(json.dumps(original).encode()))
    assert request is not None
    forwarded = json.loads(request.request_body.response.body_mutation.body)
    assert forwarded["text"]["format"]["description"] == REVERSIBLE_TOKEN
    assert forwarded["instructions"] == GUARD_INSTRUCTION
    assert handler.response_structured_json
    assert not handler.response_notice_allowed

    await handler.handle(response_headers("application/json"))
    content = json.dumps({"value": REVERSIBLE_TOKEN})
    payload = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": content}],
            }
        ]
    }
    response = await handler.handle(response_body(json.dumps(payload).encode()))
    assert response is not None
    output = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    assert json.loads(output["output"][0]["content"][0]["text"]) == {"value": "Jane Doe"}
    assert "PII Engine Notice" not in json.dumps(output)


async def test_engine_cannot_mutate_responses_text_format_controls(
    engine_client, engine_reply
) -> None:
    """The engine may change schema prose but not structured-output protocol controls."""
    original = {
        "model": "test",
        "input": "hello",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object"},
                "strict": True,
            }
        },
    }
    engine_reply.update(
        {
            "decision": "pass",
            "entities": [],
            "entity_counts": {},
            "applied_actions": [],
            "request": json.loads(json.dumps(original)),
            "report": {"rows": []},
            "reversal": {},
        }
    )
    engine_reply["request"]["text"]["format"]["name"] = "changed"
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    with pytest.raises(InvalidEngineReplyError, match="protocol controls"):
        await handler.handle(body_request(json.dumps(original).encode()))


async def test_responses_verbosity_without_format_remains_ordinary_text(
    engine_client, engine_reply
) -> None:
    """An optional text config is structured only when its format is non-null."""
    original = {
        "model": "test",
        "input": "Jane Doe",
        "text": {"verbosity": "high"},
    }
    engine_reply["request"] = {
        "model": "test",
        "input": REVERSIBLE_TOKEN,
        "text": {"verbosity": "high"},
    }
    engine_reply["notices"]["request"] = []
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())

    response = await handler.handle(body_request(json.dumps(original).encode()))

    assert response is not None
    assert not handler.response_structured_json
    assert handler.response_notice_allowed


async def test_responses_sse_is_buffered_and_rewritten_as_responses_events(
    engine_client, engine_reply
) -> None:
    """Responses SSE survives arbitrary chunks and receives its own typed notice delta."""
    engine_reply.update(
        {
            "decision": "pass",
            "entities": [],
            "entity_counts": {},
            "applied_actions": [],
            "route_class": "general",
            "request": {"model": "test", "input": "hello", "stream": True},
            "notices": {"request": [], "response": ["Protected"]},
            "report": {"rows": []},
            "reversal": {},
        }
    )
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    request = json.dumps({"model": "test", "input": "hello", "stream": True}).encode()
    await handler.handle(body_request(request))
    await handler.handle(response_headers("text/event-stream"))
    split = len(GUARD_INSTRUCTION) // 2
    guarded_answer = f"{GUARD_INSTRUCTION}answer"
    events = [
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": GUARD_INSTRUCTION[:split],
            "sequence_number": 2,
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": GUARD_INSTRUCTION[split:] + "answer",
            "sequence_number": 3,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "text": guarded_answer,
            "sequence_number": 4,
        },
        {
            "type": "response.completed",
            "sequence_number": 5,
            "response": {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": guarded_answer}],
                    }
                ]
            },
        },
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
    first = await handler.handle(response_body(stream[:80], end_of_stream=False))
    assert first is not None
    final = await handler.handle(response_body(stream[80:]))
    assert final is not None
    output = final.response_body.response.body_mutation.streamed_response.body.decode()
    assert output.index("PII Engine Notice") < output.index("response.output_text.done")
    assert '"type":"response.output_text.delta"' in output
    assert GUARD_INSTRUCTION not in output


@pytest.mark.parametrize("separate_usage_tail", [True, False], ids=["separate", "coalesced"])
async def test_chat_finish_usage_done_shapes_survive_chunk_boundaries(
    engine_client, separate_usage_tail: bool
) -> None:
    """Both usage placements survive through the global completion marker."""
    handler = StreamHandler(engine_client)
    handler.notice_messages = []
    await handler.handle(response_headers("text/event-stream"))
    finish = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    usage = {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    events = [finish, usage]
    if not separate_usage_tail:
        events = [{**finish, "usage": usage["usage"]}]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    stream += "data: [DONE]\n\n"
    split = stream.index("usage") + 3
    first = await handler.handle(response_body(stream[:split].encode(), end_of_stream=False))
    final = await handler.handle(response_body(stream[split:].encode()))
    assert first is not None
    assert final is not None
    output = _stream_text(first, final)
    assert [_sse_data(line) for line in output.splitlines() if line.startswith("data: ")] == [
        *events,
        "[DONE]",
    ]


async def test_chat_multiple_choices_continue_after_one_finishes(engine_client) -> None:
    """A finish_reason applies only to its indexed choice."""
    handler = StreamHandler(engine_client)
    handler.notice_messages = []
    await handler.handle(response_headers("text/event-stream"))
    events = [
        {"choices": [{"index": 0, "delta": {"content": "first"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [{"index": 1, "delta": {"content": "later"}}]},
        {"choices": [{"index": 1, "delta": {}, "finish_reason": "stop"}]},
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    response = await handler.handle(response_body(stream.encode()))
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    payloads = [_sse_data(line) for line in output.splitlines() if line.startswith("data: {")]
    assert payloads == events


@pytest.mark.parametrize("streaming", [False, True])
async def test_chat_tool_only_null_content_skips_body_notice(engine_client, streaming) -> None:
    """Tool-only responses remain valid and do not gain synthetic text output."""
    handler = StreamHandler(engine_client)
    handler.notice_messages = ["Protected"]
    content_type = "text/event-stream" if streaming else "application/json"
    await handler.handle(response_headers(content_type))
    if streaming:
        tool = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "lookup", "arguments": '{"id":1}'},
                            }
                        ],
                    },
                }
            ]
        }
        finish = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        body = f"data: {json.dumps(tool)}\n\ndata: {json.dumps(finish)}\n\ndata: [DONE]\n\n"
    else:
        body = json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {"type": "function", "function": {"arguments": '{"id":1}'}}
                            ],
                        },
                    }
                ]
            }
        )
    response = await handler.handle(response_body(body.encode()))
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert "PII Engine Notice" not in output
    assert '"content":null' in output


@pytest.mark.parametrize("streaming", [False, True])
async def test_responses_tool_only_output_skips_body_notice(engine_client, streaming) -> None:
    """Responses function output is valid without an output-text notice target."""
    handler = StreamHandler(engine_client)
    handler.response_api_kind = "responses"
    handler.notice_messages = ["Protected"]
    content_type = "text/event-stream" if streaming else "application/json"
    await handler.handle(response_headers(content_type))
    if streaming:
        events = [
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "call_1",
                "output_index": 0,
                "delta": '{"id":1}',
                "sequence_number": 1,
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "function_call", "arguments": '{"id":1}'},
                "sequence_number": 2,
            },
            {"type": "response.completed", "response": {"output": []}, "sequence_number": 3},
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    else:
        body = json.dumps(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": '{"id":1}',
                    }
                ]
            }
        )
    response = await handler.handle(response_body(body.encode()))
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert "PII Engine Notice" not in output
    assert "function_call" in output


async def test_structured_chat_skips_notice_and_sets_response_code(
    engine_client, engine_reply
) -> None:
    """Structured Chat output is not polluted while its analysis code is exposed."""
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        },
    }
    engine_reply["request"]["response_format"] = response_format
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "Jane Doe"}],
        "response_format": response_format,
    }
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(json.dumps(request).encode()))
    assert await handler.handle(response_headers("application/json")) is None
    headers = handler.pop_pending_response_headers()
    assert headers is not None
    assert _set_headers(headers)["x-presidio-code"] == "P02"
    response = await handler.handle(
        response_body(b'{"choices":[{"index":0,"message":{"content":"{\\"answer\\":\\"ok\\"}"}}]}')
    )
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert "PII Engine Notice" not in output


@pytest.mark.parametrize("api_kind", ["chat", "responses"])
async def test_nonstream_tool_arguments_preserve_nested_json(engine_client, api_kind) -> None:
    """Complete tool arguments reverse semantic values without breaking inner JSON."""
    handler = StreamHandler(engine_client)
    handler.response_api_kind = api_kind
    handler.notice_messages = []
    handler.reversal_map = {REVERSIBLE_TOKEN: SPECIAL_PLAINTEXT}
    handler.guard_injected = True
    await handler.handle(response_headers("application/json"))
    arguments = json.dumps({"value": GUARD_INSTRUCTION + REVERSIBLE_TOKEN})
    if api_kind == "chat":
        payload = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [{"function": {"name": "lookup", "arguments": arguments}}],
                    }
                }
            ]
        }
    else:
        payload = {"output": [{"type": "function_call", "name": "lookup", "arguments": arguments}]}
    response = await handler.handle(response_body(json.dumps(payload).encode()))
    assert response is not None
    transformed = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    if api_kind == "chat":
        nested = transformed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    else:
        nested = transformed["output"][0]["arguments"]
    assert json.loads(nested) == {"value": SPECIAL_PLAINTEXT}


async def test_buffered_chat_preserves_opaque_reasoning_values(engine_client) -> None:
    """Reasoning subtrees are not scanned while ordinary output channels are reversed."""
    unknown = "<REV_UNKNOWN_0123456789abcdef_fedcba9876543210>"
    reasoning = {
        "reasoning_content": REVERSIBLE_TOKEN,
        "reasoning": unknown,
        "reasoning_details": {"echo": GUARD_INSTRUCTION + REVERSIBLE_TOKEN},
        "thinking_blocks": [{"signature": unknown}],
        "reasoning_signature": None,
    }
    handler = StreamHandler(engine_client)
    handler.notice_messages = []
    handler.reversal_map = {REVERSIBLE_TOKEN: SPECIAL_PLAINTEXT}
    handler.guard_injected = True
    await handler.handle(response_headers("application/json"))
    arguments = json.dumps({"value": REVERSIBLE_TOKEN})
    payload = {
        "choices": [
            {
                "message": {
                    "content": REVERSIBLE_TOKEN,
                    **reasoning,
                    "tool_calls": [{"function": {"name": "lookup", "arguments": arguments}}],
                }
            }
        ]
    }

    response = await handler.handle(response_body(json.dumps(payload).encode()))

    assert response is not None
    transformed = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    message = transformed["choices"][0]["message"]
    assert {field: message[field] for field in reasoning} == reasoning
    assert message["content"] == SPECIAL_PLAINTEXT
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "value": SPECIAL_PLAINTEXT
    }


async def test_streaming_chat_preserves_opaque_reasoning_deltas(engine_client) -> None:
    """Reasoning deltas remain opaque while content deltas use reversal state."""
    reasoning = {
        "reasoning_content": GUARD_INSTRUCTION + REVERSIBLE_TOKEN,
        "reasoning": "<REV_UNKNOWN_0123456789abcdef_fedcba9876543210>",
        "reasoning_details": {"parts": [REVERSIBLE_TOKEN]},
        "thinking_blocks": [None, {"text": GUARD_INSTRUCTION}],
        "reasoning_signature": None,
    }
    handler = StreamHandler(engine_client)
    handler.notice_messages = []
    handler.reversal_map = {REVERSIBLE_TOKEN: SPECIAL_PLAINTEXT}
    handler.guard_injected = True
    await handler.handle(response_headers("text/event-stream"))
    event = {"choices": [{"index": 0, "delta": {"content": REVERSIBLE_TOKEN, **reasoning}}]}
    stream = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"

    response = await handler.handle(response_body(stream.encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    payload = next(_sse_data(line) for line in output.splitlines() if line.startswith("data: {"))
    delta = payload["choices"][0]["delta"]
    assert {field: delta[field] for field in reasoning} == reasoning
    assert delta["content"] == SPECIAL_PLAINTEXT


async def test_arbitrary_arguments_metadata_is_not_treated_as_function_json(engine_client) -> None:
    """Only exact function carriers receive nested-JSON guard and reversal processing."""
    handler = StreamHandler(engine_client)
    handler.notice_messages = []
    handler.guard_injected = True
    await handler.handle(response_headers("application/json"))
    payload = {
        "choices": [
            {
                "message": {"content": "answer"},
                "metadata": {"arguments": GUARD_INSTRUCTION},
            }
        ]
    }

    response = await handler.handle(response_body(json.dumps(payload).encode()))

    assert response is not None
    transformed = json.loads(response.response_body.response.body_mutation.streamed_response.body)
    assert transformed["choices"][0]["metadata"]["arguments"] == GUARD_INSTRUCTION


async def test_streaming_responses_near_miss_arguments_remain_protocol_data(
    engine_client,
) -> None:
    """Arguments outside exact function carriers bypass nested-JSON transformation."""
    handler = StreamHandler(engine_client)
    handler.response_api_kind = "responses"
    handler.notice_messages = []
    handler.guard_injected = True
    await handler.handle(response_headers("text/event-stream"))
    events = [
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "message", "arguments": GUARD_INSTRUCTION},
            "sequence_number": 1,
        },
        {
            "type": "response.completed",
            "response": {"output": [], "arguments": GUARD_INSTRUCTION},
            "sequence_number": 2,
        },
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    response = await handler.handle(response_body(stream.encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    payloads = [_sse_data(line) for line in output.splitlines() if line.startswith("data: {")]
    assert payloads[0]["item"]["arguments"] == GUARD_INSTRUCTION
    assert payloads[1]["response"]["arguments"] == GUARD_INSTRUCTION


async def test_streaming_chat_tool_arguments_preserve_nested_json(engine_client) -> None:
    """Chat argument deltas escape restored plaintext as an inner JSON string fragment."""
    handler = StreamHandler(engine_client)
    handler.notice_messages = []
    handler.reversal_map = {REVERSIBLE_TOKEN: SPECIAL_PLAINTEXT}
    handler.guard_injected = True
    await handler.handle(response_headers("text/event-stream"))
    guard_split = len(GUARD_INSTRUCTION) // 2
    token_split = len(REVERSIBLE_TOKEN) // 2
    fragments = [
        '{"value":"' + GUARD_INSTRUCTION[:guard_split],
        GUARD_INSTRUCTION[guard_split:] + REVERSIBLE_TOKEN[:token_split],
        REVERSIBLE_TOKEN[token_split:] + '"}',
    ]
    events: list[dict[str, object]] = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "lookup", "arguments": fragment},
                            }
                        ]
                    },
                }
            ]
        }
        for fragment in fragments
    ]
    events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    stream += "data: [DONE]\n\n"
    response = await handler.handle(response_body(stream.encode()))
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    arguments = "".join(
        payload["choices"][0]["delta"]
        .get("tool_calls", [{}])[0]
        .get("function", {})
        .get("arguments", "")
        for payload in (
            _sse_data(line) for line in output.splitlines() if line.startswith("data: {")
        )
    )
    assert json.loads(arguments) == {"value": SPECIAL_PLAINTEXT}


async def test_streaming_responses_function_arguments_preserve_nested_json(
    engine_client,
) -> None:
    """Responses argument deltas and done snapshots retain valid equivalent JSON."""
    handler = StreamHandler(engine_client)
    handler.response_api_kind = "responses"
    handler.notice_messages = []
    handler.reversal_map = {REVERSIBLE_TOKEN: SPECIAL_PLAINTEXT}
    handler.guard_injected = True
    await handler.handle(response_headers("text/event-stream"))
    arguments = json.dumps({"value": GUARD_INSTRUCTION + REVERSIBLE_TOKEN})
    events = [
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "call_1",
            "output_index": 0,
            "delta": arguments,
            "sequence_number": 1,
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "call_1",
            "output_index": 0,
            "name": "lookup",
            "arguments": arguments,
            "sequence_number": 2,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "name": "lookup", "arguments": arguments},
            "sequence_number": 3,
        },
        {"type": "response.completed", "response": {"output": []}, "sequence_number": 4},
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    response = await handler.handle(response_body(stream.encode()))
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    payloads = [_sse_data(line) for line in output.splitlines() if line.startswith("data: {")]
    delta = next(item["delta"] for item in payloads if item["type"].endswith(".delta"))
    done_arguments = [item["arguments"] for item in payloads if "arguments" in item]
    item_arguments = next(
        item["item"]["arguments"]
        for item in payloads
        if item["type"] == "response.output_item.done"
    )
    assert json.loads(delta) == {"value": SPECIAL_PLAINTEXT}
    assert all(json.loads(value) == {"value": SPECIAL_PLAINTEXT} for value in done_arguments)
    assert json.loads(item_arguments) == {"value": SPECIAL_PLAINTEXT}


async def test_streaming_responses_structured_output_preserves_nested_json(
    engine_client,
) -> None:
    """Responses structured text deltas and done snapshots remain valid inner JSON."""
    handler = StreamHandler(engine_client)
    handler.response_api_kind = "responses"
    handler.response_structured_json = True
    handler.response_notice_allowed = False
    handler.notice_messages = ["Protected"]
    handler.reversal_map = {REVERSIBLE_TOKEN: SPECIAL_PLAINTEXT}
    handler.guard_injected = True
    await handler.handle(response_headers("text/event-stream"))
    content = json.dumps({"value": GUARD_INSTRUCTION + REVERSIBLE_TOKEN})
    events = [
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": content,
            "sequence_number": 1,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "text": content,
            "sequence_number": 2,
        },
        {"type": "response.completed", "response": {"output": []}, "sequence_number": 3},
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    response = await handler.handle(response_body(stream.encode()))
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    payloads = [_sse_data(line) for line in output.splitlines() if line.startswith("data: {")]
    delta = next(item["delta"] for item in payloads if item["type"].endswith(".delta"))
    done = next(item["text"] for item in payloads if item["type"].endswith(".done"))
    assert json.loads(delta) == {"value": SPECIAL_PLAINTEXT}
    assert json.loads(done) == {"value": SPECIAL_PLAINTEXT}
    assert "PII Engine Notice" not in output


@pytest.mark.parametrize("streaming", [False, True])
async def test_structured_chat_reversal_preserves_nested_json(engine_client, streaming) -> None:
    """Structured Chat content reverses as JSON values and never receives a body notice."""
    handler = StreamHandler(engine_client)
    handler.response_structured_json = True
    handler.response_notice_allowed = False
    handler.notice_messages = ["Protected"]
    handler.reversal_map = {REVERSIBLE_TOKEN: SPECIAL_PLAINTEXT}
    handler.guard_injected = True
    content_type = "text/event-stream" if streaming else "application/json"
    await handler.handle(response_headers(content_type))
    content = json.dumps({"value": GUARD_INSTRUCTION + REVERSIBLE_TOKEN})
    if streaming:
        event = {"choices": [{"index": 0, "delta": {"content": content}}]}
        finish = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        body = f"data: {json.dumps(event)}\n\ndata: {json.dumps(finish)}\n\ndata: [DONE]\n\n"
    else:
        body = json.dumps({"choices": [{"message": {"content": content}}]})
    response = await handler.handle(response_body(body.encode()))
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    if streaming:
        payloads = [_sse_data(line) for line in output.splitlines() if line.startswith("data: {")]
        transformed_content = "".join(
            item["choices"][0]["delta"].get("content", "") for item in payloads
        )
    else:
        transformed_content = json.loads(output)["choices"][0]["message"]["content"]
    assert json.loads(transformed_content) == {"value": SPECIAL_PLAINTEXT}
    assert "PII Engine Notice" not in output


async def test_unknown_placeholder_in_nested_json_fails_closed(engine_client) -> None:
    """JSON-aware argument reversal retains unknown-placeholder fail-closed behavior."""
    handler = StreamHandler(engine_client)
    handler.notice_messages = []
    await handler.handle(response_headers("application/json"))
    arguments = json.dumps({"value": "<REV_UNKNOWN_0123456789abcdef_fedcba9876543210>"})
    body = json.dumps(
        {"choices": [{"message": {"tool_calls": [{"function": {"arguments": arguments}}]}}]}
    )
    with pytest.raises(InvalidReversalError, match="unknown placeholder"):
        await handler.handle(response_body(body.encode()))


@pytest.mark.parametrize(
    ("decision", "entities", "expected"),
    [
        ("pass", [], "P00"),
        ("pass", ["PERSON"], "P01"),
        ("apply_actions", ["PERSON"], "P02"),
        ("reroute", ["PERSON"], "P03"),
    ],
)
async def test_response_code_is_derived_from_validated_reply(
    engine_client, engine_reply, decision, entities, expected
) -> None:
    """The response header uses only the four stable documented values."""
    engine_reply["decision"] = decision
    engine_reply["entities"] = entities
    engine_reply["entity_counts"] = dict.fromkeys(entities, 1)
    report_action = (
        "pass"
        if decision == "pass"
        else "reroute"
        if decision == "reroute"
        else "reversible_replace"
    )
    transformed_count = 1 if decision == "apply_actions" else 0
    engine_reply["report"] = {
        "rows": [
            {
                "entity_type": entity,
                "action": report_action,
                "detected_count": 1,
                "transformed_count": transformed_count,
                "unique_transformed_count": transformed_count,
            }
            for entity in entities
        ],
    }
    if decision == "apply_actions":
        pass
    elif decision == "reroute":
        engine_reply.update(
            {
                "applied_actions": ["reroute"],
                "remote_allowed": False,
                "route_class": "local-sensitive",
                "request": {
                    "model": "test",
                    "messages": [{"role": "user", "content": "Jane Doe"}],
                },
                "reversal": {},
            }
        )
    else:
        engine_reply.update(
            {
                "applied_actions": [],
                "request": {
                    "model": "test",
                    "messages": [{"role": "user", "content": "Jane Doe"}],
                },
                "reversal": {},
            }
        )
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())
    await handler.handle(body_request(request_json()))
    await handler.handle(
        response_headers(
            "application/json",
            extra_headers={"x-presidio-code": "untrusted"},
        )
    )
    headers = handler.pop_pending_response_headers()
    assert headers is not None
    set_codes = [
        item.header.value
        for item in headers.response_headers.response.header_mutation.set_headers
        if item.header.key == "x-presidio-code"
    ]
    assert set_codes == [expected]
    assert "x-presidio-code" in _removed_headers(headers)


@pytest.mark.parametrize("decision", ["pass", "apply_actions"])
async def test_unscanned_current_nonterminal_reply_fails_closed(
    engine_client, engine_reply, decision
) -> None:
    """Impossible no-scan nonterminal replies fail before response metadata is assigned."""
    if decision == "pass":
        engine_reply.update(
            {
                "decision": "pass",
                "entities": [],
                "entity_counts": {},
                "applied_actions": [],
                "request": {
                    "model": "test",
                    "messages": [{"role": "user", "content": "Jane Doe"}],
                },
                "report": {"rows": []},
                "reversal": {},
            }
        )
    engine_reply["analysis"]["scan_performed"] = False
    engine_reply["analysis"]["duration_ms"] = None
    handler = StreamHandler(engine_client)
    await handler.handle(header_request())

    with pytest.raises(InvalidEngineReplyError):
        await handler.handle(body_request(request_json()))

    assert handler.presidio_code is None


async def test_non_success_response_strips_presidio_code_without_replacement(
    engine_client,
) -> None:
    """Upstream errors cannot retain or receive a Presidio response code."""
    handler = StreamHandler(engine_client)
    handler.presidio_code = "P02"
    await handler.handle(
        response_headers(
            "application/json",
            status=429,
            extra_headers={"x-presidio-code": "untrusted"},
        )
    )
    headers = handler.pop_pending_response_headers()
    assert headers is not None
    assert "x-presidio-code" in _removed_headers(headers)
    assert "x-presidio-code" not in _set_headers(headers)


async def test_responses_mixed_text_and_function_events_are_not_early_terminal(
    engine_client,
) -> None:
    """Item done events preserve later function output until response completion."""
    handler = StreamHandler(engine_client)
    handler.response_api_kind = "responses"
    handler.notice_messages = []
    handler.reversal_map = {REVERSIBLE_TOKEN: "Jane Doe"}
    await handler.handle(response_headers("text/event-stream"))
    function_arguments = json.dumps({"value": REVERSIBLE_TOKEN})
    events = [
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": REVERSIBLE_TOKEN,
            "sequence_number": 1,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "text": REVERSIBLE_TOKEN,
            "sequence_number": 2,
        },
        {
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": REVERSIBLE_TOKEN},
            "sequence_number": 3,
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "call_1",
            "output_index": 1,
            "delta": function_arguments,
            "sequence_number": 4,
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {"type": "function_call", "arguments": function_arguments},
            "sequence_number": 5,
        },
        {"type": "response.completed", "response": {"output": []}, "sequence_number": 6},
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    response = await handler.handle(response_body(stream.encode()))
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert output.count("Jane Doe") == 5
    assert output.index("response.function_call_arguments.delta") < output.index(
        "response.completed"
    )
    assert output.rstrip().endswith("data: [DONE]")


@pytest.mark.parametrize("terminal", ["response.failed", "response.incomplete", "error"])
async def test_responses_failure_terminals_skip_notice(engine_client, terminal) -> None:
    """Responses failure lifecycle events terminate cleanly without synthetic output."""
    handler = StreamHandler(engine_client)
    handler.response_api_kind = "responses"
    handler.notice_messages = ["Protected"]
    await handler.handle(response_headers("text/event-stream"))
    event = {"type": terminal, "sequence_number": 1}
    stream = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    response = await handler.handle(response_body(stream.encode()))
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert "PII Engine Notice" not in output
    assert terminal in output
    assert output.rstrip().endswith("data: [DONE]")


@pytest.mark.parametrize("terminal", ["response.failed", "response.incomplete"])
async def test_responses_failure_snapshots_mark_invalid_human_text(engine_client, terminal) -> None:
    handler = StreamHandler(engine_client)
    handler.response_api_kind = "responses"
    handler.reversal_map = {REVERSIBLE_TOKEN: "Jane Doe"}
    handler.notice_messages = []
    await handler.handle(response_headers("text/event-stream"))
    altered = "<REV_PERSON_0123456789abcdef_...>"
    event = {
        "type": terminal,
        "sequence_number": 1,
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": f"before {altered} after"}],
                }
            ]
        },
    }
    stream = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"

    response = await handler.handle(response_body(stream.encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert "before <PII_INVALID_PERSON_" in output
    assert "> after" in output
    assert altered not in output


async def test_responses_sse_emits_a_marker_for_an_incomplete_placeholder_at_eof(
    engine_client,
) -> None:
    handler = StreamHandler(engine_client)
    handler.response_api_kind = "responses"
    handler.reversal_map = {REVERSIBLE_TOKEN: "Jane Doe"}
    handler.notice_messages = []
    await handler.handle(response_headers("text/event-stream"))
    events = [
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "before <REV_PERSON_broken",
            "sequence_number": 1,
        },
        {"type": "response.completed", "response": {"output": []}, "sequence_number": 2},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"

    response = await handler.handle(response_body(body.encode()))

    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert "before " in output
    assert "<PII_INVALID_PERSON_" in output
    assert output.index("<PII_INVALID_PERSON_") < output.index("response.completed")
    assert output.rstrip().endswith("data: [DONE]")


async def test_gzip_json_is_validated_before_output(engine_client) -> None:
    """Buffered gzip JSON is decompressed and transformed as one validated document."""
    handler = StreamHandler(engine_client)
    handler.notice_messages = []
    handler.reversal_map = {REVERSIBLE_TOKEN: "Jane Doe"}
    await handler.handle(response_headers("application/json", "gzip"))
    compressed = gzip.compress(
        json.dumps({"choices": [{"message": {"content": REVERSIBLE_TOKEN}}]}).encode()
    )
    response = await handler.handle(response_body(compressed))
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert "Jane Doe" in output


def _stream_text(*responses) -> str:
    return b"".join(
        response.response_body.response.body_mutation.streamed_response.body
        for response in responses
    ).decode()


def _sse_data(line: str):
    value = line[6:]
    return value if value == "[DONE]" else json.loads(value)


def _set_headers(response) -> dict[str, str]:
    return {
        item.header.key: item.header.value
        for item in response.response_headers.response.header_mutation.set_headers
    }


def _removed_headers(response) -> set[str]:
    return set(response.response_headers.response.header_mutation.remove_headers)
