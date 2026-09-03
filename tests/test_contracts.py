"""Test the complete engine boundary and transport-only request helpers."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from agentgateway_extproc.lib.pipeline.guard import (
    GUARD_INSTRUCTION,
    GuardStreamStripper,
    inject_guard_instruction,
    strip_guard_instruction,
)
from agentgateway_extproc.lib.pipeline.request import _restore_reversal_leaves
from agentgateway_extproc.lib.session import make_session_key
from agentgateway_extproc.models.destination import ModelDestinationPolicy
from agentgateway_extproc.models.engine import (
    ENGINE_REQUEST_ADAPTER,
    EngineChatRequest,
    EngineMcpRequest,
    EngineReply,
    EngineResponsesRequest,
)
from agentgateway_extproc.models.exceptions import InvalidReversalError


def test_all_supported_request_families_validate_strictly() -> None:
    """Responses and MCP requests are no longer lost at the adapter boundary."""
    responses = ENGINE_REQUEST_ADAPTER.validate_python(
        {
            "model": "test",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        }
    )
    mcp = ENGINE_REQUEST_ADAPTER.validate_python(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lookup", "arguments": {"query": "hello"}},
        }
    )
    assert isinstance(responses, EngineResponsesRequest)
    assert isinstance(mcp, EngineMcpRequest)
    attachment = ENGINE_REQUEST_ADAPTER.validate_python(
        {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.test/a"}}
                    ],
                }
            ],
        }
    )
    assert isinstance(attachment, EngineChatRequest)
    with pytest.raises(ValidationError):
        ENGINE_REQUEST_ADAPTER.validate_python(
            {"model": "test", "messages": [{"role": "user", "image": "no"}]}
        )


@pytest.mark.parametrize("include_usage", [True, False])
def test_chat_stream_options_accepts_only_an_explicit_strict_boolean(include_usage: bool) -> None:
    request = ENGINE_REQUEST_ADAPTER.validate_python(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream_options": {"include_usage": include_usage},
        },
        strict=True,
    )

    assert isinstance(request, EngineChatRequest)
    assert request.stream_options is not None
    assert request.stream_options.include_usage is include_usage


@pytest.mark.parametrize(
    "update",
    [
        {"unknown": "rejected"},
        {"stream_options": {"include_usage": True, "unknown": "rejected"}},
        {"stream_options": {}},
        {"stream_options": {"include_usage": 1}},
        {"stream_options": {"include_usage": "true"}},
    ],
    ids=["top-level-extra", "nested-extra", "missing-boolean", "integer", "string"],
)
def test_chat_stream_options_rejects_fields_outside_the_strict_contract(
    update: dict[str, object],
) -> None:
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        **update,
    }

    with pytest.raises(ValidationError):
        ENGINE_REQUEST_ADAPTER.validate_python(payload, strict=True)


def test_responses_text_format_is_strict_and_bounded() -> None:
    """Responses structured output accepts only the bounded supported format contract."""
    request = ENGINE_REQUEST_ADAPTER.validate_python(
        {
            "model": "test",
            "input": "hello",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "description": "One answer",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            },
        }
    )
    assert isinstance(request, EngineResponsesRequest)
    assert request.model_dump(exclude_none=True)["text"]["format"]["schema"] == {"type": "object"}
    with pytest.raises(ValidationError):
        ENGINE_REQUEST_ADAPTER.validate_python(
            {
                "model": "test",
                "input": "hello",
                "text": {"format": {"type": "json_schema", "name": "bad name", "schema": {}}},
            }
        )
    verbose = ENGINE_REQUEST_ADAPTER.validate_python(
        {"model": "test", "input": "hello", "text": {"verbosity": "high"}}
    )
    assert isinstance(verbose, EngineResponsesRequest)
    assert verbose.text is not None
    assert verbose.text.format is None
    assert verbose.text.verbosity == "high"
    with pytest.raises(ValidationError):
        ENGINE_REQUEST_ADAPTER.validate_python(
            {"model": "test", "input": "hello", "text": {"verbosity": "maximal"}}
        )


def test_guard_is_the_leading_instruction_for_each_model_request_family() -> None:
    """One fixed guard precedes caller-controlled instructions and content."""
    chat = EngineChatRequest(
        model="test",
        messages=[
            {"role": "system", "content": "caller system"},
            {"role": "user", "content": "hello"},
        ],
    )
    responses = EngineResponsesRequest(
        model="test", input="hello", instructions="caller instructions"
    )

    injected_chat = inject_guard_instruction(chat)
    assert isinstance(injected_chat, EngineChatRequest)
    assert injected_chat.messages[0].role == "system"
    assert injected_chat.messages[0].content == GUARD_INSTRUCTION
    assert [message.content for message in injected_chat.messages[1:]] == [
        "caller system",
        "hello",
    ]
    assert [message.content for message in chat.messages] == ["caller system", "hello"]

    injected_responses = inject_guard_instruction(responses)
    assert isinstance(injected_responses, EngineResponsesRequest)
    assert injected_responses.instructions == f"{GUARD_INSTRUCTION}\n\ncaller instructions"
    assert responses.instructions == "caller instructions"


def test_guard_is_added_to_tool_only_and_instructionless_requests() -> None:
    """PII-enabled dispatch gets the guard even without ordinary input text."""
    chat = EngineChatRequest(
        model="test",
        messages=[
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
    )
    responses = EngineResponsesRequest(
        model="test",
        input=[
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": {"id": 1},
            }
        ],
    )

    injected_chat = inject_guard_instruction(chat)
    assert isinstance(injected_chat, EngineChatRequest)
    assert injected_chat.messages[0].content == GUARD_INSTRUCTION
    assert injected_chat.messages[1].tool_calls[0].function.name == "lookup"

    injected_responses = inject_guard_instruction(responses)
    assert isinstance(injected_responses, EngineResponsesRequest)
    assert injected_responses.instructions == GUARD_INSTRUCTION
    assert injected_responses.input == responses.input


def test_complete_guard_stripping_is_exact_and_repeatable() -> None:
    """Only complete exact copies of the protocol instruction are removed."""
    modified = GUARD_INSTRUCTION.replace("indivisible opaque alias", "ordinary alias")

    assert strip_guard_instruction(f"a{GUARD_INSTRUCTION}b{GUARD_INSTRUCTION}c") == "abc"
    assert strip_guard_instruction(modified) == modified


def test_stream_guard_stripping_spans_chunks_and_fails_closed_on_truncation() -> None:
    """One channel retains only the bounded suffix needed to identify an echo."""
    stripper = GuardStreamStripper()
    split = len(GUARD_INSTRUCTION) // 2

    assert stripper.feed(f"before{GUARD_INSTRUCTION[:split]}") == "before"
    assert stripper.feed(f"{GUARD_INSTRUCTION[split:]}after", final=True) == "after"

    truncated = GuardStreamStripper()
    assert truncated.feed(GUARD_INSTRUCTION[:20]) == ""
    with pytest.raises(InvalidReversalError, match="incomplete guard"):
        truncated.feed("", final=True)


def test_engine_reply_control_flow_fields_must_agree(engine_reply: dict[str, object]) -> None:
    """A malformed routing decision cannot influence trusted AgentGateway headers."""
    engine_reply["decision"] = "reroute"
    engine_reply["route_class"] = "trusted-local"
    with pytest.raises(ValidationError):
        EngineReply.model_validate(engine_reply)


def test_engine_reply_accepts_classified_pass(engine_reply: dict[str, object]) -> None:
    """A pass decision may include the engine's non-binding classifier result."""
    engine_reply.update(
        {
            "decision": "pass",
            "applied_actions": ["pass"],
            "route_class": "general",
            "report": {
                "rows": [
                    {
                        "entity_type": "PERSON",
                        "action": "pass",
                        "detected_count": 1,
                        "transformed_count": 0,
                        "unique_transformed_count": 0,
                    }
                ],
            },
            "reversal": {},
        }
    )
    assert EngineReply.model_validate(engine_reply).route_class == "general"


def test_engine_reply_accepts_block_action(engine_reply: dict[str, object]) -> None:
    """The engine reports block as the action that produced the terminal decision."""
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
    assert EngineReply.model_validate(engine_reply).decision == "block"


def test_engine_reply_requires_explicit_api_version(engine_reply: dict[str, object]) -> None:
    """An unversioned engine response cannot silently inherit the current contract."""
    del engine_reply["api_version"]
    with pytest.raises(ValidationError):
        EngineReply.model_validate(engine_reply)


def test_engine_reply_requires_detailed_report_without_a_compatibility_fallback(
    engine_reply: dict[str, object],
) -> None:
    """The coordinated release rejects replies that omit the required report."""
    del engine_reply["report"]
    with pytest.raises(ValidationError):
        EngineReply.model_validate(engine_reply)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"detected_count": 2}, "must match engine entity counts"),
        (
            {"action": "pass", "transformed_count": 0, "unique_transformed_count": 0},
            "action decisions",
        ),
    ],
)
def test_engine_reply_rejects_inconsistent_current_report(
    engine_reply: dict[str, object], update: dict[str, object], message: str
) -> None:
    report = cast(dict[str, object], engine_reply["report"])
    rows = cast(list[dict[str, object]], report["rows"])
    rows[0].update(update)

    with pytest.raises(ValidationError, match=message):
        EngineReply.model_validate(engine_reply)


@pytest.mark.parametrize(
    ("decision", "rows", "source", "cached", "scan_performed", "valid"),
    [
        ("pass", [("PERSON", "pass", 0)], "current_request", False, True, True),
        ("pass", [("PERSON", "mask", 0)], "current_request", False, True, False),
        (
            "apply_actions",
            [("EMAIL_ADDRESS", "mask", 1), ("PERSON", "pass", 0)],
            "current_request",
            False,
            True,
            True,
        ),
        (
            "apply_actions",
            [("EMAIL_ADDRESS", "mask", 1), ("PERSON", "block", 0)],
            "current_request",
            False,
            True,
            False,
        ),
        (
            "apply_actions",
            [("EMAIL_ADDRESS", "mask", 1), ("PERSON", "reroute", 0)],
            "current_request",
            False,
            True,
            False,
        ),
        ("reroute", [("PERSON", "mask", 1)], "current_request", False, True, False),
        (
            "reroute",
            [("EMAIL_ADDRESS", "block", 0), ("PERSON", "reroute", 0)],
            "current_request",
            False,
            True,
            False,
        ),
        ("reroute", [("PERSON", "mask", 1)], "current_request", True, True, True),
        ("block", [], "current_request", False, False, True),
        (
            "block",
            [("EMAIL_ADDRESS", "block", 0), ("PERSON", "mask", 1)],
            "current_request",
            False,
            True,
            False,
        ),
        ("block", [("PERSON", "reroute", 0)], "cached_decision", True, False, False),
        ("block", [("PERSON", "block", 0)], "cached_decision", True, False, True),
        ("reroute", [("PERSON", "block", 0)], "cached_decision", True, False, False),
        ("reroute", [("PERSON", "reroute", 0)], "cached_decision", True, False, True),
    ],
)
def test_engine_reply_report_rows_match_the_effective_decision(
    engine_reply: dict[str, object],
    decision: str,
    rows: list[tuple[str, str, int]],
    source: str,
    cached: bool,
    scan_performed: bool,
    valid: bool,
) -> None:
    report_rows = [
        {
            "entity_type": entity,
            "action": action,
            "detected_count": 1,
            "transformed_count": transformed,
            "unique_transformed_count": transformed,
        }
        for entity, action, transformed in rows
    ]
    entities = [entity for entity, _action, _transformed in rows]
    engine_reply.update(
        {
            "decision": decision,
            "entities": entities,
            "entity_counts": dict.fromkeys(entities, 1),
            "applied_actions": {
                "pass": [],
                "apply_actions": ["mask"],
                "reroute": ["reroute"],
                "block": ["block"],
            }[decision],
            "remote_allowed": decision not in {"block", "reroute"},
            "route_class": "local" if decision == "reroute" else None,
            "request": (
                None
                if decision == "block"
                else {"model": "test", "messages": [{"role": "user", "content": "text"}]}
            ),
            "report": {"rows": report_rows},
            "reversal": {},
        }
    )
    analysis = cast(dict[str, object], engine_reply["analysis"])
    analysis.update(
        {
            "source": source,
            "scan_performed": scan_performed,
            "duration_ms": 1 if scan_performed else None,
            "cached_decision_applied": cached,
        }
    )

    if valid:
        assert EngineReply.model_validate(engine_reply).decision == decision
    else:
        with pytest.raises(ValidationError):
            EngineReply.model_validate(engine_reply)


@pytest.mark.parametrize("decision", ["pass", "apply_actions"])
def test_engine_reply_rejects_unscanned_current_nonterminal_decisions(
    engine_reply: dict[str, object], decision: str
) -> None:
    if decision == "pass":
        engine_reply.update(
            {
                "decision": "pass",
                "entities": [],
                "entity_counts": {},
                "applied_actions": [],
                "request": {
                    "model": "test",
                    "messages": [{"role": "user", "content": "text"}],
                },
                "report": {"rows": []},
                "reversal": {},
            }
        )
    analysis = cast(dict[str, object], engine_reply["analysis"])
    analysis.update({"scan_performed": False, "duration_ms": None})

    with pytest.raises(ValidationError, match="unscanned current success"):
        EngineReply.model_validate(engine_reply)


def test_engine_reply_accepts_authoritative_no_text_mcp_with_omitted_optionals(
    engine_reply: dict[str, object],
) -> None:
    engine_reply.update(
        {
            "decision": "pass",
            "entities": [],
            "entity_counts": {},
            "applied_actions": [],
            "remote_allowed": True,
            "route_class": None,
            "request": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "lookup"},
            },
            "notices": {"request": [], "response": []},
            "report": {"rows": []},
            "reversal": {},
        }
    )
    analysis = cast(dict[str, object], engine_reply["analysis"])
    analysis.update(
        {
            "scan_performed": False,
            "duration_ms": None,
            "overlap_count": 0,
            "text_leaf_count": 0,
        }
    )

    reply = EngineReply.model_validate(engine_reply, strict=True)
    assert isinstance(reply.request, EngineMcpRequest)
    assert reply.request.params.arguments is None
    assert reply.request.params.meta is None


@pytest.mark.parametrize("optional", ["arguments", "_meta"])
def test_engine_reply_rejects_explicit_null_mcp_optionals(
    engine_reply: dict[str, object], optional: str
) -> None:
    engine_reply["request"] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "lookup", optional: None},
    }
    engine_reply["notices"] = {"request": [], "response": []}

    with pytest.raises(ValidationError, match="optional MCP params must be objects"):
        EngineReply.model_validate(engine_reply, strict=True)


@pytest.mark.parametrize(
    "updates",
    [
        {"notices": {"request": ["not allowed"], "response": []}},
        {"route_class": "local"},
    ],
)
def test_engine_reply_rejects_mcp_routing_and_notices(
    engine_reply: dict[str, object], updates: dict[str, object]
) -> None:
    engine_reply.update(
        {
            "request": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "lookup", "arguments": {"query": "safe"}},
            },
            "notices": {"request": [], "response": []},
        }
    )
    engine_reply.update(updates)

    with pytest.raises(ValidationError, match="MCP analysis"):
        EngineReply.model_validate(engine_reply, strict=True)


def test_engine_reply_accepts_request_text_above_obsolete_leaf_limit(
    engine_reply: dict[str, object],
) -> None:
    request = cast(dict[str, object], engine_reply["request"])
    messages = cast(list[dict[str, object]], request["messages"])
    messages[0]["content"] = "x" * 100_001
    reply = EngineReply.model_validate(engine_reply)
    assert isinstance(reply.request, EngineChatRequest)
    assert len(cast(str, reply.request.messages[0].content)) == 100_001


def test_engine_reply_accepts_more_than_617_reversal_entries(
    engine_reply: dict[str, object],
) -> None:
    entries = {
        f"<REV_PERSON_{index:016x}_{index + 1:016x}>": f"value-{index}" for index in range(618)
    }
    engine_reply["reversal"] = entries
    assert len(EngineReply.model_validate(engine_reply).reversal) == 618


@pytest.mark.parametrize(
    "reversal",
    [
        {"invalid": "value"},
        {"<REV_PERSON_0123456789abcdef_fedcba9876543210>": ""},
        {"<REV_PERSON_0123456789abcdef_fedcba9876543210>": "x" * 4_000_001},
    ],
    ids=["invalid-key", "empty-value", "oversized-value"],
)
def test_engine_reply_rejects_invalid_reversal_key_or_value(
    engine_reply: dict[str, object], reversal: dict[str, str]
) -> None:
    engine_reply["reversal"] = reversal
    with pytest.raises(ValidationError):
        EngineReply.model_validate(engine_reply)


def test_engine_reply_accepts_exact_four_million_character_reversal_value(
    engine_reply: dict[str, object],
) -> None:
    token = next(iter(cast(dict[str, str], engine_reply["reversal"])))
    engine_reply["reversal"] = {token: "x" * 4_000_000}
    assert len(next(iter(EngineReply.model_validate(engine_reply).reversal.values()))) == 4_000_000


def test_reversal_restoration_handles_618_entries_in_one_leaf_pass() -> None:
    reversal = {
        f"<REV_PERSON_{index:016x}_{index + 1:016x}>": f"value-{index}" for index in range(618)
    }
    original = "|".join(reversal.values())
    transformed = "|".join(reversal)
    entities = dict.fromkeys(reversal, "PERSON")

    seen, counts = _restore_reversal_leaves(
        {("messages", 0, "content"): original},
        {("messages", 0, "content"): transformed},
        reversal,
        entities,
    )

    assert seen == set(reversal)
    assert counts == {"PERSON": 618}


def test_cached_report_requires_a_cached_terminal_decision(
    engine_reply: dict[str, object],
) -> None:
    analysis = cast(dict[str, object], engine_reply["analysis"])
    analysis.update(
        {
            "source": "cached_decision",
            "scan_performed": False,
            "duration_ms": None,
            "cached_decision_applied": True,
        }
    )

    with pytest.raises(ValidationError, match="cached terminal decision"):
        EngineReply.model_validate(engine_reply)


@pytest.mark.parametrize(
    "update",
    [
        {"duration_ms": None},
        {"scan_performed": False},
        {"duration_ms": -1},
        {"overlap_count": -1},
        {"overlap_resolution": "first_match"},
        {"source": "cached_decision"},
        {"unexpected": True},
    ],
)
def test_engine_reply_rejects_malformed_analysis_metadata(
    engine_reply: dict[str, object], update: dict[str, object]
) -> None:
    analysis = cast(dict[str, object], engine_reply["analysis"])
    analysis.update(update)
    with pytest.raises(ValidationError):
        EngineReply.model_validate(engine_reply)


def test_headerless_model_session_keys_are_request_scoped() -> None:
    """Prompt content cannot become a cross-request conversation identifier."""
    request = EngineChatRequest(
        model="test", messages=[{"role": "user", "content": "first message"}]
    )
    first_policy = ModelDestinationPolicy(
        contract_version=1,
        destination_kind="model",
        principal_id="user-1",
        models={"test": True},
    )
    other_policy = first_policy.model_copy(update={"principal_id": "user-2"})
    first = make_session_key(first_policy, request, request_nonce=b"a" * 32)
    second = make_session_key(first_policy, request, request_nonce=b"b" * 32)
    other = make_session_key(other_policy, request, request_nonce=b"a" * 32)
    assert len(first) == 64
    assert first != second
    assert first != other
    assert "first message" not in first
    with pytest.raises(ValueError, match="conversation or request nonce"):
        make_session_key(first_policy, request)


def test_session_key_prefers_gateway_conversation_header() -> None:
    """LibreChat's stable conversation header survives changing prompt history."""
    first = EngineChatRequest(model="test", messages=[{"role": "user", "content": "one"}])
    second = EngineChatRequest(model="test", messages=[{"role": "user", "content": "two"}])
    policy = ModelDestinationPolicy(
        contract_version=1,
        destination_kind="model",
        principal_id="user-1",
        models={"test": True},
    )
    assert make_session_key(policy, first, conversation_id="conversation-1") == make_session_key(
        policy,
        second,
        conversation_id="conversation-1",
    )
