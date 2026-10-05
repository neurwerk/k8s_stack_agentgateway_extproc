"""Keep measured size failures strict and content-free across the engine boundary."""

from __future__ import annotations

import json
from typing import override

import httpx
import pytest

from agentgateway_extproc.config.settings import EngineSettings
from agentgateway_extproc.lib import json_limits
from agentgateway_extproc.lib.engine.client import EngineClient, _parse_engine_error, _read_bounded
from agentgateway_extproc.models.engine import EngineChatRequest
from agentgateway_extproc.models.exceptions import EnginePolicyError, InvalidEngineReplyError


def _error():
    return {
        "api_version": "v2",
        "error": {
            "code": "request_too_large",
            "message": "The analysis request exceeds the configured size limit.",
            "retryable": False,
            "limit": {
                "component": "pii_engine",
                "stage": "admission",
                "reason": "bytes",
                "measured": 6,
                "maximum": 5,
                "unit": "bytes",
                "exact": False,
            },
        },
    }


def test_v2_limit_is_preserved_and_v1_still_accepted():
    payload = _error()
    error = _parse_engine_error(413, json.dumps(payload).encode(), correlation_id="test-id")
    assert error.status_code == 413
    assert error.limit is not None
    assert error.limit.model_dump() == payload["error"]["limit"]
    assert error.limit.component == "pii_engine"
    assert error.correlation_id == "test-id"
    payload["api_version"] = "v1"
    del payload["error"]["limit"]
    assert _parse_engine_error(413, json.dumps(payload).encode()).limit is None


async def test_engine_call_threads_correlation_header_and_rejection():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-correlation-id"] == "test-correlation"
        return httpx.Response(413, json=_error())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = EngineClient(EngineSettings(base_url="https://engine.test"), transport)
        with pytest.raises(EnginePolicyError) as caught:
            await client.analyze_request(
                EngineChatRequest.model_validate(
                    {
                        "model": "test",
                        "messages": [{"role": "user", "content": "sample"}],
                    }
                ),
                "a" * 64,
                correlation_id="test-correlation",
            )
    assert caught.value.correlation_id == "test-correlation"
    assert caught.value.limit is not None
    assert caught.value.limit.measured == 6


@pytest.mark.parametrize(
    "invalid",
    [
        {"measured": True},
        {"exact": "false"},
        {"maximum": -1},
        {"reason": "request-content"},
        {"preview": "private content"},
        {"measured": 5},
        {"unit": "characters"},
    ],
)
def test_untrusted_limit_fields_fail_closed(invalid):
    payload = _error()
    payload["error"]["limit"].update(invalid)
    with pytest.raises(InvalidEngineReplyError):
        _parse_engine_error(413, json.dumps(payload).encode())


def test_v2_limits_are_status_and_version_bound():
    payload = _error()
    with pytest.raises(InvalidEngineReplyError):
        _parse_engine_error(400, json.dumps(payload).encode())
    payload["api_version"] = "v1"
    with pytest.raises(InvalidEngineReplyError):
        _parse_engine_error(413, json.dumps(payload).encode())


def test_json_budget_records_only_observed_prefix(monkeypatch):
    monkeypatch.setattr(json_limits, "MAX_JSON_TOKENS", 2)
    with pytest.raises(json_limits.JsonBudgetError) as caught:
        json_limits.bounded_json_text('["private",1,2]')
    assert caught.value.limit is not None
    assert caught.value.limit.model_dump() == {
        "component": "extproc",
        "stage": "json",
        "reason": "tokens",
        "measured": 3,
        "maximum": 2,
        "unit": "items",
        "exact": False,
    }
    assert "private" not in str(caught.value)


async def test_engine_reply_byte_limit_stops_reading():
    consumed = 0

    class Stream(httpx.AsyncByteStream):
        @override
        async def __aiter__(self):
            nonlocal consumed
            for chunk in [b"123", b"456", b"unread"]:
                consumed += 1
                yield chunk

    with pytest.raises(InvalidEngineReplyError) as caught:
        await _read_bounded(httpx.Response(200, stream=Stream()), 5)
    assert consumed == 2
    assert caught.value.limit is not None
    assert caught.value.limit.model_dump() == {
        "component": "extproc",
        "stage": "engine_response",
        "reason": "decoded_bytes",
        "measured": 6,
        "maximum": 5,
        "unit": "bytes",
        "exact": False,
    }
