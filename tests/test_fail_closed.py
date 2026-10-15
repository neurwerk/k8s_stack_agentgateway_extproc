"""Fail-closed Engine errors at the gRPC boundary."""

from __future__ import annotations

import json

import grpc
import httpx
import pytest

from agentgateway_extproc.config.settings import EngineSettings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.models.exceptions import ENGINE_ERROR_CONTRACT

from .conftest import body_request, header_request, request_json

ERROR_CASES = tuple((code, *contract) for code, contract in ENGINE_ERROR_CONTRACT.items())


class StreamAbortedError(Exception):
    pass


class FakeAioContext:
    def __init__(self) -> None:
        """Capture one safe abort status for assertions."""
        self.code: grpc.StatusCode | None = None
        self.details = ""

    async def abort(self, code: grpc.StatusCode, details: str = "") -> None:
        self.code = code
        self.details = details
        raise StreamAbortedError


async def test_grpc_servicer_returns_503_on_engine_failure() -> None:
    """Unexpected engine transport errors become an immediate 503 response."""

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = EngineClient(
        EngineSettings(base_url="https://pii-engine.test"),
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://pii-engine.test"
        ),
    )
    servicer = ExtProcServicer(client)

    async def requests():
        yield header_request()
        yield body_request(request_json())

    responses = [response async for response in servicer.Process(requests(), object())]
    assert responses[-1].immediate_response.status.code == 503


@pytest.mark.parametrize(("code", "status", "message", "retryable"), ERROR_CASES)
async def test_grpc_servicer_propagates_valid_typed_engine_errors(
    code: str, status: int, message: str, retryable: bool, caplog
) -> None:
    """Validated rejections retain fixed statuses and OpenAI-compatible safe fields."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={
                "api_version": "v1",
                "error": {"code": code, "message": message, "retryable": retryable},
            },
            request=request,
        )

    client = EngineClient(
        EngineSettings(base_url="https://pii-engine.test"),
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://pii-engine.test"
        ),
    )
    servicer = ExtProcServicer(client)

    async def requests():
        yield header_request()
        yield body_request(request_json())

    responses = [response async for response in servicer.Process(requests(), object())]
    immediate = responses[-1].immediate_response
    assert immediate.status.code == status
    payload = json.loads(immediate.body)
    request_id = payload["error"].pop("request_id")
    assert len(request_id) == 32
    assert any(
        item.header.key == "x-request-id" and item.header.value == request_id
        for item in immediate.headers.set_headers
    )
    assert payload == {
        "error": {
            "message": message,
            "type": "pii_engine_error",
            "param": None,
            "code": code,
            "retryable": retryable,
        }
    }
    assert immediate.headers.set_headers[0].header.value == "application/json"
    assert message not in caplog.text


async def test_grpc_servicer_does_not_forward_invalid_engine_error_body(caplog) -> None:
    """Malformed upstream failures become a generic 503 without payload or log leakage."""
    upstream_text = "private upstream error body"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": upstream_text}, request=request)

    client = EngineClient(
        EngineSettings(base_url="https://pii-engine.test"),
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://pii-engine.test"
        ),
    )
    servicer = ExtProcServicer(client)

    async def requests():
        yield header_request()
        yield body_request(request_json())

    responses = [response async for response in servicer.Process(requests(), object())]
    immediate = responses[-1].immediate_response
    assert immediate.status.code == 503
    assert json.loads(immediate.body)["error"]["code"] == "processing_failed"
    assert json.loads(immediate.body)["error"]["message"] == "internal processing error"
    assert upstream_text not in caplog.text


async def test_header_only_request_fails_closed(engine_client) -> None:
    """A header-only LLM request cannot bypass engine analysis."""
    servicer = ExtProcServicer(engine_client)

    async def requests():
        yield header_request(end_of_stream=True)

    responses = [response async for response in servicer.Process(requests(), object())]
    assert responses[-1].immediate_response.status.code == 400


async def test_incomplete_request_stream_fails_closed(engine_client) -> None:
    """Closing the gRPC stream before the request body produces a denial."""
    servicer = ExtProcServicer(engine_client)

    async def requests():
        yield header_request()

    responses = [response async for response in servicer.Process(requests(), object())]
    assert responses[-1].immediate_response.status.code == 400
