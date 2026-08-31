"""Test fail-closed behavior at the gRPC stream boundary."""

from __future__ import annotations

import json

import grpc
import httpx
import pytest

from agentgateway_extproc.config.settings import EngineSettings, Settings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.models.exceptions import ENGINE_ERROR_CONTRACT

from .conftest import (
    MODEL_POLICY,
    REVERSIBLE_TOKEN,
    add_policy,
    body_request,
    header_request,
    request_json,
    response_body,
    response_headers,
)

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
    assert json.loads(immediate.body) == {
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
    assert immediate.body == '{"error":"internal processing error"}'
    assert upstream_text not in caplog.text


async def test_grpc_servicer_handles_trailers(engine_client) -> None:
    """Each buffered request phase receives exactly one matching response."""
    servicer = ExtProcServicer(engine_client)

    async def requests():
        yield header_request()
        yield body_request(request_json(), end_of_stream=False)
        yield add_policy(
            ext_proc_pb2.ProcessingRequest(
                request_trailers=ext_proc_pb2.HttpTrailers(trailers=ext_proc_pb2.HeaderMap())
            )
        )

    responses = [response async for response in servicer.Process(requests(), object())]
    assert [response.WhichOneof("response") for response in responses] == [
        "request_headers",
        "request_body",
        "request_trailers",
    ]
    mutation = responses[1].request_body.response.body_mutation.body
    assert mutation
    assert REVERSIBLE_TOKEN.encode() in mutation


async def test_request_trailer_metadata_change_fails_without_duplicate_phase_response(
    engine_client,
) -> None:
    servicer = ExtProcServicer(engine_client)
    changed_policy = {**MODEL_POLICY, "principal_id": "principal-2"}

    async def requests():
        yield header_request()
        yield body_request(request_json(), end_of_stream=False)
        yield add_policy(
            ext_proc_pb2.ProcessingRequest(
                request_trailers=ext_proc_pb2.HttpTrailers(trailers=ext_proc_pb2.HeaderMap())
            ),
            changed_policy,
        )

    responses = [response async for response in servicer.Process(requests(), object())]
    assert [response.WhichOneof("response") for response in responses] == [
        "request_headers",
        "request_body",
        "immediate_response",
    ]
    assert responses[-1].immediate_response.status.code == 503


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


async def test_missing_promised_request_trailers_fails_closed(engine_client) -> None:
    servicer = ExtProcServicer(engine_client)

    async def requests():
        yield header_request()
        yield body_request(request_json(), end_of_stream=False)

    responses = [response async for response in servicer.Process(requests(), object())]
    assert [response.WhichOneof("response") for response in responses] == [
        "request_headers",
        "request_body",
        "immediate_response",
    ]
    assert responses[-1].immediate_response.status.code == 400


async def test_grpc_servicer_flushes_buffered_json_before_response_trailers(engine_client) -> None:
    """Full-duplex JSON is emitted once, before its trailer acknowledgement."""
    servicer = ExtProcServicer(engine_client)

    async def requests():
        yield header_request()
        yield body_request(request_json())
        yield response_headers("application/json")
        yield response_body(
            b'{"choices":[{"index":0,"message":{"content":"answer"}}]}',
            end_of_stream=False,
        )
        yield add_policy(
            ext_proc_pb2.ProcessingRequest(
                response_trailers=ext_proc_pb2.HttpTrailers(
                    trailers={
                        "headers": [
                            {"key": "etag", "value": '"stale"'},
                            {"key": "content-md5", "value": "stale"},
                            {"key": "digest", "value": "sha-256=stale"},
                            {"key": "x-safe", "value": "preserved"},
                        ]
                    }
                )
            )
        )

    responses = [response async for response in servicer.Process(requests(), object())]
    assert sum(response.HasField("response_body") for response in responses) == 1
    assert responses[-2].HasField("response_body")
    assert responses[-1].HasField("response_trailers")
    assert list(responses[-1].response_trailers.header_mutation.remove_headers) == [
        "content-md5",
        "digest",
        "etag",
    ]


async def test_buffered_json_headers_are_emitted_only_after_validation(engine_client) -> None:
    """Invalid buffered JSON can still become a precommit immediate response."""
    servicer = ExtProcServicer(engine_client)

    async def invalid_requests():
        yield header_request()
        yield body_request(request_json())
        yield response_headers("application/json")
        yield response_body(b'{"choices":')

    invalid = [response async for response in servicer.Process(invalid_requests(), object())]
    assert not any(response.HasField("response_headers") for response in invalid)
    assert invalid[-1].HasField("immediate_response")

    async def valid_requests():
        yield header_request()
        yield body_request(request_json())
        yield response_headers("application/json")
        yield response_body(b'{"choices":[{"message":{"content":"answer"}}]}')

    valid = [response async for response in servicer.Process(valid_requests(), object())]
    kinds = [response.WhichOneof("response") for response in valid]
    assert kinds[-2:] == ["response_headers", "response_body"]


async def test_late_sse_failure_occurs_after_streaming_header_commit(engine_client) -> None:
    """SSE remains full duplex, so a later unsafe placeholder aborts after headers."""
    servicer = ExtProcServicer(engine_client)
    context = FakeAioContext()

    async def requests():
        yield header_request()
        yield body_request(request_json())
        yield response_headers("text/event-stream")
        yield response_body(b"data: <REV_UNKNOWN_0123456789abcdef_fedcba9876543210>\n\n")

    responses = []
    with pytest.raises(StreamAbortedError):
        async for response in servicer.Process(requests(), context):
            responses.append(response)
    kinds = [response.WhichOneof("response") for response in responses]
    assert kinds[-1] == "response_headers"
    assert "immediate_response" not in kinds
    assert context.code == grpc.StatusCode.INTERNAL
    assert context.details == "response processing failed closed"


async def test_normal_iterator_eof_finalizes_streaming_response(engine_client) -> None:
    """A clean gRPC iterator EOF flushes an unterminated-by-transport SSE response."""
    servicer = ExtProcServicer(engine_client)

    async def requests():
        yield header_request()
        yield body_request(request_json())
        yield response_headers("text/event-stream")
        yield response_body(b"data: hello\n\n", end_of_stream=False)

    responses = [response async for response in servicer.Process(requests(), object())]
    bodies = [response for response in responses if response.HasField("response_body")]
    assert len(bodies) == 2
    assert bodies[-1].response_body.response.body_mutation.streamed_response.end_of_stream


async def test_buffered_provider_output_overflow_fails_before_header_commit(
    engine_client, engine_reply
) -> None:
    plaintext = "x" * 1_500
    engine_reply["request"]["messages"][0]["content"] = REVERSIBLE_TOKEN
    engine_reply["reversal"] = {REVERSIBLE_TOKEN: plaintext}
    engine_reply["notices"] = {"request": [], "response": []}
    servicer = ExtProcServicer(engine_client, Settings(max_response_bytes=1_024))

    async def requests():
        yield header_request()
        yield body_request(
            json.dumps(
                {"model": "test", "messages": [{"role": "user", "content": plaintext}]}
            ).encode()
        )
        yield response_headers("application/json")
        yield response_body(
            json.dumps({"choices": [{"message": {"content": REVERSIBLE_TOKEN}}]}).encode()
        )

    responses = [response async for response in servicer.Process(requests(), object())]
    kinds = [response.WhichOneof("response") for response in responses]
    assert "response_headers" not in kinds
    assert kinds[-1] == "immediate_response"
    assert responses[-1].immediate_response.status.code == 503


@pytest.mark.parametrize("content_type", ["text/plain", "text/event-stream"])
async def test_streamed_provider_output_overflow_aborts_after_commit(
    engine_client, engine_reply, content_type: str
) -> None:
    plaintext = "x" * 1_500
    engine_reply["request"]["messages"][0]["content"] = REVERSIBLE_TOKEN
    engine_reply["reversal"] = {REVERSIBLE_TOKEN: plaintext}
    engine_reply["notices"] = {"request": [], "response": []}
    servicer = ExtProcServicer(engine_client, Settings(max_response_bytes=1_024))
    context = FakeAioContext()
    provider_body = REVERSIBLE_TOKEN.encode()
    if content_type == "text/event-stream":
        event = {"choices": [{"index": 0, "delta": {"content": REVERSIBLE_TOKEN}}]}
        provider_body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()

    async def requests():
        yield header_request()
        yield body_request(
            json.dumps(
                {"model": "test", "messages": [{"role": "user", "content": plaintext}]}
            ).encode()
        )
        yield response_headers(content_type)
        yield response_body(provider_body)

    responses = []
    with pytest.raises(StreamAbortedError):
        async for response in servicer.Process(requests(), context):
            responses.append(response)

    assert responses[-1].HasField("response_headers")
    assert not any(response.HasField("immediate_response") for response in responses)
    assert context.code == grpc.StatusCode.INTERNAL
