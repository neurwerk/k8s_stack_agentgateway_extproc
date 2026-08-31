"""Test strict engine transport and response validation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast, override
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from agentgateway_extproc.config.settings import EngineSettings
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.models.engine import EngineChatRequest
from agentgateway_extproc.models.exceptions import (
    ENGINE_ERROR_CONTRACT,
    MAX_ENGINE_ERROR_MESSAGE_LENGTH,
    EnginePolicyError,
    EngineUnavailableError,
    InvalidEngineReplyError,
)

ERROR_CASES = tuple((code, *contract) for code, contract in ENGINE_ERROR_CONTRACT.items())
MAX_ENGINE_RESPONSE_BYTES = 10_485_760


class ChunkedResponseStream(httpx.AsyncByteStream):
    """Yield controlled decoded HTTP response chunks without Content-Length."""

    def __init__(self, content: bytes, chunk_size: int = 1_048_576) -> None:
        """Store response bytes and the deterministic transport chunk size."""
        self.content = content
        self.chunk_size = chunk_size

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield each configured response chunk."""
        for offset in range(0, len(self.content), self.chunk_size):
            yield self.content[offset : offset + self.chunk_size]


def _client_returning(status: int, content: bytes) -> EngineClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=content, request=request)

    return EngineClient(
        EngineSettings(base_url="https://pii-engine.test"),
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://pii-engine.test"
        ),
    )


def _client_streaming(status: int, content: bytes) -> EngineClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            stream=ChunkedResponseStream(content),
            request=request,
        )

    return EngineClient(
        EngineSettings(base_url="https://pii-engine.test"),
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://pii-engine.test"
        ),
    )


def _padded_json(payload: dict[str, object], size: int) -> bytes:
    content = json.dumps(payload, separators=(",", ":")).encode()
    assert len(content) <= size
    return content + b" " * (size - len(content))


def test_engine_readiness_deadline_is_bounded_below_probe_timeout() -> None:
    """Reject readiness deadlines that can outlast the Kubernetes HTTP probe."""
    with pytest.raises(ValidationError):
        EngineSettings(readiness_timeout=1.01)


def test_engine_client_loads_mtls_identity_into_verification_context() -> None:
    """Use one SSL context so HTTPX presents the client certificate reliably."""
    context = MagicMock()
    settings = EngineSettings(
        base_url="https://pii-engine.test",
        ca_cert="ca.crt",
        client_cert="tls.crt",
        client_key="tls.key",
    )

    with (
        patch("agentgateway_extproc.lib.engine.client.ssl.create_default_context") as create,
        patch("agentgateway_extproc.lib.engine.client.httpx.AsyncClient") as async_client,
    ):
        create.return_value = context
        EngineClient(settings)

    create.assert_called_once_with(cafile="ca.crt")
    context.load_cert_chain.assert_called_once_with(certfile="tls.crt", keyfile="tls.key")
    assert async_client.call_args.kwargs["verify"] is context
    assert "cert" not in async_client.call_args.kwargs


async def test_engine_client_posts_typed_request(engine_client: EngineClient) -> None:
    """A valid engine response is parsed into the typed reply model."""
    result = await engine_client.analyze_request(
        EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
        "a" * 64,
    )
    assert next(iter(result.reversal.values())) == "Jane Doe"


@pytest.mark.parametrize(("code", "status", "message", "retryable"), ERROR_CASES)
async def test_engine_client_maps_valid_typed_errors(
    code: str, status: int, message: str, retryable: bool
) -> None:
    """Every contract code produces only its fixed status-bound domain error."""
    content = json.dumps(
        {
            "api_version": "v1",
            "error": {"code": code, "message": message, "retryable": retryable},
        }
    ).encode()

    with pytest.raises(EnginePolicyError) as raised:
        await _client_returning(status, content).analyze_request(
            EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
            "a" * 64,
        )

    assert raised.value.status_code == status
    assert raised.value.code == code
    assert raised.value.message == message
    assert raised.value.retryable is retryable
    assert raised.value.args == ()


async def test_engine_client_rejects_obsolete_overlap_error() -> None:
    content = json.dumps(
        {
            "api_version": "v1",
            "error": {
                "code": "ambiguous_entity_spans",
                "message": "The analysis produced ambiguous entity spans.",
                "retryable": False,
            },
        }
    ).encode()

    with pytest.raises(InvalidEngineReplyError):
        await _client_returning(422, content).analyze_request(
            EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
            "a" * 64,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "api_version": "v2",
            "error": {
                "code": "invalid_request",
                "message": "Invalid request.",
                "retryable": False,
            },
        },
        {
            "api_version": "v1",
            "error": {
                "code": "invalid_request",
                "message": "The analysis request is invalid.",
                "retryable": False,
            },
            "unexpected": True,
        },
        {
            "api_version": "v1",
            "error": {
                "code": "invalid_request",
                "message": "Invalid request.",
                "retryable": False,
                "unexpected": True,
            },
        },
        {
            "api_version": "v1",
            "error": {
                "code": "undocumented_error",
                "message": "Invalid request.",
                "retryable": False,
            },
        },
        {
            "api_version": "v1",
            "error": {
                "code": "invalid_request",
                "message": "Invalid request.",
                "retryable": 0,
            },
        },
    ],
    ids=["wrong-version", "outer-field", "error-field", "unknown-code", "non-strict-boolean"],
)
async def test_engine_client_rejects_unrecognized_error_contracts(
    payload: dict[str, object],
) -> None:
    """Unknown fields, codes, and coerced values cannot become client errors."""
    client = _client_returning(400, json.dumps(payload).encode())

    with pytest.raises(InvalidEngineReplyError):
        await client.analyze_request(
            EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
            "a" * 64,
        )


@pytest.mark.parametrize(
    "message",
    [
        "x" * (MAX_ENGINE_ERROR_MESSAGE_LENGTH + 1),
        "Unsafe\nmessage",
        " padded message ",
        "unsafe\u202eformat",
    ],
    ids=["oversized", "control", "surrounding-space", "format-character"],
)
async def test_engine_client_rejects_unsafe_error_messages(message: str) -> None:
    """Only a bounded printable single-line engine message may cross the boundary."""
    content = json.dumps(
        {
            "api_version": "v1",
            "error": {
                "code": "invalid_request",
                "message": message,
                "retryable": False,
            },
        }
    ).encode()

    with pytest.raises(InvalidEngineReplyError):
        await _client_returning(400, content).analyze_request(
            EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
            "a" * 64,
        )


async def test_engine_client_rejects_malformed_error_json() -> None:
    """A non-JSON engine failure remains an invalid engine reply."""
    with pytest.raises(InvalidEngineReplyError):
        await _client_returning(400, b'{"api_version":').analyze_request(
            EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
            "a" * 64,
        )


async def test_engine_client_rejects_duplicate_error_keys() -> None:
    """Duplicate error fields cannot select a more favorable interpretation."""
    content = (
        b'{"api_version":"v1","error":{"code":"invalid_request",'
        b'"code":"internal_error","message":"Invalid request.","retryable":false}}'
    )

    with pytest.raises(InvalidEngineReplyError):
        await _client_returning(400, content).analyze_request(
            EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
            "a" * 64,
        )


async def test_engine_client_rejects_error_status_mismatch() -> None:
    """An engine error code cannot override its contract-defined HTTP status."""
    content = json.dumps(
        {
            "api_version": "v1",
            "error": {
                "code": "invalid_request",
                "message": "Invalid request.",
                "retryable": False,
            },
        }
    ).encode()

    with pytest.raises(InvalidEngineReplyError):
        await _client_returning(503, content).analyze_request(
            EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
            "a" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("message", "A different but printable message."), ("retryable", True)],
)
async def test_engine_client_rejects_error_contract_field_mismatch(
    field: str, value: object
) -> None:
    """Client-visible fields must match the fixed contract for the selected code."""
    error: dict[str, object] = {
        "code": "invalid_request",
        "message": "The analysis request is invalid.",
        "retryable": False,
    }
    error[field] = value
    content = json.dumps({"api_version": "v1", "error": error}).encode()

    with pytest.raises(InvalidEngineReplyError):
        await _client_returning(400, content).analyze_request(
            EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
            "a" * 64,
        )


async def test_engine_readiness_has_a_separate_overall_deadline() -> None:
    """A stalled readiness handshake cannot inherit the analysis timeout."""

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200, request=request)

    client = EngineClient(
        EngineSettings(
            base_url="https://pii-engine.test",
            timeout=45,
            readiness_timeout=0.01,
        ),
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://pii-engine.test"
        ),
    )
    with pytest.raises(EngineUnavailableError):
        await client.check_ready()


async def test_engine_client_sends_only_opaque_session_key(engine_reply: dict[str, object]) -> None:
    """Session state is selected through the trusted adapter-only header."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["session"] = request.headers["x-pii-session-key"]
        return httpx.Response(200, json=engine_reply, request=request)

    client = EngineClient(
        EngineSettings(base_url="https://pii-engine.test"),
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://pii-engine.test"
        ),
    )
    await client.analyze_request(
        EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
        "f" * 64,
    )
    assert captured == {"session": "f" * 64}


async def test_engine_client_rejects_unknown_reply_fields(engine_reply: dict[str, object]) -> None:
    """Undocumented engine response fields fail closed."""
    engine_reply["unexpected"] = True

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=engine_reply, request=request)

    client = EngineClient(
        EngineSettings(base_url="https://pii-engine.test"),
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://pii-engine.test"
        ),
    )
    with pytest.raises(InvalidEngineReplyError):
        await client.analyze_request(
            EngineChatRequest.model_construct(model="test", messages=[]), "a" * 64
        )


async def test_engine_client_rejects_duplicate_reversal_keys(
    engine_reply: dict[str, object],
) -> None:
    """Duplicate wire keys cannot collapse conflicting plaintext mappings."""
    reversal = cast(dict[str, str], engine_reply["reversal"])
    token = next(iter(reversal))
    encoded = json.dumps(engine_reply)
    original = json.dumps(engine_reply["reversal"])
    duplicate = f'{{{json.dumps(token)}:"Jane Doe",{json.dumps(token)}:"John Doe"}}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=encoded.replace(original, duplicate), request=request)

    client = EngineClient(
        EngineSettings(base_url="https://pii-engine.test"),
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://pii-engine.test"
        ),
    )
    with pytest.raises(InvalidEngineReplyError):
        await client.analyze_request(
            EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
            "a" * 64,
        )


@pytest.mark.parametrize("chunked", [False, True])
async def test_engine_response_accepts_exact_ten_mibibytes(
    engine_reply: dict[str, object], chunked: bool
) -> None:
    content = _padded_json(engine_reply, MAX_ENGINE_RESPONSE_BYTES)
    client = _client_streaming(200, content) if chunked else _client_returning(200, content)

    reply = await client.analyze_request(
        EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
        "a" * 64,
    )

    assert reply.api_version == "v1"


@pytest.mark.parametrize("chunked", [False, True])
async def test_engine_response_rejects_ten_mibibytes_plus_one_before_parsing(
    engine_reply: dict[str, object], chunked: bool
) -> None:
    content = _padded_json(engine_reply, MAX_ENGINE_RESPONSE_BYTES + 1)
    client = _client_streaming(200, content) if chunked else _client_returning(200, content)

    with (
        patch("agentgateway_extproc.lib.engine.client.json.loads") as loads,
        pytest.raises(InvalidEngineReplyError),
    ):
        await client.analyze_request(
            EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
            "a" * 64,
        )

    loads.assert_not_called()


@pytest.mark.parametrize("chunked", [False, True])
async def test_engine_error_body_accepts_exact_ten_mibibytes(chunked: bool) -> None:
    status, message, retryable = ENGINE_ERROR_CONTRACT["invalid_request"]
    content = _padded_json(
        {
            "api_version": "v1",
            "error": {
                "code": "invalid_request",
                "message": message,
                "retryable": retryable,
            },
        },
        MAX_ENGINE_RESPONSE_BYTES,
    )
    client = _client_streaming(status, content) if chunked else _client_returning(status, content)

    with pytest.raises(EnginePolicyError) as raised:
        await client.analyze_request(
            EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
            "a" * 64,
        )

    assert raised.value.code == "invalid_request"


@pytest.mark.parametrize("chunked", [False, True])
async def test_engine_error_body_rejects_ten_mibibytes_plus_one(chunked: bool) -> None:
    status, message, retryable = ENGINE_ERROR_CONTRACT["invalid_request"]
    content = _padded_json(
        {
            "api_version": "v1",
            "error": {
                "code": "invalid_request",
                "message": message,
                "retryable": retryable,
            },
        },
        MAX_ENGINE_RESPONSE_BYTES + 1,
    )
    client = _client_streaming(status, content) if chunked else _client_returning(status, content)

    with pytest.raises(InvalidEngineReplyError):
        await client.analyze_request(
            EngineChatRequest(model="test", messages=[{"role": "user", "content": "hello"}]),
            "a" * 64,
        )
