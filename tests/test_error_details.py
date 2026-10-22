import json
import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel, ValidationError

from agentgateway_extproc.config.settings import Settings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.documents import DocumentError
from agentgateway_extproc.lib.pipeline.diagnostics import exception_diagnostic, policy_block_message
from agentgateway_extproc.lib.pipeline.request import _image_policy_block
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.engine import EngineReply
from agentgateway_extproc.models.exceptions import InvalidEngineReplyError

from .conftest import body_request, header_request, mcp_headers, mcp_policy, request_json
from .test_admission import admission as admission
from .test_admission import body as model_body
from .test_admission import headers as model_headers
from .test_native_profiles import native as native


def test_error_level_is_standalone(monkeypatch):
    monkeypatch.delenv("ERROR_LOG_LEVEL", raising=False)
    monkeypatch.setenv("EXTPROC_LOG_LEVEL", "DEBUG")
    assert Settings().error_log_level == "info"
    monkeypatch.setenv("EXTPROC_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("ERROR_LOG_LEVEL", "DeBuG")
    assert Settings().error_log_level == "debug"
    assert Settings(error_log_level="INFO").error_log_level == "info"
    with pytest.raises(ValidationError):
        Settings(error_log_level="trace")


def block_reply(reply):
    reply.update(
        decision="block",
        applied_actions=["block"],
        remote_allowed=False,
        request=None,
        reversal={},
        safety_rule="sensitive_content",
    )
    reply["report"]["rows"][0].update(
        action="block", transformed_count=0, unique_transformed_count=0
    )


async def blocked_response(client, settings, headers, body, policy):
    async def requests():
        yield header_request(headers, policy=policy)
        yield body_request(body, policy=policy)

    replies = [
        reply async for reply in ExtProcServicer(client, settings).Process(requests(), object())
    ]
    return replies[-1].immediate_response


@pytest.mark.parametrize("level", ["info", "debug"])
@pytest.mark.parametrize("kind", ["chat", "responses", "mcp"])
async def test_policy_reason_survives_protocol_envelope(
    engine_client, engine_reply, caplog, level, kind, admission
):
    block_reply(engine_reply)
    caplog.set_level(logging.ERROR)
    policy = (
        mcp_policy()
        if kind == "mcp"
        else {
            "contract_version": 1,
            "destination_kind": "model",
            "principal_id": "principal-1",
            "models": {"test": True},
        }
    )
    if kind == "responses":
        policy = admission
        policy["request_context"]["protocol_profile"] = "openai-responses-v1"
    headers = {"content-type": "application/json"}
    body = request_json()
    if kind == "responses":
        body = model_body(responses=True)
        headers = model_headers(**{":path": "/bindings/openai-api/v1/responses"})
    if kind == "mcp":
        headers = mcp_headers()
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "tools/call",
                "params": {"name": "search", "arguments": {"query": "Jane Doe"}},
            }
        ).encode()
    immediate = await blocked_response(
        engine_client, Settings(error_log_level=level, log_level="ERROR"), headers, body, policy
    )
    payload = json.loads(immediate.body)
    error = payload["error"]
    assert immediate.status.code == (200 if kind == "mcp" else 403)
    assert "sensitive_content" in error["message"] and "PERSON" in error["message"]
    assert ("cached_decision_applied" in error["message"]) is (level == "debug")
    details = error["data"] if kind == "mcp" else error
    assert details["code"] == "policy_blocked"
    assert any(
        item.header.key == "x-request-id" and item.header.value == details["request_id"]
        for item in immediate.headers.set_headers
    )
    if kind == "mcp":
        assert payload["id"] == 0 and error["code"] == -32000
    assert "sensitive_content" in caplog.text and '"source": "current_request"' in caplog.text
    assert "Jane Doe" not in immediate.body + caplog.text


@pytest.mark.parametrize("level", ["info", "debug"])
async def test_native_policy_reason(engine_client, engine_reply, native, level):
    _, policy, headers, body = native
    block_reply(engine_reply)
    immediate = await blocked_response(
        engine_client, Settings(error_log_level=level), headers, json.dumps(body).encode(), policy
    )
    assert immediate.status.code == 403
    error = json.loads(immediate.body)["error"]
    assert error["code"] == "policy_blocked"
    assert "sensitive_content" in error["message"] and "PERSON" in error["message"]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("level", ["info", "debug"])
def test_missing_policy_reason_is_explicit(engine_client, engine_reply, cached, level, caplog):
    block_reply(engine_reply)
    engine_reply.update(
        safety_rule=None,
        entities=[],
        entity_counts={},
        report={"rows": []},
        notices={"request": [], "response": []},
    )
    if cached:
        engine_reply["analysis"].update(
            source="cached_decision",
            cached_decision_applied=True,
            scan_performed=False,
            duration_ms=None,
        )
    reply = EngineReply.model_validate(engine_reply)
    handler = StreamHandler(engine_client, Settings(error_log_level=level, log_level="ERROR"))
    caplog.set_level(logging.ERROR)
    message = policy_block_message(reply, handler)
    assert ("earlier session" in message) is cached
    assert ("original reason is unavailable" if cached else "reason not supplied") in message
    if level == "debug":
        assert f'"cached_decision_applied": {str(cached).lower()}' in message
        assert ('"source": "cached_decision"' in message) is cached
    assert '"cached_decision_applied"' in caplog.text
    reply.notices.response.append("Policy explanation from the Engine")
    assert "Policy explanation from the Engine" in policy_block_message(reply, handler)


@pytest.mark.parametrize("level", ["info", "debug"])
async def test_exception_chain_is_safe_at_error_level(engine_client, caplog, level):
    class Payload(BaseModel):
        count: int

    credential = "arbitrary-credential-sentinel"

    async def fail(self, request):
        try:
            Payload.model_validate({"count": credential})
        except ValidationError as exc:
            raise InvalidEngineReplyError("invalid reply for " + credential) from exc

    async def requests():
        yield header_request({"authorization": "Bearer " + credential})

    caplog.set_level(logging.ERROR)
    with patch.object(StreamHandler, "handle", fail):
        replies = [
            reply
            async for reply in ExtProcServicer(
                engine_client, Settings(error_log_level=level, log_level="ERROR")
            ).Process(requests(), object())
        ]
    immediate = replies[-1].immediate_response
    error = json.loads(immediate.body)["error"]
    assert immediate.status.code == 503 and error["code"] == "processing_failed"
    assert "ValidationError" in error["message"]
    for internal in ("traceback", '"file"', '"line"', '"function"', ".py"):
        assert internal not in error["message"]
    assert '"path": ["count"]' in caplog.text and "int_parsing" in caplog.text
    assert '"function": "fail"' in caplog.text
    assert "traceback" in caplog.text and error["request_id"] in caplog.text
    assert credential not in immediate.body + caplog.text
    assert '"input"' not in caplog.text


@pytest.mark.parametrize("level", ["info", "debug"])
@pytest.mark.parametrize("role", ["user", "string"])
async def test_request_validation_details_exclude_input(engine_client, caplog, level, role):
    caplog.set_level(logging.ERROR)
    raw = {
        "model": "test",
        "messages": [
            {
                "role": role,
                "content": {
                    "password": "validation-password-sentinel",
                    "private_text": "private-body-sentinel",
                },
            }
        ],
    }

    async def requests():
        yield header_request()
        yield body_request(json.dumps(raw).encode())

    replies = [
        reply
        async for reply in ExtProcServicer(
            engine_client, Settings(error_log_level=level, log_level="ERROR")
        ).Process(requests(), object())
    ]
    immediate = replies[-1].immediate_response
    error = json.loads(immediate.body)["error"]
    assert immediate.status.code == 400 and error["code"] == "invalid_request"
    assert "messages" in error["message"] and "content" in error["message"]
    if role == "string":
        assert "role" in error["message"] and "literal_error" in error["message"]
        assert "literal_error" in caplog.text
    assert error["request_id"] and error["request_id"] in caplog.text
    assert any(
        item.header.key == "x-request-id" and item.header.value == error["request_id"]
        for item in immediate.headers.set_headers
    )
    for internal in ("traceback", '"file"', '"line"', '"function"', ".py"):
        assert internal not in error["message"]
    assert "traceback" in caplog.text and "ValidationError" in caplog.text
    assert '"function"' in caplog.text and '"path"' in caplog.text
    assert "validation-password-sentinel" not in immediate.body + caplog.text
    assert "private-body-sentinel" not in immediate.body + caplog.text


@pytest.mark.parametrize("level", ["info", "debug"])
async def test_unknown_exception_cannot_echo_discarded_input(engine_client, caplog, level):
    private = "unlabelled private request excerpt"

    async def fail(self, request):
        self.request_body_chunks.clear()
        self.request_diagnostic = None
        raise RuntimeError(private)

    async def requests():
        yield header_request()

    caplog.set_level(logging.ERROR)
    with patch.object(StreamHandler, "handle", fail):
        replies = [
            reply
            async for reply in ExtProcServicer(
                engine_client, Settings(error_log_level=level, log_level="ERROR")
            ).Process(requests(), object())
        ]
    body = replies[-1].immediate_response.body
    assert "RuntimeError" in body + caplog.text
    assert '"function": "fail"' in caplog.text
    assert private not in body + caplog.text


def test_short_input_does_not_corrupt_diagnostics(engine_client, engine_reply):
    block_reply(engine_reply)
    handler = StreamHandler(engine_client, Settings(error_log_level="debug"))
    payload = {"model": "test", "messages": [{"role": "user", "content": "a"}]}
    handler.request_body_chunks.append(json.dumps(payload).encode())
    message = policy_block_message(
        EngineReply.model_validate(engine_reply), handler, input_data=payload
    )
    assert "safety rule sensitive_content" in message
    assert '"policy_version": "test"' in message
    assert "[redacted]" not in message

    class Payload(BaseModel):
        count: int

    with pytest.raises(ValidationError) as raised:
        Payload.model_validate({"count": "a"})
    detail = json.loads(exception_diagnostic(raised.value, handler))
    assert detail[0]["type"] == "ValidationError"
    assert detail[0]["validation"]["errors"] == [{"path": ["count"], "code": "int_parsing"}]
    handler.request_headers["x-api-key"] = "a"
    # A true short credential is still redacted in a single pass.
    assert "[red[" not in exception_diagnostic(raised.value, handler)


def test_face_block_preserves_specific_document_explanation(engine_client):
    handler = StreamHandler(engine_client)
    error = DocumentError(403, reason="face_policy_blocked")
    response = _image_policy_block(handler, error, detail="safety rule face_policy")
    payload = json.loads(response.immediate_response.body)
    assert response.immediate_response.status.code == 403
    assert payload["error"]["message"] == error.message + " safety rule face_policy"


async def test_nonvisual_image_block_preserves_document_explanation(engine_client, engine_reply):
    block_reply(engine_reply)
    policy: dict[str, object] = {
        "contract_version": 2,
        "destination_kind": "model",
        "principal_id": "principal-1",
        "models": {"test": True},
        "attachment_modes": {"test": "process"},
        "image_forwarding": {"test": "pii-unchecked"},
        "face_protection": {"test": False},
        "local_models": {"test": True},
    }
    docling = Mock(spec=DoclingClient)
    docling.convert = AsyncMock(return_value=["extracted image text"])
    handler = StreamHandler(engine_client, Settings(), docling)
    await handler.handle(header_request(policy=policy))
    payload = {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                ],
            }
        ],
    }
    with patch(
        "agentgateway_extproc.lib.pipeline.request._image_locations", return_value={(0, 0): "image"}
    ):
        response = await handler.handle(body_request(json.dumps(payload).encode(), policy=policy))
    assert response is not None and response.immediate_response.status.code == 403
    message = json.loads(response.immediate_response.body)["error"]
    assert message.startswith(DocumentError(403).message)
    assert "safety rule sensitive_content" in message
