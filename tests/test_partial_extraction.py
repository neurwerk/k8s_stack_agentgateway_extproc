"""Partial extraction must warn callers without bypassing document validation."""

import json
from unittest.mock import patch

import httpx
import pytest
from google.protobuf.json_format import MessageToDict

from agentgateway_extproc.config.settings import DoclingSettings, EngineSettings, Settings
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.documents import DocumentError, Upload
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.destination import DESTINATION_POLICY_NAMESPACE
from agentgateway_extproc.models.engine import EngineAttachmentPart

from .conftest import REVERSIBLE_TOKEN, body_request, header_request
from .test_documents import _document, _upload
from .test_typed_attachments_v4 import _typed_policy

WARNING = (
    "Warning: document extraction is incomplete. Acknowledge that content is "
    "missing and do not infer the missing content.\n\n"
)
ERROR = {
    "component_type": "model",
    "module_name": "synthetic",
    "error_message": "PRIVATE ERROR CONTENT MUST NOT BE FORWARDED",
}


def _transport(result, task_status="success"):
    def respond(request):
        return httpx.Response(
            200,
            json=result
            if request.url.path.startswith("/v1/result/")
            else {"task_id": "synthetic", "task_type": "convert", "task_status": task_status},
        )

    return httpx.MockTransport(respond)


@pytest.mark.parametrize("api,pii", [("chat", True), ("responses", False)])
async def test_partial_warning_reaches_model_and_policy(api, pii, engine_reply):
    calls = []

    def analyze(request):
        sent = json.loads(request.content)
        calls.append(sent)
        reply = {key: value for key, value in engine_reply.items() if key != "request"}
        reply.update(
            api_version="v2",
            segments=[
                {**segment, "text": segment["text"].replace("Jane Doe", REVERSIBLE_TOKEN)}
                for segment in sent["segments"]
            ],
        )
        return httpx.Response(200, json=reply)

    result = {
        "status": "partial_success",
        "errors": [ERROR],
        "document": {"json_content": _document()},
    }
    policy = _typed_policy(
        models={"test": pii},
        document_modes={"test": "extract-text"},
        face_protection={"test": True},
    )
    policy = MessageToDict(
        header_request(policy=policy).metadata_context.filter_metadata[DESTINATION_POLICY_NAMESPACE]
    )
    if api == "responses":
        policy["request_context"]["protocol_profile"] = "openai-responses-v1"
    field = "input" if api == "responses" else "messages"
    part = (
        {"type": "input_file", **_upload()}
        if api == "responses"
        else {"type": "file", "file": _upload()}
    )
    request = {"model": "test", field: [{"role": "user", "content": [part]}]}
    settings = DoclingSettings(enabled=True, base_url="https://docling.test", api_key="test-only")
    async with (
        httpx.AsyncClient(transport=_transport(result)) as http,
        httpx.AsyncClient(transport=httpx.MockTransport(analyze)) as engine_http,
    ):
        engine = EngineClient(EngineSettings(base_url="https://pii.test"), engine_http)
        client = DoclingClient(settings, http)
        try:
            handler = StreamHandler(engine, Settings(), client)
            await handler.handle(header_request(policy=policy))
            response = await handler.handle(
                body_request(json.dumps(request).encode(), policy=policy)
            )
            assert response is not None and response.HasField("request_body")
            forwarded = json.loads(response.request_body.response.body_mutation.body)
            text = forwarded[field][-1]["content"][0]["text"]
            assert text.startswith(WARNING + "Document: report.txt\n")
            assert (REVERSIBLE_TOKEN if pii else "Jane Doe") in text
            assert ERROR["error_message"] not in json.dumps(forwarded)
            if pii:
                assert calls
                assert any(WARNING in segment["text"] for segment in calls[0]["segments"])
        finally:
            await client.close()


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "task-partial",
        "errors-object",
        "errors-string",
        "errors-incomplete",
        "success-errors",
        "empty",
        "pages",
        "orphan",
        "table",
        "missing-json",
    ],
)
async def test_partial_result_still_requires_usable_valid_document(case):
    document = _document()
    result = {
        "status": "partial_success",
        "errors": [ERROR],
        "document": {"json_content": document},
    }
    malformed_errors = {
        "errors-object": {},
        "errors-string": ["private"],
        "errors-incomplete": [{"error_message": "private"}],
    }
    if case in malformed_errors:
        result["errors"] = malformed_errors[case]
    elif case == "success-errors":
        result["status"] = "success"
    elif case == "empty":
        for kind in ("texts", "tables", "pictures", "groups"):
            document[kind] = []
        document["body"]["children"] = document["furniture"]["children"] = []
    elif case == "pages":
        document["pages"] = {}
    elif case == "orphan":
        document["texts"].append({"self_ref": "#/texts/7", "text": "lost"})
    elif case == "table":
        document["tables"][0]["data"]["table_cells"][0]["end_col_offset_idx"] = 999
    elif case == "missing-json":
        result["document"] = {}
    settings = DoclingSettings(enabled=True, base_url="https://docling.test", api_key="test-only")
    task_status = "partial_success" if case == "task-partial" else "success"
    async with httpx.AsyncClient(transport=_transport(result, task_status)) as http:
        client = DoclingClient(settings, http)
        try:
            parts = [EngineAttachmentPart(type="file", file=_upload("pdf"))]
            with patch(
                "agentgateway_extproc.lib.docling.preflight",
                return_value=[Upload("synthetic.pdf", "pdf", b"synthetic", pages=1)],
            ):
                if case in {"valid", "task-partial"}:
                    texts = await client.convert(parts)
                    assert texts[0].startswith(WARNING)
                    assert ERROR["error_message"] not in texts[0]
                else:
                    with pytest.raises(DocumentError) as error:
                        await client.convert(parts)
                    assert error.value.status == 503
                    assert error.value.reason == "extraction_failed"
        finally:
            await client.close()
