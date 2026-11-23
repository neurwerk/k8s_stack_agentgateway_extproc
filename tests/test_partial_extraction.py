"""Partial extraction must warn callers without bypassing document validation."""

import json
from unittest.mock import patch

import httpx
import pytest
from google.protobuf.json_format import MessageToDict

from agentgateway_extproc.config.settings import DoclingSettings, EngineSettings, Settings
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.documents import DocumentError, Upload, project_document
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.notice.preferences import NoticePreferences
from agentgateway_extproc.lib.pipeline.response import SseResponseProcessor, process_json_response
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.destination import DESTINATION_POLICY_NAMESPACE
from agentgateway_extproc.models.engine import EngineAttachmentPart
from agentgateway_extproc.models.exceptions import InvalidReversalError

from .conftest import (
    REVERSIBLE_TOKEN,
    body_request,
    header_request,
    response_body,
    response_headers,
)
from .openai_fixtures import event_bytes, response_payload, stream_events
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
        assert request.headers["Accept-Docling-Document-Version"] == "1.10.0"
        return httpx.Response(
            200,
            json=result
            if request.url.path.startswith("/v1/result/")
            else {"task_id": "synthetic", "task_type": "convert", "task_status": task_status},
        )

    return httpx.MockTransport(respond)


@pytest.mark.parametrize("api,pii", [("chat", True), ("responses", False)])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", ["partial", "complete", "structured", "tools"])
async def test_partial_warning_reaches_model_and_policy(api, pii, streaming, mode, engine_reply):
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
        "status": "success" if mode == "complete" else "partial_success",
        "errors": [] if mode == "complete" else [ERROR],
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
    request = {"model": "test", "stream": streaming, field: [{"role": "user", "content": [part]}]}
    request.update(_response_controls(api, mode))
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
            assert response is not None
            if mode in {"structured", "tools"}:
                assert response.HasField("immediate_response")
                assert response.immediate_response.status.code == 422
                assert not calls
                return
            assert response.HasField("request_body")
            forwarded = json.loads(response.request_body.response.body_mutation.body)
            text = forwarded[field][-1]["content"][0]["text"]
            prefix = "" if mode == "complete" else WARNING
            assert text.startswith(prefix + "Document: report.txt\n")
            assert (REVERSIBLE_TOKEN if pii else "Jane Doe") in text
            assert ERROR["error_message"] not in json.dumps(forwarded)
            if pii:
                assert calls
                assert any(prefix in segment["text"] for segment in calls[0]["segments"])
            handler.notice_preferences = NoticePreferences(notices_enabled=False)
            await _assert_response_warning(
                handler, api, streaming, policy, partial=mode != "complete"
            )
        finally:
            await client.close()


def _response_controls(api, mode):
    if mode == "structured":
        return (
            {"response_format": {"type": "json_object"}}
            if api == "chat"
            else {"text": {"format": {"type": "json_object"}}}
        )
    if mode == "tools":
        function = {"name": "lookup", "parameters": {"type": "object", "properties": {}}}
        return {
            "tools": [
                {"type": "function", "function": function}
                if api == "chat"
                else {"type": "function", **function}
            ]
        }
    return {}


async def _assert_response_warning(handler, api, streaming, policy, *, partial):
    warning = "Warning: document extraction is incomplete. Some document content is missing."
    await handler.handle(
        response_headers("text/event-stream" if streaming else "application/json", policy=policy)
    )
    if streaming:
        wire = b"".join(
            event_bytes(api, event) for event in stream_events(api, text="Model answer")
        )
        split = len(wire) // 2
        first = await handler.handle(
            response_body(wire[:split], end_of_stream=False, policy=policy)
        )
        last = await handler.handle(response_body(wire[split:], policy=policy))
        assert first is not None and last is not None
        output = (
            first.response_body.response.body_mutation.streamed_response.body
            + last.response_body.response.body_mutation.streamed_response.body
        ).decode()
        assert (warning in output) is partial
        if partial:
            assert output.index(warning) < output.rfind(
                "[DONE]" if api == "chat" else "response.completed"
            )
    else:
        response = await handler.handle(
            response_body(
                json.dumps(response_payload(api, text="Model answer")).encode(), policy=policy
            )
        )
        assert response is not None
        output = response.response_body.response.body_mutation.streamed_response.body.decode()
        assert output.count(warning) == int(partial)
    assert "Model answer" in output
    assert ERROR["error_message"] not in output


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize(
    "pages",
    [{"1": {"page_no": 1}, "3": {"page_no": 3}}, {}, {"4": {"page_no": 4}}, {"1": {"page_no": 2}}],
)
def test_partial_pdf_page_subset_preserves_bounds(partial, pages):
    document = _document()
    document["pages"] = pages
    upload = Upload("synthetic.pdf", "pdf", b"synthetic", pages=3, extraction_incomplete=partial)
    if partial and set(pages) == {"1", "3"}:
        text, count = project_document(document, upload, 3)
        assert "Jane Doe" in text
        assert count == 2
    else:
        with pytest.raises(DocumentError):
            project_document(document, upload, 3)


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


@pytest.mark.parametrize(
    "case,diagnostic",
    [("invalid-schema", "schema_unsupported"), ("empty-table", "projection_incomplete")],
)
def test_partial_projection_requires_supported_schema_and_substantive_text(case, diagnostic):
    document = _document()
    if case == "invalid-schema":
        document["version"] = "unsupported"
    else:
        for node in document["texts"]:
            node["text"] = ""
            node["label"] = "paragraph"
        for table in document["tables"]:
            for cell in table["data"]["table_cells"]:
                cell["text"] = ""
    upload = Upload("synthetic.pdf", "pdf", b"synthetic", pages=1, extraction_incomplete=True)
    with pytest.raises(DocumentError) as error:
        project_document(document, upload, 1)
    assert error.value.diagnostic == diagnostic


@pytest.mark.parametrize("streaming", [False, True])
def test_incomplete_model_response_keeps_mandatory_warning(engine_client, streaming):
    handler = StreamHandler(engine_client)
    handler.response_api_kind = "responses"
    handler.incomplete_extractions.add(0)
    handler.notice_preferences = NoticePreferences(notices_enabled=False)
    if streaming:
        events = stream_events("responses", text="Partial model answer")
        for event in events:
            if isinstance(event, dict) and event.get("type") == "response.completed":
                event["type"] = "response.incomplete"
                event["response"]["status"] = "incomplete"
                event["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
        output = SseResponseProcessor(handler).feed(
            b"".join(event_bytes("responses", event) for event in events).decode(), final=True
        )
    else:
        payload = response_payload("responses", text="Partial model answer")
        payload["status"] = "incomplete"
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
        output = process_json_response(handler, json.dumps(payload))
    assert "Partial model answer" in output
    assert "Some document content is missing." in output


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
def test_partial_response_without_safe_warning_carrier_is_rejected(engine_client, api, streaming):
    handler = StreamHandler(engine_client)
    handler.response_api_kind = api
    handler.incomplete_extractions.add(0)
    handler.notice_preferences = NoticePreferences(notices_enabled=False)
    with pytest.raises(InvalidReversalError, match="visible response warning"):
        if streaming:
            wire = b"".join(
                event_bytes(api, event) for event in stream_events(api, text="", tools=True)
            ).decode()
            SseResponseProcessor(handler).feed(wire, final=True)
        else:
            process_json_response(handler, json.dumps(response_payload(api, text="", tools=True)))


def test_failed_responses_stream_rejects_partial_extraction_without_admission(engine_client):
    handler = StreamHandler(engine_client)
    handler.response_api_kind = "responses"
    handler.incomplete_extractions.add(0)
    events = stream_events("responses", text="Unfinished answer")
    for event in events:
        if isinstance(event, dict) and event.get("type") == "response.completed":
            event["type"] = "response.failed"
            event["response"]["status"] = "failed"
    wire = b"".join(event_bytes("responses", event) for event in events).decode()
    with pytest.raises(InvalidReversalError, match="visible response warning"):
        SseResponseProcessor(handler).feed(wire, final=True)
