"""Protect fail-closed extraction diagnostics and their privacy boundary."""

import base64
import json
import logging
import re
import threading
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agentgateway_extproc.config.settings import DoclingSettings, Settings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.documents import DocumentError, Upload, project_document
from agentgateway_extproc.lib.extraction_diagnostics import log_native_errors
from agentgateway_extproc.main import main

from .conftest import MODEL_POLICY, body_request, header_request


@pytest.mark.parametrize("stop_reason", [None, "length", "content_filter"])
@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_partial_result_is_rejected_with_correlated_safe_diagnostics(
    engine_client, caplog, stop_reason, level
):
    caplog.set_level(level, logger="agentgateway_extproc.lib.extraction_diagnostics")
    sentinels = ("private-filename.txt", "private-document-text", "private-upstream-error")
    settings = Settings(
        docling=DoclingSettings(
            enabled=True, api_key="private-credential", base_url="https://docling.test"
        )
    )

    def native(request):
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"task_id": "native-job", "task_type": "convert", "task_status": "success"},
            )
        return httpx.Response(
            200,
            json={
                "status": "partial_success",
                "errors": [
                    {
                        "component_type": "pipeline",
                        "module_name": sentinels[2],
                        "error_message": (
                            f"VLM output incomplete (stop_reason={stop_reason})."
                            if stop_reason
                            else sentinels[2]
                        ),
                        "category": "inference_failure",
                        "page_no": 1,
                        "max_tokens": 9876,
                        "headers": {"Authorization": "private-credential"},
                        "document": "private-extra-document",
                    }
                ],
                "document": {"json_content": {"name": sentinels[0], "text": sentinels[1]}},
            },
        )

    policy = {**MODEL_POLICY, "models": {"test": True}, "attachment_modes": {"test": "extract"}}
    body = json.dumps(
        {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "filename": sentinels[0],
                                "file_data": "data:text/plain;base64,"
                                + base64.b64encode(sentinels[1].encode()).decode(),
                            },
                        }
                    ],
                }
            ],
        }
    ).encode()

    async def requests():
        yield header_request({"content-type": "application/json"}, policy=policy)
        yield body_request(body, policy=policy)

    async with httpx.AsyncClient(transport=httpx.MockTransport(native)) as http:
        docling = DoclingClient(settings.docling, http)
        servicer = ExtProcServicer(engine_client, settings, docling)
        with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
            replies = [reply async for reply in servicer.Process(requests(), object())]
            analyze.assert_not_called()
        await docling.close()
    response = replies[-1].immediate_response
    assert response.status.code == 503
    error = json.loads(response.body)["error"]
    assert error["code"] == "processing_failed"
    assert "text extraction could not be completed" in error["message"]
    diagnostic = next(record for record in caplog.records if hasattr(record, "extraction_code"))
    assert diagnostic.extraction_code == "result_rejected"
    metadata = diagnostic.extraction_metadata.copy()
    assert re.fullmatch(r"[0-9a-f]{16}", metadata.pop("task_id_hash"))
    if stop_reason:
        assert metadata.pop("stop_reason") == (
            "length" if stop_reason == "length" else "content_filtered"
        )
        assert metadata.pop("error_code") == (
            "vlm_output_token_limit" if stop_reason == "length" else "vlm_content_filtered"
        )
        explanation = (
            "Document conversion stopped because the VLM output token limit was reached."
            if stop_reason == "length"
            else "Document conversion stopped because the VLM output was content-filtered."
        )
        assert f"Page 1: {explanation}" in error["message"]
        assert explanation in diagnostic.getMessage()
    else:
        assert error["message"].endswith("text extraction could not be completed.")
    assert metadata == {
        "native_status": "partial_success",
        "error_count": 1,
        "errors_present": True,
        "document_present": True,
        "category": "inference_failure",
        "page_no": 1,
    }
    assert diagnostic.request_id == error["request_id"]
    assert any(
        item.header.key == "x-request-id" and item.header.value == diagnostic.request_id
        for item in response.headers.set_headers
    )
    assert any(
        f"gateway rejection request_id={diagnostic.request_id}" in record.getMessage()
        for record in caplog.records
    )
    assert all(value not in response.body for value in (*sentinels, "private-credential"))
    assert all(
        value not in caplog.text
        for value in (*sentinels[:2], "private-credential", "private-extra-document")
    )
    if level == logging.DEBUG:
        assert sentinels[2] in caplog.text
        assert '"component_type": "pipeline"' in caplog.text
        assert '"page_no": 1' in caplog.text
    else:
        assert sentinels[2] not in caplog.text
    assert "9876" not in error["message"]


def test_native_debug_errors_are_bounded_and_escape_control_characters(caplog):
    caplog.set_level(logging.DEBUG, logger="agentgateway_extproc.lib.extraction_diagnostics")
    error = {
        "error_message": "unknown\n\r\x1b\u2028" + "x" * 3000 + "omitted-tail",
        "component_type": "c" * 200,
        "module_name": {"nested": "omitted-nested"},
        "category": "new_category",
        "page_no": True,
    }
    for malformed in (None, {}, "omitted-non-list"):
        log_native_errors(malformed, "job")
    log_native_errors([None, "omitted-entry", error] + [error] * 10, "private-task-id")
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    entries = json.loads(message.split(" errors=", 1)[1])
    assert len(entries) == 6
    assert entries[0]["error_message"].startswith("unknown\n\r\x1b\u2028")
    assert len(entries[0]["error_message"]) < 2100
    assert len(entries[0]["component_type"]) < 160
    assert entries[0]["category"] == "new_category"
    assert "page_no" not in entries[0]
    assert "module_name" not in entries[0]
    assert not any(
        value in message for value in ("omitted-", "private-task-id", "\n", "\r", "\x1b", "\u2028")
    )
    assert len(message) < 120000


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_debug_startup_warns_about_sensitive_native_errors(caplog, level):
    caplog.set_level(level, logger="agentgateway_extproc.lib.extraction_diagnostics")
    with (
        patch("agentgateway_extproc.main.get_settings", return_value=Settings()),
        patch("agentgateway_extproc.main._run", new_callable=AsyncMock),
    ):
        main()
    if level == logging.DEBUG:
        assert "upstream diagnostic errors may contain sensitive document data" in caplog.text
        assert "Disable DEBUG after troubleshooting" in caplog.text
        assert caplog.records[0].levelno == logging.WARNING
    else:
        assert "upstream diagnostic errors" not in caplog.text


async def test_failed_task_logs_native_message_only_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="agentgateway_extproc.lib.extraction_diagnostics")
    client = DoclingClient(DoclingSettings())
    with (
        patch.object(
            client,
            "_http",
            return_value={
                "task_id": "job",
                "task_type": "convert",
                "task_status": "failure",
                "error_message": "unknown task diagnostic",
                "document": "private-body",
            },
        ),
        pytest.raises(DocumentError) as caught,
    ):
        await client._convert_one(Upload("private-file.txt", "txt", b""), threading.Event())
    assert caught.value.diagnostic == "task_unsuccessful"
    assert "unknown task diagnostic" in caplog.text
    assert "unknown task diagnostic" not in caught.value.message
    assert "private-" not in caplog.text
    await client.close()


@pytest.mark.parametrize(
    ("message", "page"),
    [
        ("private-error", True),
        ("private-error VLM output incomplete (stop_reason=length).", 0),
        ("VLM output incomplete (stop_reason=length). private-error", -1),
        ("VLM output incomplete (stop_reason=private-error).", 1001),
        ({"private-error": "length"}, "1"),
        (["private-error"], 1.0),
        (None, None),
    ],
)
@pytest.mark.parametrize("known_stop", [False, True])
async def test_native_error_details_reject_untrusted_values(message, page, caplog, known_stop):
    client = DoclingClient(DoclingSettings())
    result = {
        "status": "partial_success",
        "errors": [
            {
                "error_message": (
                    "VLM output incomplete (stop_reason=length)." if known_stop else message
                ),
                "category": ["private-category"],
                "page_no": page,
                "stop_reason": "length",
                "error_code": "vlm_output_token_limit",
                "output_token_limit": 9876,
            }
        ],
        "document": {},
    }
    with (
        patch.object(
            client,
            "_http",
            side_effect=[
                {"task_id": "job", "task_type": "convert", "task_status": "success"},
                result,
            ],
        ),
        pytest.raises(DocumentError) as caught,
    ):
        await client._convert_one(Upload("private-file.txt", "txt", b""), threading.Event())
    error = caught.value
    assert (error.status, error.reason, error.diagnostic) == (
        503,
        "extraction_failed",
        "result_rejected",
    )
    expected_message = "neurwerk: text extraction could not be completed."
    expected_keys = {
        "task_id_hash",
        "native_status",
        "error_count",
        "errors_present",
        "document_present",
    }
    if known_stop:
        expected_message += (
            " Document conversion stopped because the VLM output token limit was reached."
        )
        expected_keys.update({"error_code", "stop_reason"})
    assert error.message == expected_message
    assert set(error.metadata) == expected_keys
    error.log("request-1")
    assert "private-" not in caplog.text + str(error)
    assert "9876" not in caplog.text + str(error)
    await client.close()


@pytest.mark.parametrize("case", ["schema", "pages"])
def test_projection_diagnostics_expose_only_support_flags_and_counts(case, caplog):
    document = {
        "schema_name": "private-schema" if case == "schema" else "DoclingDocument",
        "version": "private-version" if case == "schema" else "1.10.0",
        "pages": {"1": {"page_no": 1}, "2": {"page_no": 2}},
        "name": "private-filename.pdf",
    }
    with pytest.raises(DocumentError) as caught:
        project_document(document, Upload("private-filename.pdf", "pdf", b"", pages=1), 10)
    error = caught.value
    assert (error.status, error.reason) == (503, "extraction_failed")
    error.log("request-1")
    assert error.diagnostic == ("schema_unsupported" if case == "schema" else "page_count_mismatch")
    assert error.metadata == (
        {"schema_name_supported": False, "schema_version_supported": False}
        if case == "schema"
        else {"expected_pages": 1, "actual_pages": 2}
    )
    assert "private-" not in caplog.text + str(error)
