"""Exercise the document dispatch boundary and retained native-job admission."""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import subprocess
import threading
import zipfile
import zlib
from email import policy as email_policy
from email.parser import BytesParser
from unittest.mock import patch

import httpx
import pytest
from pypdf import PdfWriter

from agentgateway_extproc.config.settings import DoclingSettings, EngineSettings, Settings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.documents import _MIMES, MAX_CELLS, DocumentError
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.json_limits import MAX_JSON_DEPTH, MAX_JSON_TOKENS
from agentgateway_extproc.models.engine import EngineAttachmentPart

from .conftest import MODEL_POLICY, REVERSIBLE_TOKEN, body_request, header_request


def _document():
    return {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "name": "untrusted-name",
        "origin": {"uri": "https://raw.test/source", "filename": "untrusted-name"},
        "body": {"self_ref": "#/body", "children": [{"$ref": "#/groups/0"}]},
        "furniture": {"self_ref": "#/furniture", "children": [{"$ref": "#/texts/2"}]},
        "groups": [
            {
                "self_ref": "#/groups/0",
                "children": [
                    {"$ref": "#/texts/0"},
                    {"$ref": "#/tables/0"},
                    {"$ref": "#/pictures/0"},
                    {"$ref": "#/texts/4"},
                    {"$ref": "#/texts/5"},
                    {"$ref": "#/texts/6"},
                ],
            }
        ],
        "texts": [
            {"self_ref": "#/texts/0", "text": "Jane Doe", "orig": "raw-copy"},
            {"self_ref": "#/texts/1", "text": "Caption"},
            {"self_ref": "#/texts/2", "text": "Footer"},
            {"self_ref": "#/texts/3", "text": "Value"},
            {"self_ref": "#/texts/4", "label": "checkbox_selected", "text": ""},
            {"self_ref": "#/texts/5", "label": "checkbox_unselected", "text": ""},
            {"self_ref": "#/texts/6", "label": "checkbox_unselected", "text": "Optional"},
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "children": [{"$ref": "#/texts/3"}],
                "data": {
                    "num_rows": 1,
                    "num_cols": 2,
                    "table_cells": [
                        {
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": index,
                            "end_col_offset_idx": index + 1,
                            "text": text if index == 0 else "raw-cell-copy",
                            **({"ref": {"$ref": "#/texts/3"}} if index == 1 else {}),
                        }
                        for index, text in enumerate(("Name", "Value"))
                    ],
                },
            }
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "captions": [{"$ref": "#/texts/1"}],
                "image": {"uri": "data:image/png;base64,raw-image"},
            }
        ],
        "pages": {"1": {"page_no": 1, "image": {"uri": "raw-page"}}},
    }


def _upload(extension="txt", case="ok"):  # noqa: C901
    data = b"Jane Doe"
    if extension == "pdf":
        output = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        if case == "encrypted":
            writer.encrypt("test-password")
        writer.write(output)
        data = output.getvalue()
        if case == "pdf-expansion":
            compressor = zlib.compressobj()
            compressed = b"".join(compressor.compress(b"\x02" * 65_536) for _ in range(256))
            compressed += compressor.flush()
            data = (
                b"%PDF-1.5\n1 0 obj\n<< /Type /XRef /Root 2 0 R /Size 16777216 "
                b"/W [1 0 0] /Index [0 16777216] /Filter /FlateDecode /Length "
                + str(len(compressed)).encode()
                + b" >>\nstream\n"
                + compressed
                + b"\nendstream\nendobj\nstartxref\n9\n%%EOF\n"
            )
    if extension in {"docx", "xlsx", "pptx"}:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            names = [
                "[Content_Types].xml",
                "_rels/.rels",
                {
                    "docx": "word/document.xml",
                    "xlsx": "xl/workbook.xml",
                    "pptx": "ppt/presentation.xml",
                }[extension],
            ]
            if case == "zip-path":
                names.append("../escape")
            if case == "zip-part":
                names.pop()
            for name in names:
                archive.writestr(name, b"<document/>")
            if case == "zip-limit":
                archive.writestr("large", b"x" * (25 * 1_048_576 + 1))
            if extension == "xlsx":
                dimension = "A1:XFD1048576" if case == "xlsx-dimension" else "A1:B2"
                cell = "XFD1048576" if case == "xlsx-sparse" else "B2"
                merged = "A1:XFD1048576" if case == "xlsx-merge" else "A1:B1"
                worksheet = (
                    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                    f'<dimension ref="{dimension}"/><sheetData><row r="1"><c r="A1"><v>1</v></c>'
                    f'</row><row r="2"><c r="{cell}"><v>2</v></c></row></sheetData>'
                    f'<mergeCells><mergeCell ref="{merged}"/></mergeCells></worksheet>'
                )
                if case == "xlsx-doctype":
                    worksheet = (
                        '<!DOCTYPE worksheet [<!ENTITY secret SYSTEM "file:///not-read">]>'
                        + worksheet
                    )
                if case == "xlsx-hyperlink":
                    worksheet = worksheet.replace(
                        "</worksheet>",
                        '<hyperlinks><hyperlink ref="A1:XFD1048576" location="A1"/>'
                        "</hyperlinks></worksheet>",
                    )
                if case == "xlsx-repeated-hyperlinks":
                    worksheet = worksheet.replace("B2", "A2").replace("B1", "A1")
                    worksheet = worksheet.replace(
                        "</worksheet>",
                        "<hyperlinks>"
                        + '<hyperlink ref="A1:A100000" location="A1"/>' * 1000
                        + "</hyperlinks></worksheet>",
                    )
                archive.writestr("xl/worksheets/sheet1.xml", worksheet.encode("utf-16"))
                if case == "xlsx-chart":
                    archive.writestr(
                        "xl/charts/chart1.xml",
                        '<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart">'
                        "<c:f>Sheet1!A1:XFD1048576</c:f></c:chartSpace>",
                    )
        data = output.getvalue()
    if case == "nul":
        data = b"private\x00text"
    if case == "csv-columns":
        data = b"," * (MAX_CELLS + 1)
    if case == "csv-area":
        data = b"," * 1000 + b"\nsingle" * 100
    if case == "long-string":
        data = b"a" * 1_048_576
    return {
        "filename": f"/private/path/report.{extension}",
        "file_data": f"data:{_MIMES[extension]};base64,{base64.b64encode(data).decode()}",
    }


@pytest.mark.parametrize(
    "api,pii,mode,extension,case,status",
    [
        (api, pii, mode, extension, "ok", 0)
        for api, pii, mode in [
            ("chat", True, "cpu"),
            ("chat", False, "remote"),
            ("responses", True, "remote"),
            ("responses", False, "cpu"),
        ]
        for extension in ("pdf", "docx", "xlsx", "pptx", "txt", "md", "csv")
    ]
    + [
        ("chat", True, "cpu", extension, case, status)
        for extension, case, status in [
            ("txt", "base64", 400),
            ("txt", "mime", 400),
            ("txt", "url", 400),
            ("txt", "extra-ref", 400),
            ("txt", "nul", 400),
            ("txt", "filename", 400),
            ("txt", "file-limit", 413),
            ("txt", "total-limit", 413),
            ("txt", "count", 413),
            ("pdf", "encrypted", 400),
            ("pdf", "page-limit", 413),
            ("pdf", "magic", 400),
            ("pdf", "pdf-expansion", 413),
            ("docx", "zip-path", 400),
            ("xlsx", "zip-part", 400),
            ("xlsx", "xlsx-sparse", 413),
            ("xlsx", "xlsx-dimension", 413),
            ("xlsx", "xlsx-merge", 413),
            ("xlsx", "xlsx-doctype", 400),
            ("xlsx", "xlsx-chart", 413),
            ("xlsx", "xlsx-hyperlink", 413),
            ("xlsx", "xlsx-repeated-hyperlinks", 413),
            ("csv", "csv-columns", 413),
            ("csv", "csv-area", 413),
            ("pptx", "zip-limit", 413),
            ("txt", "partial", 503),
            ("txt", "errors", 503),
            ("txt", "empty", 503),
            ("txt", "formula-empty", 503),
            ("txt", "formula-missing", 503),
            ("txt", "input-json-nodes", 413),
            ("txt", "input-json-depth", 413),
            ("txt", "input-json-unterminated", 400),
            ("txt", "docling-json-nodes", 503),
            ("txt", "pii-json-nodes", 503),
            ("txt", "long-string", 0),
            ("txt", "cycle", 503),
            ("txt", "unresolved", 503),
            ("txt", "orphan", 503),
            ("txt", "bad-table", 503),
            ("txt", "unknown", 503),
            ("txt", "response-limit", 503),
            ("txt", "text-limit", 413),
            ("txt", "wire-limit", 413),
            ("txt", "pii-control", 503),
            ("txt", "pii-reversal", 503),
            ("txt", "pii-block", 403),
            ("txt", "pii-pass", 0),
            ("txt", "pii-reroute", 0),
        ]
    ]
    + [("responses", False, "cpu", "txt", case, 413) for case in ("text-limit", "wire-limit")],
)
async def test_document_dispatch(engine_reply, api, pii, mode, extension, case, status):  # noqa: C901
    policy = {**MODEL_POLICY, "models": {"test": pii}, "attachment_modes": {"test": "extract"}}
    settings = Settings(
        docling=DoclingSettings(
            enabled=True,
            api_key="test-only",
            base_url="https://docling.test",
            inference_mode=mode,
            file_bytes=2 if case == "file-limit" else 20 * 1_048_576,
            total_bytes=10 if case == "total-limit" else 40 * 1_048_576,
            count=1 if case == "count" else 5,
            pages=1 if case == "page-limit" else 200,
            max_response_bytes=1024 if case == "response-limit" else 16 * 1_048_576,
        )
    )
    first = _upload(extension)
    second = _upload(extension, case)
    if case == "base64":
        second["file_data"] += "!"
    if case == "mime":
        second["file_data"] = second["file_data"].replace("text/plain", "application/pdf")
    if case == "url":
        second["file_data"] = "https://raw.test/file"
    if case == "extra-ref":
        second["file_id"] = "private-id"
    if case == "filename":
        second["filename"] = "private\nname.txt"
    if case == "magic":
        second["file_data"] = "data:application/pdf;base64,bm90LXBkZg=="
    parts = [
        {"type": "file", "file": value} if api == "chat" else {"type": "input_file", **value}
        for value in (first, second)
    ]
    field = "messages" if api == "chat" else "input"
    messages = [
        {"role": "assistant", "content": [parts[0]]},
        {"role": "user", "content": [parts[1]]},
    ]
    if api == "chat":
        messages[0].update(reasoning_content="opaque", reasoning_signature="signature")
    original = {"model": "test", field: messages, "stream": True, "temperature": 0.2}
    if api == "chat":
        original["stream_options"] = {"include_usage": True}
    doc = _document()
    if case == "empty":
        doc["texts"][0].pop("text")
    if case == "cycle":
        doc["groups"][0]["children"].append({"$ref": "#/body"})
    if case == "unresolved":
        doc["body"]["children"].append({"$ref": "#/texts/99"})
    if case == "orphan":
        doc["texts"].append({"self_ref": "#/texts/7", "text": "lost"})
    if case == "unknown":
        doc["form_items"] = [{"text": "cannot silently discard"}]
    if case == "bad-table":
        doc["tables"][0]["data"]["table_cells"][0]["end_col_offset_idx"] = 999
    if case in {"text-limit", "wire-limit"}:
        doc["texts"][0]["text"] = "x" * 4_000_001 if case == "text-limit" else "\t" * 2_700_000
    if case.startswith("formula-"):
        doc["texts"][1].update(label="formula", text="", orig="must-not-be-used")
        if case == "formula-missing":
            doc["texts"][1].pop("text")
    excessive_json = b"[" + b"{}," * (MAX_JSON_TOKENS // 2) + b"{}]"
    wire = json.dumps(original).encode()
    if case.startswith("input-json-"):
        value = (
            excessive_json
            if case.endswith("nodes")
            else b"[" * (MAX_JSON_DEPTH + 1) + b"0" + b"]" * (MAX_JSON_DEPTH + 1)
        )
        wire = wire.replace(json.dumps(second["file_data"]).encode(), value)
        if case.endswith("unterminated"):
            wire = b'{"file_data":"' + b'\\"' * 4096
    native_calls, engine_calls = [], []

    def native(request):
        native_calls.append(request.url.path)
        assert request.headers["x-api-key"] == "test-only"
        assert "authorization" not in request.headers
        if request.method == "POST":
            form = BytesParser(policy=email_policy.default).parsebytes(
                f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
                + request.content
            )
            fields = {
                part.get_param("name", header="content-disposition"): part
                for part in form.iter_parts()
            }
            assert (
                fields["files"].get_filename()
                == f"upload.{'md' if extension == 'txt' else extension}"
            )
            assert b"/private/path" not in request.content
            assert fields["pipeline"].get_payload() == ("standard" if mode == "cpu" else "vlm")
            assert fields["to_formats"].get_payload() == "json"
            assert fields["target_type"].get_payload() == "inbody"
            assert "ocr_lang" not in fields
            assert fields["abort_on_error"].get_payload() == "true"
            assert fields["include_images"].get_payload() == "false"
            assert fields["include_page_images"].get_payload() == "false"
            if mode == "cpu":
                assert fields["ocr_preset"].get_payload() == "rapidocr"
                assert fields["do_ocr"].get_payload() == "true"
                assert fields["do_table_structure"].get_payload() == "true"
            else:
                assert fields["vlm_pipeline_preset"].get_payload() == "default"
        if request.url.path.startswith("/v1/result/"):
            if case == "docling-json-nodes":
                return httpx.Response(200, content=excessive_json)
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "errors": ["private"] if case == "errors" else [],
                    "document": {"json_content": doc},
                },
            )
        return httpx.Response(
            200,
            json={
                "task_id": "safe-job",
                "task_type": "convert",
                "task_status": "pending"
                if request.method == "POST"
                else "partial_success"
                if case == "partial"
                else "success",
            },
        )

    def engine(request):
        engine_calls.append(request)
        assert request.url.path == "/v1/adapter/analyze-document-request"
        sent = json.loads(request.content)
        assert "file_data" not in request.content.decode()
        if case == "pii-json-nodes":
            return httpx.Response(200, content=excessive_json)
        transformed = copy.deepcopy(sent)
        for message in transformed[field]:
            for part in message["content"]:
                part["text"] = part["text"].replace("Jane Doe", REVERSIBLE_TOKEN)
        engine_reply["request"] = transformed
        engine_reply["entity_counts"] = {"PERSON": 2}
        engine_reply["report"]["rows"][0].update(detected_count=2, transformed_count=2)
        if case == "pii-control":
            transformed["temperature"] = 1.0
        if case == "pii-reversal":
            engine_reply["reversal"][REVERSIBLE_TOKEN] = "not in converted document"
        if case == "pii-block":
            engine_reply.update(
                decision="block",
                request=None,
                reversal={},
                applied_actions=["block"],
                remote_allowed=False,
            )
            engine_reply["report"]["rows"][0].update(
                action="block", transformed_count=0, unique_transformed_count=0
            )
        if case in {"pii-pass", "pii-reroute"}:
            decision = case.removeprefix("pii-")
            engine_reply.update(
                decision=decision,
                request=sent,
                reversal={},
                applied_actions=[decision],
                route_class="local" if decision == "reroute" else None,
                remote_allowed=decision != "reroute",
            )
            engine_reply["report"]["rows"][0].update(
                action=decision, transformed_count=0, unique_transformed_count=0
            )
        return httpx.Response(200, json=engine_reply)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(native)) as native_http,
        httpx.AsyncClient(transport=httpx.MockTransport(engine)) as engine_http,
    ):
        docling = DoclingClient(settings.docling, native_http)
        client = EngineClient(EngineSettings(base_url="https://pii.test"), engine_http)
        servicer = ExtProcServicer(client, settings, docling)

        async def requests():
            yield header_request(
                {"authorization": "not-forwarded", "x-session-id": "same"}, policy=policy
            )
            yield body_request(wire, policy=policy)

        with (
            patch("json.loads", wraps=json.loads) as decode,
            patch(
                "pypdf.PdfReader", side_effect=AssertionError("PDF inspection must run in a child")
            ),
        ):
            replies = [reply async for reply in servicer.Process(requests(), object())]
        if "json-" in case:
            rejected = wire if case.startswith("input-") else excessive_json
            assert all(
                call.args[0] not in (rejected, rejected.decode()) for call in decode.call_args_list
            )
        if status:
            assert replies[-1].immediate_response.status.code == status
            assert "private" not in replies[-1].immediate_response.body
            if (
                status == 400
                or case in {"file-limit", "total-limit", "count", "page-limit"}
                or case.startswith("zip-")
                or case.startswith(("xlsx-", "csv-", "pdf-", "input-json-"))
            ):
                assert not native_calls
            if not case.startswith("pii-"):
                assert not engine_calls
        else:
            forwarded = json.loads(replies[-1].request_body.response.body_mutation.body)
            output_messages = forwarded[field][1:] if pii and api == "chat" else forwarded[field]
            assert len(output_messages) == 2
            for message in output_messages:
                assert len(message["content"]) == 1
                text = message["content"][0]["text"]
                assert text.startswith(f"Document: report.{extension}\n")
                assert "Name | Value" in text and "Caption" in text and "Footer" in text
                assert "[x]\n[ ]\n[ ] Optional" in text
                protected = pii and case not in {"pii-pass", "pii-reroute"}
                assert (REVERSIBLE_TOKEN if protected else "Jane Doe") in text
            if api == "chat":
                assert output_messages[0]["reasoning_content"] == "opaque"
                assert output_messages[0]["reasoning_signature"] == "signature"
                assert forwarded["stream_options"] == {"include_usage": True}
            assert forwarded["temperature"] == 0.2 and forwarded["stream"] is True
            assert all(
                value not in json.dumps(forwarded)
                for value in (
                    "raw-copy",
                    "raw-cell-copy",
                    "raw-image",
                    "raw-page",
                    "raw.test",
                    "untrusted-name",
                )
            )
            assert len(engine_calls) == int(pii)
            assert native_calls.count("/v1/convert/file/async") == 2
            if pii and extension == "txt" and case == "ok":
                repeated = [reply async for reply in servicer.Process(requests(), object())]
                assert repeated[-1].HasField("request_body")
                assert len(engine_calls) == 2
                assert (
                    engine_calls[0].headers["x-pii-session-key"]
                    != engine_calls[1].headers["x-pii-session-key"]
                )
        await docling.close()


@pytest.mark.parametrize(
    "case",
    [
        "cancel",
        "deadline",
        "thread",
        "post-lost",
        "status-lost",
        "bad-id",
        "bad-type",
        "bad-status",
        "native-failure",
        "pdf-timeout",
    ],
)
async def test_document_admission_retains_native_jobs(case):  # noqa: C901
    entered, release = asyncio.Event(), asyncio.Event()
    thread_entered, thread_release = threading.Event(), threading.Event()
    calls = []

    async def native(request):
        calls.append(request.url.path)
        if request.method == "POST" and case == "post-lost":
            raise httpx.ReadError("private upstream content")
        if request.method == "GET":
            entered.set()
            await release.wait()
            if case == "status-lost":
                return httpx.Response(404, text="private upstream content")
        return httpx.Response(
            200,
            json={
                "task_id": "../unsafe" if case == "bad-id" else "safe-job",
                "task_type": "chunk" if case == "bad-type" else "convert",
                "task_status": "unknown"
                if case == "bad-status"
                else "pending"
                if request.method == "POST"
                else "failure"
                if case == "native-failure"
                else "success",
            },
        )

    def blocked_preflight(*args):
        thread_entered.set()
        thread_release.wait(timeout=5)
        return []

    settings = DoclingSettings(
        enabled=True,
        base_url="https://docling.test",
        api_key="test-only",
        timeout=0.2 if case == "deadline" else 5,
    )
    parts = [EngineAttachmentPart(type="file", file=_upload()) for _ in range(2)]
    if case == "pdf-timeout":
        parts = [EngineAttachmentPart(type="file", file=_upload("pdf"))]
        with patch(
            "agentgateway_extproc.lib.documents.subprocess.run",
            side_effect=subprocess.TimeoutExpired("fixed-command", 10),
        ) as run:
            async with httpx.AsyncClient(transport=httpx.MockTransport(native)) as http:
                client = DoclingClient(settings, http)
                with pytest.raises(DocumentError) as error:
                    await client.convert(parts)
                assert error.value.status == 413
                assert run.call_args.kwargs["timeout"] == 10
                assert run.call_args.kwargs["stderr"] == subprocess.DEVNULL
                assert run.call_args.kwargs["env"] == {}
                assert "shell" not in run.call_args.kwargs
                assert not calls and client._task is None
                await client.close()
        return
    async with httpx.AsyncClient(transport=httpx.MockTransport(native)) as http:
        client = DoclingClient(settings, http)
        preflight_patch = patch(
            "agentgateway_extproc.lib.docling.preflight", side_effect=blocked_preflight
        )
        if case == "thread":
            preflight_patch.start()
        try:
            caller = asyncio.create_task(client.convert(parts))
            if case in {"post-lost", "bad-id", "bad-type", "bad-status"}:
                with pytest.raises(DocumentError) as error:
                    await caller
                assert error.value.status == 503
            else:
                if case == "thread":
                    assert await asyncio.to_thread(thread_entered.wait, 2)
                else:
                    await asyncio.wait_for(entered.wait(), 2)
                with pytest.raises(DocumentError):
                    await client.convert(parts)
                retained = client._task
                assert retained is not None
                if case in {"cancel", "thread", "status-lost", "native-failure"}:
                    caller.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await caller
                else:
                    with pytest.raises(DocumentError) as error:
                        await caller
                    assert error.value.status == 504
                assert client._task is retained and not retained.done()
                with pytest.raises(DocumentError):
                    await client.convert(parts)
                release.set()
                thread_release.set()
                await asyncio.gather(retained, return_exceptions=True)
                await asyncio.sleep(0)
                assert client._task is None
            assert len([path for path in calls if path == "/v1/convert/file/async"]) <= 1
            assert not any(path.startswith("/v1/result/") for path in calls)
            poisoned = case in {"post-lost", "status-lost", "bad-id", "bad-type", "bad-status"}
            assert client._poisoned is poisoned
            if poisoned:
                with pytest.raises(DocumentError):
                    await client.convert(parts)
        finally:
            thread_release.set()
            release.set()
            if case == "thread":
                preflight_patch.stop()
            await client.close()
        with pytest.raises(DocumentError):
            await client.convert(parts)
    owned = DoclingClient(settings)
    assert owned._client is not None
    await owned.close()
    assert owned._client.is_closed
