from __future__ import annotations

import asyncio
import base64
import io
import json
import subprocess
import sys
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest
from google.protobuf.json_format import MessageToDict
from pydantic import SecretStr
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from agentgateway_extproc.config.settings import DoclingSettings, Settings
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.documents import (
    NATIVE_PDF_NOTICE,
    DocumentError,
    ImageBatch,
    Upload,
    extract_native_pdf,
    project_document,
)
from agentgateway_extproc.lib.pipeline.response import _render_handler_notice
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.destination import DESTINATION_POLICY_NAMESPACE
from agentgateway_extproc.models.engine import EngineAttachmentPart, EngineReply

from .conftest import (
    MODEL_POLICY,
    REVERSIBLE_TOKEN,
    body_request,
    header_request,
    response_body,
    response_headers,
)
from .openai_fixtures import event_bytes, response_payload, stream_events
from .test_documents import _document, _upload
from .test_images import image_part


@pytest.fixture(autouse=True)
def _darwin_worker_limits():
    # macOS rejects RLIMIT_AS; Linux CI executes the unchanged production command.
    if sys.platform != "darwin":
        yield
        return
    original_run = subprocess.run

    def run(command, **kwargs):
        if "agentgateway_extproc.lib.pdf_probe" in command:
            wrapper = (
                "import resource,runpy,sys; original=resource.setrlimit; "
                "resource.setrlimit=lambda kind,value: None if kind==resource.RLIMIT_AS "
                "else original(kind,value); sys.argv=['pdf_probe']+sys.argv[1:]; "
                "runpy.run_module('agentgateway_extproc.lib.pdf_probe',run_name='__main__')"
            )
            command = [*command[:3], "-c", wrapper, *command[5:]]
        return original_run(command, **kwargs)

    with patch("agentgateway_extproc.lib.documents.subprocess.run", side_effect=run):
        yield


def _pdf(*, text="Jane Doe", encrypted=False, pages=1):
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for _ in range(pages):
        page = writer.add_blank_page(width=100, height=100)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 10 Tf 10 50 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    # An actual embedded bitmap with a marker that must never reach any model.
    from pypdf.generic import NumberObject

    image = DecodedStreamObject()
    image.set_data(b"VISUAL-ONLY-DO-NOT-FORWARD")
    image.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(8),
            NameObject("/Height"): NumberObject(1),
            NameObject("/BitsPerComponent"): NumberObject(8),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
        }
    )
    resources = writer.pages[0]["/Resources"]
    assert isinstance(resources, DictionaryObject)
    resources[NameObject("/XObject")] = DictionaryObject(
        {NameObject("/Im0"): writer._add_object(image)}
    )
    if encrypted:
        writer.encrypt("test-only")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _part(data):
    return EngineAttachmentPart.model_validate(
        {
            "type": "file",
            "file": {
                "filename": "example.pdf",
                "file_data": "data:application/pdf;base64," + base64.b64encode(data).decode(),
            },
        }
    )


def _settings(**kwargs):
    return DoclingSettings(enabled=True, api_key=SecretStr("test-only"), **kwargs)


async def test_native_pdf_reads_all_pages_without_docling_or_vision():
    client = DoclingClient(_settings(inference_mode="remote"))
    notices = []
    try:
        with patch.object(client, "_http", new_callable=AsyncMock) as http:
            texts = await client.convert(
                [_part(_pdf(pages=84))], native_pdf=True, pdf_notices=notices
            )
        http.assert_not_awaited()
        assert texts[0].count("Jane Doe") == 84
        assert "[Page 84]" in texts[0]
        assert "VISUAL-ONLY" not in texts[0]
        assert notices == [NATIVE_PDF_NOTICE]
    finally:
        await client.close()


@pytest.mark.parametrize("case", ["empty", "encrypted", "malformed", "pages", "bytes"])
async def test_native_pdf_rejects_unreadable_or_over_limit_without_fallback(case):
    data = _pdf(text="" if case == "empty" else "Jane Doe", encrypted=case == "encrypted")
    if case == "malformed":
        data = b"%PDF-1.7\ninvalid\n%%EOF"
    client = DoclingClient(_settings(pages=1) if case == "pages" else _settings())
    if case == "pages":
        data = _pdf(pages=2)
    if case == "bytes":
        client = DoclingClient(_settings(file_bytes=1))
    try:
        with (
            patch.object(client, "_http", new_callable=AsyncMock) as http,
            pytest.raises(DocumentError) as exc,
        ):
            await client.convert([_part(data)], native_pdf=True)
        http.assert_not_awaited()
        if case == "empty":
            assert exc.value.status == 422
            assert "local vision model" in exc.value.message
    finally:
        await client.close()


@pytest.mark.parametrize("failure", ["timeout", "memory", "worker", "text"])
def test_native_worker_failures_do_not_return_partial_text(failure):
    result = subprocess.CompletedProcess(
        [], {"memory": 2, "worker": 1}.get(failure, 0), stdout=b"partial"
    )
    with patch("agentgateway_extproc.lib.documents.subprocess.run") as run:
        if failure == "timeout":
            run.side_effect = subprocess.TimeoutExpired("fixed-worker", 1)
        else:
            run.return_value = result
        with (
            patch(
                "agentgateway_extproc.lib.documents.MAX_TEXT", 1 if failure == "text" else 4_000_000
            ),
            pytest.raises(DocumentError),
        ):
            extract_native_pdf(
                Upload("example.pdf", "pdf", b"pdf", pages=1),
                1,
                threading.Event(),
                time.monotonic() + 1,
            )


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("pii", [False, True])
@pytest.mark.parametrize("responses", [False, True])
async def test_destination_controls_pdf_route_and_preserves_pii(
    engine_client, engine_reply, local, typed, pii, responses
):
    policy = {
        **MODEL_POLICY,
        "contract_version": 3,
        "models": {"test": pii},
        "attachment_modes": {"test": "extract"},
        "local_models": {"test": local},
    }
    if typed:
        policy.pop("attachment_modes")
        policy.update(
            contract_version=4,
            document_modes={"test": "extract-text"},
            image_modes={"test": "block"},
            image_forwarding={"test": "none"},
            face_protection={"test": True},
            image_models={"test": False},
        )
    data = _pdf()
    part = _part(data).model_dump()
    if responses:
        part = {"type": "input_file", **part["file"]}
    original = {
        "model": "test",
        "stream": typed,
        "input" if responses else "messages": [{"role": "user", "content": [part]}],
    }
    client = DoclingClient(_settings())
    handler = StreamHandler(engine_client, Settings(), client)
    analyzed = []

    async def analyze(request, _session, **kwargs):
        assert kwargs["document"] is True
        analyzed.append(request.model_dump())
        replacement = json.loads(
            json.dumps(request.model_dump(exclude_unset=True, by_alias=True)).replace(
                "Jane Doe", REVERSIBLE_TOKEN
            )
        )
        result = EngineReply.model_validate({**engine_reply, "request": replacement}, strict=True)
        assert result.request is not None
        result.request._controls = request._controls
        return result

    try:
        with (
            patch.object(
                client, "_convert_one", new_callable=AsyncMock, return_value=_document()
            ) as convert,
            patch.object(engine_client, "analyze_request", side_effect=analyze),
        ):
            if responses:
                metadata = header_request(policy=policy).metadata_context.filter_metadata[
                    DESTINATION_POLICY_NAMESPACE
                ]
                policy = MessageToDict(metadata)
                policy["request_context"]["protocol_profile"] = "openai-responses-v1"
            await handler.handle(header_request(policy=policy))
            reply = await handler.handle(body_request(json.dumps(original).encode(), policy=policy))
        assert reply is not None and reply.HasField("request_body")
        outgoing = reply.request_body.response.body_mutation.body.decode()
        assert "file_data" not in outgoing and base64.b64encode(data).decode() not in outgoing
        assert "image_url" not in outgoing and "VISUAL-ONLY" not in outgoing
        assert bool(analyzed) == pii
        if pii:
            assert "Jane Doe" in json.dumps(analyzed)
            assert REVERSIBLE_TOKEN in outgoing and "Jane Doe" not in outgoing
        else:
            assert "Jane Doe" in outgoing
        assert convert.await_count == int(local)
        assert bool(handler.document_notice_messages) != local
        if not local:
            assert "visual content" in _render_handler_notice(handler)
            kind = "responses" if responses else "chat"
            content_type = "text/event-stream" if typed else "application/json"
            await handler.handle(response_headers(content_type, policy=policy))
            response = (
                b"".join(event_bytes(kind, event) for event in stream_events(kind))
                if typed
                else json.dumps(response_payload(kind)).encode()
            )
            delivered = await handler.handle(response_body(response, policy=policy))
            assert delivered is not None and delivered.HasField("response_body")
            assert b"visual content" in delivered.SerializeToString()
    finally:
        await client.close()


def test_broken_docling_table_remains_rejected():
    document = _document()
    document["tables"][0]["data"] = {"num_rows": 0, "num_cols": 0, "table_cells": []}
    with pytest.raises(DocumentError):
        project_document(document, Upload("example.pdf", "pdf", b"", pages=1), 1)


@pytest.mark.parametrize("format", ["docx", "xlsx", "image"])
async def test_mixed_batch_preserves_non_pdf_extraction(format):
    other = image_part() if format == "image" else {"type": "file", "file": _upload(format)}
    parts = [_part(_pdf()), EngineAttachmentPart.model_validate(other)]
    client = DoclingClient(_settings())
    try:
        with (
            patch.object(
                client, "_convert_one", new_callable=AsyncMock, return_value=_document()
            ) as convert,
            patch("agentgateway_extproc.lib.documents._normalize_image", return_value=(b"png", 0)),
        ):
            texts = await client.convert(
                parts,
                native_pdf=True,
                images=ImageBatch(protect_faces=False, policy_version=4)
                if format == "image"
                else None,
            )
        assert len(texts) == 2 and all("Jane Doe" in text for text in texts)
        convert.assert_awaited_once()
        assert convert.call_args.args[0].format == ("img" if format == "image" else format)
    finally:
        await client.close()


async def test_cancelled_native_worker_keeps_admission_until_worker_finishes():
    client = DoclingClient(_settings())
    entered, release = threading.Event(), threading.Event()

    def worker(*args):
        entered.set()
        assert release.wait(3)
        return "native text"

    try:
        with patch("agentgateway_extproc.lib.docling.extract_native_pdf", side_effect=worker):
            task = asyncio.create_task(client.convert([_part(_pdf())], native_pdf=True))
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(DocumentError) as exc:
                await client.convert([_part(_pdf())], native_pdf=True)
            assert exc.value.reason == "extraction_unavailable"
            assert client._task is not None
            release.set()
            await asyncio.shield(client._task)
    finally:
        release.set()
        await client.close()
