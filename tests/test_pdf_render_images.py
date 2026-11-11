"""Exercise PDF overrides, ordered pixels, and independent local admission."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from neurwerk_request_segments import CompatibilitySettings, parse_request
from PIL import Image
from pydantic import SecretStr, ValidationError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject

from agentgateway_extproc.config.settings import DoclingSettings, Settings
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.documents import DocumentError, ImageBatch, Upload, preflight
from agentgateway_extproc.lib.local_attachments import LocalAttachments, render_pdf
from agentgateway_extproc.lib.pdf_render import render
from agentgateway_extproc.lib.pipeline.mcp_sources import source_marker
from agentgateway_extproc.lib.pipeline.request import _converted_unchecked_request
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.destination import (
    DESTINATION_POLICY_NAMESPACE,
    ModelDestinationPolicy,
)
from agentgateway_extproc.models.engine import EngineAttachmentPart, VisualFindings
from agentgateway_extproc.models.exceptions import EnginePolicyError

from .conftest import body_request, header_request
from .test_documents import _document, _upload
from .test_images import image_part
from .test_typed_attachments_v4 import _pass_reply, _v4_policy


def policy(**extra):
    return _v4_policy(
        **{
            "contract_version": "4.1",
            "document_modes": {"test": "extract-text"},
            "pdf_modes": {"test": "render-images"},
            "image_inspection": {"test": "document-only"},
            "image_textless": {"test": "block"},
            **extra,
        }
    )


def pdf_part(pages=2, *, encrypted=False, dimension=100, chat=True):
    writer = PdfWriter()
    for index in range(pages):
        page = writer.add_blank_page(width=dimension, height=dimension)
        stream = DecodedStreamObject()
        color = "1 0 0" if index % 2 == 0 else "0 0 1"
        stream.set_data(f"{color} rg 0 0 {dimension} {dimension} re f".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("")
    output = io.BytesIO()
    writer.write(output)
    value = {
        "filename": "pages.pdf",
        "file_data": "data:application/pdf;base64," + base64.b64encode(output.getvalue()).decode(),
    }
    return {"type": "file", "file": value} if chat else {"type": "input_file", **value}


@pytest.fixture
def real_children(monkeypatch):
    # macOS does not implement RLIMIT_AS. Keep real isolated parser/render children;
    # Linux CI exercises their address-space limits as well as the other limits.
    if sys.platform != "darwin":
        return
    run = subprocess.run

    def portable(command, **kwargs):
        module = command[command.index("-m") + 1]
        code = (
            "import resource,runpy; original=resource.setrlimit; "
            "resource.setrlimit=lambda key,value: None if key==resource.RLIMIT_AS "
            "else original(key,value); "
            f"runpy.run_module({module!r},run_name='__main__')"
        )
        return run(
            [*command[: command.index("-m")], "-c", code, *command[command.index("-m") + 2 :]],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "run", portable)


def test_sparse_override_and_old_contract_rejection():
    default = policy(pdf_modes={})
    validated = ModelDestinationPolicy.model_validate(default)
    assert not validated.pdf_modes
    assert "pdf_modes" not in validated.model_dump()
    for version in (1, 2, 3, 4):
        with pytest.raises(ValidationError):
            ModelDestinationPolicy.model_validate(policy(contract_version=version, pdf_modes={}))
    for mapping in ({"unknown": "block"}, {"test": "passthrough"}):
        with pytest.raises(ValidationError):
            ModelDestinationPolicy.model_validate(policy(pdf_modes=mapping))


@pytest.mark.parametrize(
    "extra",
    [
        {"models": {"test": True}},
        {"document_modes": {"test": "block"}},
        {"local_models": {"test": False}},
        {"image_models": {"test": False}},
        {"image_modes": {"test": "extract-text"}},
        {"image_forwarding": {"test": "none"}},
    ],
)
def test_render_policy_requires_all_invariants(extra):
    with pytest.raises(ValidationError):
        ModelDestinationPolicy.model_validate(policy(**extra))


async def dispatch(
    engine_client, docling, parts, *, chat=True, selected_policy=None, history=None, settings=None
):
    selected_policy = selected_policy or policy()
    text = "text" if chat else "input_text"
    payload = {
        "model": "test",
        "messages" if chat else "input": [
            *(history or []),
            {
                "role": "user",
                "content": [
                    {"type": text, "text": "before"},
                    *parts,
                    {"type": text, "text": "after"},
                ],
            },
        ],
    }
    handler = StreamHandler(engine_client, settings or Settings(), docling)
    headers = header_request(
        {":path": "/v1/chat/completions" if chat else "/v1/responses"}, policy=selected_policy
    )
    body = body_request(json.dumps(payload).encode(), policy=selected_policy)
    if not chat:
        for message in (headers, body):
            message.metadata_context.filter_metadata[DESTINATION_POLICY_NAMESPACE][
                "request_context"
            ]["protocol_profile"] = "openai-responses-v1"
    await handler.handle(headers)
    return await handler.handle(body)


@pytest.mark.parametrize("chat", [True, False])
@pytest.mark.parametrize("poisoned", [True, False])
async def test_ordered_real_pdf_pages_without_http(engine_client, real_children, chat, poisoned):
    docling = DoclingClient(DoclingSettings())
    docling._poisoned = poisoned
    with patch.object(docling, "_http", new_callable=AsyncMock) as http:
        response = await dispatch(engine_client, docling, [pdf_part(chat=chat)], chat=chat)
    await docling.close()
    http.assert_not_awaited()
    assert response.HasField("request_body")
    payload = json.loads(response.request_body.response.body_mutation.body)
    content = payload["messages" if chat else "input"][0]["content"]
    assert len(content) == 4
    assert content[0]["text"] == "before" and content[-1]["text"] == "after"
    for part, color in zip(content[1:3], [(255, 0, 0), (0, 0, 255)], strict=True):
        uri = part["image_url"]["url"] if chat else part["image_url"]
        with Image.open(io.BytesIO(base64.b64decode(uri.partition(",")[2]))) as image:
            assert image.getpixel((50, 50)) == color
            assert image.mode == "RGB"
            assert not image.info


async def test_unchecked_image_also_ignores_poisoned_http(engine_client, real_children):
    docling = DoclingClient(DoclingSettings())
    docling._poisoned = True
    with patch.object(docling, "_http", new_callable=AsyncMock) as http:
        response = await dispatch(engine_client, docling, [image_part()])
    await docling.close()
    assert response.HasField("request_body")
    http.assert_not_awaited()


@pytest.mark.parametrize("version", [4, "4.1"])
async def test_image_only_batch_uses_attachment_count_not_pdf_budget(
    engine_client, real_children, version
):
    docling = DoclingClient(DoclingSettings(count=6))
    selected = _v4_policy() if version == 4 else policy(pdf_modes={})
    with patch.object(docling, "_http", new_callable=AsyncMock) as http:
        response = await dispatch(
            engine_client, docling, [image_part() for _ in range(6)], selected_policy=selected
        )
    await docling.close()
    http.assert_not_awaited()
    assert response.HasField("request_body")
    content = json.loads(response.request_body.response.body_mutation.body)["messages"][0][
        "content"
    ]
    assert sum(part["type"] == "image_url" for part in content) == 6


@pytest.mark.parametrize("pdf_mode", [None, "extract-text", "render-images"])
async def test_mixed_batch_retains_nonpdf_extraction(engine_client, real_children, pdf_mode):
    docling = DoclingClient(DoclingSettings(enabled=True, api_key=SecretStr("test-only")))
    formats = []

    async def extract(upload, abandoned):
        formats.append(upload.format)
        return _document()

    with patch.object(docling, "_convert_one", side_effect=extract):
        response = await dispatch(
            engine_client,
            docling,
            [pdf_part(1), {"type": "file", "file": _upload()}],
            selected_policy=policy(pdf_modes={} if pdf_mode is None else {"test": pdf_mode}),
        )
    await docling.close()
    assert response.HasField("request_body")
    assert formats == (["txt"] if pdf_mode == "render-images" else ["pdf", "txt"])
    assert b"Document: report.txt" in response.request_body.response.body_mutation.body


@pytest.mark.parametrize("mode", ["extract-text", "native-first"])
async def test_pdf_extract_overrides_blocked_documents(engine_client, real_children, mode):
    docling = DoclingClient(DoclingSettings(enabled=True, api_key=SecretStr("test-only")))
    selected_policy = policy(
        document_modes={"test": "block"},
        pdf_modes={"test": mode},
        pdf_fallbacks={"test": "extract-text"} if mode == "native-first" else {},
    )
    with patch.object(docling, "_convert_one", return_value=_document()) as extract:
        good = await dispatch(
            engine_client, docling, [pdf_part(1)], selected_policy=selected_policy
        )
        bad = await dispatch(
            engine_client,
            docling,
            [pdf_part(1), {"type": "file", "file": _upload()}],
            selected_policy=selected_policy,
        )
    await docling.close()
    assert good.HasField("request_body")
    assert bad.immediate_response.status.code == 403
    assert extract.await_count == (2 if mode == "native-first" else 1)


@pytest.mark.parametrize(
    "case", ["encrypted", "malformed", "mime", "late-invalid", "six-pages", "pixels", "blocked"]
)
async def test_invalid_batches_fail_before_render_or_network(engine_client, real_children, case):
    part = pdf_part(
        6 if case == "six-pages" else 1,
        encrypted=case == "encrypted",
        dimension=6000 if case == "pixels" else 100,
    )
    if case == "malformed":
        part["file"]["file_data"] = (
            "data:application/pdf;base64," + base64.b64encode(b"%PDF-bad\n%%EOF").decode()
        )
    if case == "mime":
        part["file"]["file_data"] = part["file"]["file_data"].replace(
            "application/pdf", "text/plain"
        )
    parts = [part]
    if case == "late-invalid":
        parts.append({"type": "file", "file": {"filename": "bad.txt", "file_data": "bad"}})
    docling = DoclingClient(DoclingSettings(enabled=True, api_key=SecretStr("test-only")))
    selected_policy = policy(pdf_modes={"test": "block"}) if case == "blocked" else policy()
    with patch.object(docling, "_http", new_callable=AsyncMock) as http:
        response = await dispatch(engine_client, docling, parts, selected_policy=selected_policy)
    await docling.close()
    assert response.HasField("immediate_response")
    assert response.immediate_response.status.code in {400, 403, 413}
    http.assert_not_awaited()


async def test_page_count_includes_history_images(engine_client, real_children):
    docling = DoclingClient(DoclingSettings(count=5))
    good = await dispatch(engine_client, docling, [pdf_part(5)])
    assert good.HasField("request_body")
    with patch("agentgateway_extproc.lib.local_attachments.render_pdf") as renderer:
        bad = await dispatch(
            engine_client,
            docling,
            [pdf_part(5)],
            history=[{"role": "user", "content": [image_part()]}],
        )
    assert bad.immediate_response.status.code == 413
    renderer.assert_not_called()
    await docling.close()


@pytest.mark.parametrize("failure", ["exit", "invalid-json", "missing-page", "bad-png", "timeout"])
def test_child_failures_are_content_free(failure):
    result = SimpleNamespace(
        returncode=1 if failure == "exit" else 0, stdout=b"private-parser-output"
    )
    if failure == "missing-page":
        result.stdout = b"[]"
    if failure == "bad-png":
        result.stdout = b'["data:image/png;base64,cHJpdmF0ZQ=="]'
    effect = (
        subprocess.TimeoutExpired("fixed", 1, stderr=b"private-parser-output")
        if failure == "timeout"
        else None
    )
    with (
        patch(
            "agentgateway_extproc.lib.local_attachments.subprocess.run",
            return_value=result,
            side_effect=effect,
        ),
        pytest.raises(DocumentError) as error,
    ):
        render_pdf(Upload("x.pdf", "pdf", b"not logged", 1), threading.Event(), float("inf"))
    assert "private" not in str(error.value)


@pytest.mark.parametrize("cancel", [True, False])
async def test_local_admission_retained_until_worker_drains(real_children, cancel):
    local = LocalAttachments(DoclingSettings())
    if not cancel:
        local.settings = local.settings.model_copy(update={"timeout": 0.5})
    entered, release = threading.Event(), threading.Event()

    def held_render(*args):
        entered.set()
        assert release.wait(5)
        return ["unused"]

    part = EngineAttachmentPart.model_validate(pdf_part(1))
    with patch("agentgateway_extproc.lib.local_attachments.render_pdf", side_effect=held_render):
        task = asyncio.create_task(
            local.prepare([part], ImageBatch(protect_faces=False), "render-images")
        )
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            if cancel:
                task.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else DocumentError):
                await task
            with pytest.raises(DocumentError) as error:
                await local.prepare([part], ImageBatch(), "render-images")
            assert error.value.reason == "extraction_unavailable"
        finally:
            release.set()
            if local._task is not None:
                await asyncio.gather(local._task, return_exceptions=True)
    assert local._task is None
    local.settings = DoclingSettings()
    images = ImageBatch(protect_faces=False)
    assert await local.prepare([part], images, "render-images") == set()
    assert len(images.pdf_images[0]) == 1
    await local.close()


def test_renderer_canonical_pixel_limit():
    part = pdf_part(1, dimension=2000)
    data = base64.b64decode(part["file"]["file_data"].partition(",")[2])
    uri = render(data)[0]
    with Image.open(io.BytesIO(base64.b64decode(uri.partition(",")[2]))) as image:
        assert max(image.size) <= 2048 and image.width * image.height <= 2_000_000


async def test_combined_image_wire_budget_precedes_extraction(
    engine_client, real_children, monkeypatch
):
    part = pdf_part(1)
    pdf_uri = render(base64.b64decode(part["file"]["file_data"].partition(",")[2]))[0]
    image = image_part()
    local = LocalAttachments(DoclingSettings())
    batch = ImageBatch(protect_faces=False, policy_version=4)
    await local.prepare([EngineAttachmentPart.model_validate(image)], batch, None)
    budget = len(pdf_uri) + len(batch.images[0])
    await local.close()
    docling = DoclingClient(DoclingSettings(enabled=True, api_key=SecretStr("test-only")))
    parts = [part, image, {"type": "file", "file": _upload()}]
    with patch.object(docling, "_convert_one", return_value=_document()) as extract:
        monkeypatch.setattr(
            "agentgateway_extproc.lib.local_attachments.MAX_IMAGE_BYTES", budget - 1
        )
        bad = await dispatch(engine_client, docling, parts)
        assert bad.immediate_response.status.code == 413
        extract.assert_not_awaited()
        monkeypatch.setattr("agentgateway_extproc.lib.local_attachments.MAX_IMAGE_BYTES", budget)
        good = await dispatch(engine_client, docling, parts)
        assert good.HasField("request_body")
        extract.assert_awaited_once()
    await docling.close()


async def test_late_invalid_upload_never_starts_rendering(engine_client, real_children):
    docling = DoclingClient(DoclingSettings())
    with patch("agentgateway_extproc.lib.local_attachments.render_pdf") as renderer:
        response = await dispatch(
            engine_client,
            docling,
            [
                pdf_part(),
                {"type": "file", "file": {"filename": "bad.txt", "file_data": "bad"}},
            ],
        )
    await docling.close()
    assert response.immediate_response.status.code == 400
    renderer.assert_not_called()


@pytest.mark.parametrize("chat", [True, False])
async def test_expanded_locations_survive_source_analysis(source_engine, real_children, chat):
    client, calls = source_engine
    docling = DoclingClient(DoclingSettings())
    text = "text" if chat else "input_text"
    response = await dispatch(
        client,
        docling,
        [pdf_part(chat=chat), pdf_part(chat=chat)],
        chat=chat,
        history=[
            {
                "role": "user",
                "content": [
                    {"type": text, "text": source_marker(["MCP_TOOL_SALESFORCE"])},
                ],
            }
        ],
    )
    await docling.close()
    assert calls[0]["mcp_sources"] == ["MCP_TOOL_SALESFORCE"]
    assert not calls[0]["text_pii_enabled"]
    assert "data:image" not in json.dumps(calls)
    assert response.HasField("request_body")
    payload = json.loads(response.request_body.response.body_mutation.body)
    content = payload["messages" if chat else "input"][-1]["content"]
    assert len(content) == 6
    for part, color in zip(content[1:-1], [(255, 0, 0), (0, 0, 255)] * 2, strict=True):
        uri = part["image_url"]["url"] if chat else part["image_url"]
        with Image.open(io.BytesIO(base64.b64decode(uri.partition(",")[2]))) as image:
            assert image.getpixel((50, 50)) == color


async def test_forwarded_pdf_bytes_enforce_transformed_limit(engine_client, real_children):
    docling = DoclingClient(DoclingSettings())
    try:
        with pytest.raises(EnginePolicyError) as error:
            await dispatch(
                engine_client,
                docling,
                [pdf_part()],
                settings=Settings(max_transformed_request_bytes=1024),
            )
        assert error.value.limit is not None
        assert error.value.limit.reason == "transformed_bytes"
        assert error.value.limit.measured > 1024
    finally:
        await docling.close()


def test_expansion_preserves_reviewed_controls():
    controls = CompatibilitySettings.model_validate(
        {
            "controls": [
                {
                    "endpoint": "chat",
                    "location": "request",
                    "field": "vendor_mode",
                    "kinds": ["string"],
                    "enum": ["safe"],
                },
            ]
        }
    )
    payload = {
        "model": "test",
        "vendor_mode": "safe",
        "max_tokens": None,
        "messages": [{"role": "user", "content": [pdf_part()]}],
    }
    original = parse_request(payload, controls=controls)
    batch = ImageBatch(protect_faces=False, pdf_images={0: ["first", "second"]})
    request, _, locations = _converted_unchecked_request(
        payload, {}, batch, controls=original._controls
    )
    assert request.model_dump(exclude_unset=True)["vendor_mode"] == "safe"
    assert request.model_dump(exclude_unset=True)["max_tokens"] is None
    assert locations == {(0, 0): "first", (0, 1): "second"}


def native_document(pages=2):
    doc = _document()
    doc.update(groups=[], tables=[], pictures=[])
    doc["texts"] = [
        {
            "self_ref": f"#/texts/{index}",
            "text": f"Native page {index + 1}",
            "prov": [{"page_no": index + 1, "charspan": [0, len(f"Native page {index + 1}")]}],
        }
        for index in range(pages)
    ]
    doc["body"] = {
        "self_ref": "#/body",
        "children": [{"$ref": node["self_ref"]} for node in doc["texts"]],
    }
    doc["furniture"] = {"self_ref": "#/furniture", "children": []}
    doc["pages"] = {str(page): {"page_no": page} for page in range(1, pages + 1)}
    return doc


def incomplete_native(case):
    doc = native_document()
    if case in {"picture", "table"}:
        kind = "pictures" if case == "picture" else "tables"
        node = {"self_ref": f"#/{kind}/0", "prov": [{"page_no": 1}]}
        if case == "table":
            node["data"] = {
                "num_rows": 1,
                "num_cols": 1,
                "table_cells": [
                    {
                        "start_row_offset_idx": 0,
                        "end_row_offset_idx": 1,
                        "start_col_offset_idx": 0,
                        "end_col_offset_idx": 1,
                        "text": "cell",
                    }
                ],
            }
        doc[kind] = [node]
        doc["body"]["children"].append({"$ref": node["self_ref"]})
    elif case == "empty":
        doc["texts"] = []
        doc["body"]["children"] = []
    elif case == "missing-page":
        doc["texts"].pop()
        doc["body"]["children"].pop()
    elif case == "whitespace":
        doc["texts"][1]["text"] = " " * len(doc["texts"][1]["text"])
    elif case == "missing-prov":
        del doc["texts"][1]["prov"]
    elif case == "invalid-prov":
        doc["texts"][1]["prov"][0]["page_no"] = True
    elif case == "partial-span":
        doc["texts"][1]["prov"][0]["charspan"][1] -= 1
    elif case == "ambiguous-span":
        doc["texts"][0]["prov"].append(
            {"page_no": 2, "charspan": doc["texts"][0]["prov"][0]["charspan"]}
        )
    return doc


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "empty",
        "missing-page",
        "whitespace",
        "picture",
        "table",
        "missing-prov",
        "invalid-prov",
        "partial-span",
        "ambiguous-span",
    ],
)
async def test_checked_native_coverage_retries_original_without_rendering(real_children, case):
    docling = DoclingClient(DoclingSettings(enabled=True, api_key=SecretStr("test-only")))
    part = EngineAttachmentPart.model_validate(pdf_part())
    original = base64.b64decode(part.model_dump()["file"]["file_data"].partition(",")[2])
    uploads, calls = [], []

    async def convert(upload, abandoned, *, native=False, force_ocr=False):
        uploads.append(upload)
        calls.append((native, force_ocr, upload.data))
        return incomplete_native(case) if native else native_document()

    with (
        patch.object(docling, "_convert_one", side_effect=convert),
        patch("agentgateway_extproc.lib.docling.render_pdf") as render_mock,
    ):
        result = await docling.convert([part], pdf_mode="native-first", pdf_fallback="extract-text")
    await docling.close()
    assert "Native page 2" in result[0]
    assert calls == [(True, False, original)] + (
        [] if case == "complete" else [(False, True, original)]
    )
    assert all(upload.data == b"" for upload in uploads)
    render_mock.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("inference", ["internal-standard", "private-vlm"])
async def test_native_http_options_and_original_retry_bytes(real_children, enabled, inference):
    posts = []

    def transport(request):
        if request.method == "POST":
            posts.append(request.read())
            return httpx.Response(
                200,
                json={"task_id": str(len(posts)), "task_type": "convert", "task_status": "success"},
            )
        doc = incomplete_native("picture") if enabled and len(posts) == 1 else native_document()
        return httpx.Response(
            200, json={"status": "success", "errors": [], "document": {"json_content": doc}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        docling = DoclingClient(
            DoclingSettings(
                enabled=True,
                api_key=SecretStr("test-only"),
                inference_mode=inference,
            ),
            client,
        )
        part = EngineAttachmentPart.model_validate(pdf_part())
        result = await docling.convert(
            [part],
            pdf_mode="native-first" if enabled else "extract-text",
            pdf_fallback="extract-text" if enabled else None,
        )
        pdf_posts = list(posts)
        await docling.convert(
            [EngineAttachmentPart.model_validate({"type": "file", "file": _upload()})],
            pdf_mode="native-first",
            pdf_fallback="extract-text",
        )
        await docling.close()
    assert "Native page 2" in result[0]
    original = base64.b64decode(part.model_dump()["file"]["file_data"].partition(",")[2])
    assert all(original in body for body in pdf_posts)
    assert len(pdf_posts) == (2 if enabled else 1)
    if enabled:
        assert b'name="pipeline"\r\n\r\nnative' in posts[0]
        assert b'name="pdf_backend"\r\n\r\nthreaded_docling_parse' in posts[0]
        assert b'name="do_ocr"\r\n\r\nfalse' in posts[0]
        assert b"vlm_pipeline_preset" not in posts[0]
        assert b"force_ocr" not in posts[0]
    expected = b"standard" if inference == "internal-standard" else b"vlm"
    assert b'name="pipeline"\r\n\r\n' + expected in pdf_posts[-1]
    assert b'name="pipeline"\r\n\r\n' + expected in posts[-1]
    assert b"force_ocr" not in posts[-1]
    if enabled and inference == "internal-standard":
        assert b'name="force_ocr"\r\n\r\ntrue' in pdf_posts[-1]
    else:
        assert b"force_ocr" not in pdf_posts[-1]


@pytest.mark.parametrize("chat", [True, False])
async def test_native_local_mixed_results_preserve_full_order(engine_client, real_children, chat):
    docling = DoclingClient(DoclingSettings(enabled=True, api_key=SecretStr("test-only")))
    calls = []

    async def convert(upload, abandoned, *, native=False):
        calls.append((upload.format, native))
        return native_document(1) if len(calls) != 2 else incomplete_native("missing-page")

    text_file = {"type": "file", "file": _upload()} if chat else {"type": "input_file", **_upload()}
    with patch.object(docling, "_convert_one", side_effect=convert):
        response = await dispatch(
            engine_client,
            docling,
            [pdf_part(1, chat=chat), pdf_part(2, chat=chat), text_file],
            chat=chat,
            selected_policy=policy(
                pdf_modes={"test": "native-first"}, pdf_fallbacks={"test": "render-images"}
            ),
        )
    await docling.close()
    assert calls == [("pdf", True), ("pdf", True), ("txt", False)]
    assert response.HasField("request_body")
    content = json.loads(response.request_body.response.body_mutation.body)[
        "messages" if chat else "input"
    ][0]["content"]
    assert [part["type"] for part in content] == [
        "text" if chat else "input_text",
        "text" if chat else "input_text",
        "image_url" if chat else "input_image",
        "image_url" if chat else "input_image",
        "text" if chat else "input_text",
        "text" if chat else "input_text",
    ]
    assert "Document: pages.pdf" in content[1]["text"]
    assert "Document: report.txt" in content[4]["text"]


@pytest.mark.parametrize("mode", ["render-images", "native-first"])
@pytest.mark.parametrize("limit", [1, 20])
async def test_configured_render_budget_is_independent_of_upload_count(
    engine_client, real_children, limit, mode
):
    docling = DoclingClient(DoclingSettings(enabled=True, api_key=SecretStr("test-only")))
    selected = policy(
        pdf_modes={"test": mode},
        pdf_max_rendered_pages={"test": limit},
        pdf_fallbacks={"test": "render-images"} if mode == "native-first" else {},
    )

    async def empty(upload, abandoned, *, native=False):
        doc = native_document(upload.pages)
        doc["texts"] = []
        doc["body"]["children"] = []
        return doc

    with patch.object(docling, "_convert_one", side_effect=empty):
        good = await dispatch(engine_client, docling, [pdf_part(limit)], selected_policy=selected)
        bad = await dispatch(
            engine_client, docling, [pdf_part(limit), pdf_part(1)], selected_policy=selected
        )
        history = await dispatch(
            engine_client,
            docling,
            [pdf_part(limit)],
            selected_policy=selected,
            history=[{"role": "user", "content": [image_part()]}],
        )
    await docling.close()
    assert good.HasField("request_body")
    assert bad.immediate_response.status.code == 413
    assert history.immediate_response.status.code == 413


@pytest.mark.parametrize("limit", [0, 21, True, 1.5, "5"])
def test_render_budget_rejects_coercion_and_out_of_bounds(limit):
    with pytest.raises(ValidationError):
        ModelDestinationPolicy.model_validate(policy(pdf_max_rendered_pages={"test": limit}))


def test_checked_policy_cannot_enable_native_image_fallback():
    for extra in (
        {"models": {"test": True}},
        {"image_models": {"test": False}},
        {"local_models": {"test": False}},
    ):
        with pytest.raises(ValidationError):
            ModelDestinationPolicy.model_validate(
                policy(
                    pdf_modes={"test": "native-first"},
                    pdf_fallbacks={"test": "render-images"},
                    **extra,
                )
            )


@pytest.mark.parametrize("failure", ["http", "schema", "pages", "limit"])
async def test_native_failure_never_falls_back_and_clears_remaining_uploads(real_children, failure):
    docling = DoclingClient(DoclingSettings(enabled=True, api_key=SecretStr("test-only")))
    uploads = [
        Upload("one.pdf", "pdf", b"original-one", 2),
        Upload("two.pdf", "pdf", b"original-two", 2),
    ]
    doc = native_document()
    if failure == "schema":
        doc["version"] = "unknown"
    elif failure == "pages":
        doc["pages"].pop("2")
    elif failure == "limit":
        doc["texts"][0]["text"] = "x" * 4_000_001
    with (
        patch("agentgateway_extproc.lib.docling.preflight", return_value=uploads),
        patch.object(
            docling,
            "_convert_one",
            side_effect=DocumentError() if failure == "http" else None,
            return_value=doc,
        ) as convert,
        patch("agentgateway_extproc.lib.docling.render_pdf") as renderer,
        pytest.raises(DocumentError),
    ):
        await docling.convert(
            [EngineAttachmentPart.model_validate(pdf_part())] * 2,
            pdf_mode="native-first",
            pdf_fallback="extract-text",
        )
    await docling.close()
    assert convert.await_count == 1
    renderer.assert_not_called()
    assert all(upload.data == b"" for upload in uploads)


async def test_cancelled_native_job_drains_before_cleanup_without_fallback(real_children):
    docling = DoclingClient(DoclingSettings(enabled=True, api_key=SecretStr("test-only")))
    uploads = [Upload("one.pdf", "pdf", b"first", 2), Upload("two.pdf", "pdf", b"second", 2)]
    entered, release = asyncio.Event(), asyncio.Event()

    async def convert(upload, abandoned, *, native=False):
        entered.set()
        await release.wait()
        return incomplete_native("empty")

    with (
        patch("agentgateway_extproc.lib.docling.preflight", return_value=uploads),
        patch.object(docling, "_convert_one", side_effect=convert) as conversion,
    ):
        task = asyncio.create_task(
            docling.convert(
                [EngineAttachmentPart.model_validate(pdf_part())] * 2,
                pdf_mode="native-first",
                pdf_fallback="extract-text",
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert uploads[0].data == b"first"
        with pytest.raises(DocumentError):
            await docling.convert([])
        release.set()
        await docling.close()
    assert conversion.await_count == 1
    assert all(upload.data == b"" for upload in uploads)


@pytest.mark.parametrize("failure", ["lost-post", "lost-poll", "result", "partial"])
async def test_native_unknown_jobs_and_bad_envelopes_do_not_retry(real_children, failure):
    posts = []

    def transport(request):
        if request.method == "POST":
            posts.append(request.read())
            if failure == "lost-post":
                raise httpx.ReadError("not logged")
            return httpx.Response(
                200,
                json={
                    "task_id": "one",
                    "task_type": "convert",
                    "task_status": "started" if failure == "lost-poll" else "success",
                },
            )
        if failure == "lost-poll":
            raise httpx.ReadError("not logged")
        return httpx.Response(
            200,
            json={
                "status": "partial_success" if failure == "partial" else "success",
                "errors": [],
                "document": None,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        docling = DoclingClient(
            DoclingSettings(enabled=True, api_key=SecretStr("test-only")),
            client,
        )
        with pytest.raises(DocumentError):
            await docling.convert(
                [EngineAttachmentPart.model_validate(pdf_part())],
                pdf_mode="native-first",
                pdf_fallback="extract-text",
            )
        assert docling._poisoned == (failure in {"lost-post", "lost-poll"})
        await docling.close()
    assert len(posts) == 1


@pytest.mark.parametrize("failure", [False, True])
async def test_local_upload_bytes_released_after_render_success_or_error(failure):
    local = LocalAttachments(DoclingSettings())
    uploads = [Upload("one.pdf", "pdf", b"first", 1), Upload("two.pdf", "pdf", b"second", 1)]
    with (
        patch("agentgateway_extproc.lib.local_attachments.preflight", return_value=uploads),
        patch(
            "agentgateway_extproc.lib.local_attachments.render_pdf",
            side_effect=DocumentError(reason="pdf_render_failed") if failure else None,
            return_value=["image"],
        ),
    ):
        if failure:
            with pytest.raises(DocumentError):
                await local.prepare([], ImageBatch(), "render-images")
        else:
            await local.prepare([], ImageBatch(), "render-images")
    await local.close()
    assert all(upload.data == b"" for upload in uploads)


def test_failed_preflight_clears_already_decoded_uploads():
    upload = Upload("one.pdf", "pdf", b"first", 1)
    with (
        patch("agentgateway_extproc.lib.documents._decode_upload", return_value=upload),
        patch("agentgateway_extproc.lib.documents._pdf_pages", side_effect=DocumentError(400)),
        pytest.raises(DocumentError),
    ):
        preflight([EngineAttachmentPart.model_validate(pdf_part())], DoclingSettings())
    assert upload.data == b""


@pytest.mark.parametrize("document_mode", ["block", "extract-text"])
@pytest.mark.parametrize("case", ["complete", "picture"])
async def test_checked_native_result_still_passes_text_pii_policy(
    engine_client, real_children, case, document_mode
):
    docling = DoclingClient(DoclingSettings(enabled=True, api_key=SecretStr("test-only")))
    selected = policy(
        pdf_modes={"test": "native-first"},
        pdf_fallbacks={"test": "extract-text"},
        document_modes={"test": document_mode},
        models={"test": True},
        image_modes={"test": "block"},
        image_forwarding={"test": "none"},
        face_protection={"test": document_mode == "extract-text"},
        local_models={"test": False},
        image_models={"test": False},
    )

    async def convert(upload, abandoned, *, native=False, force_ocr=False):
        assert force_ocr == (not native)
        return incomplete_native(case) if native else native_document()

    async def analyze(request, session_key, **kwargs):
        assert kwargs["text_pii_enabled"]
        assert "data:image" not in request.model_dump_json()
        return _pass_reply(
            request,
            VisualFindings.model_validate(
                {
                    "faces": {"scan_status": "not_scanned", "count": None},
                }
            ),
        )

    with (
        patch.object(docling, "_convert_one", side_effect=convert),
        patch("agentgateway_extproc.lib.docling.render_pdf") as renderer,
        patch.object(engine_client, "analyze_request", side_effect=analyze) as analysis,
    ):
        response = await dispatch(
            engine_client,
            docling,
            [pdf_part()],
            selected_policy=selected,
            history=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": source_marker(["MCP_TOOL_SALESFORCE"]),
                        }
                    ],
                }
            ],
        )
    await docling.close()
    assert response.HasField("request_body")
    analysis.assert_awaited_once()
    renderer.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        {"pdf_modes": {"test": "native-first"}},
        {"pdf_fallbacks": {"test": "extract-text"}},
        {"pdf_modes": {"test": "extract-text"}, "pdf_fallbacks": {"test": "extract-text"}},
        {"pdf_modes": {"test": "block"}, "pdf_fallbacks": {"test": "render-images"}},
        {"pdf_modes": {"test": "native-first"}, "pdf_fallbacks": {"test": "block"}},
        {"pdf_modes": {"test": "native-first"}, "pdf_fallbacks": {"other": "extract-text"}},
        {
            "pdf_modes": {"test": "native-first"},
            "pdf_fallbacks": {"test": "extract-text"},
            "pdf_max_rendered_pages": {"test": 10},
        },
        {"pdf_modes": {"test": "extract-text"}, "pdf_max_rendered_pages": {"test": 10}},
    ],
)
def test_native_fallback_and_render_budget_must_match_pdf_mode(extra):
    with pytest.raises(ValidationError):
        ModelDestinationPolicy.model_validate(policy(**extra))
