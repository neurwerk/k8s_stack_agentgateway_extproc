"""Exercise image trust boundaries with the real bundled CPU detector."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import json
import subprocess
import threading
import time
from email import policy as email_policy
from email.parser import BytesParser
from unittest.mock import patch

import httpx
import pytest
from PIL import Image, ImageDraw, PngImagePlugin
from pydantic import ValidationError

from agentgateway_extproc.config.settings import DoclingSettings, EngineSettings, Settings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.documents import DocumentError, ImageBatch, preflight
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.image_probe import MODEL_PATH, MODEL_SHA256, has_faces, normalize
from agentgateway_extproc.lib.pipeline.request import request_mutation
from agentgateway_extproc.models.destination import ModelDestinationPolicy
from agentgateway_extproc.models.engine import EngineAttachmentPart

from .conftest import (
    MODEL_POLICY,
    REVERSIBLE_TOKEN,
    body_request,
    header_request,
    response_body,
    response_headers,
)
from .test_documents import _document, _upload


def image_part(api="chat", *, format="PNG", size=(64, 64), frames=False, face=False):
    buffer = io.BytesIO()
    image = Image.new("RGB", size, "white")
    if face:
        drawing = ImageDraw.Draw(image)
        drawing.ellipse((10, 4, 54, 44), fill="peachpuff", outline="black")
        drawing.ellipse((20, 15, 24, 19), fill="black")
        drawing.ellipse((40, 15, 44, 19), fill="black")
        drawing.arc((22, 24, 42, 36), 0, 180, fill="black", width=2)
    info = PngImagePlugin.PngInfo()
    info.add_text("private", "DO-NOT-FORWARD-METADATA")
    image.save(
        buffer,
        format=format,
        pnginfo=info,
        save_all=frames,
        append_images=[Image.new("RGB", size, "black")] if frames else [],
    )
    data = buffer.getvalue() + b"DO-NOT-FORWARD-TRAILER"
    uri = (
        f"data:image/{format.lower().replace('jpg', 'jpeg')};base64,"
        + base64.b64encode(data).decode()
    )
    return (
        {"type": "image_url", "image_url": {"url": uri}}
        if api == "chat"
        else {
            "type": "input_image",
            "image_url": uri,
        }
    )


@pytest.mark.parametrize(
    "version,field",
    [(1, field) for field in ("image_forwarding", "face_protection", "local_models")]
    + [(version, field) for version in (1, 2) for field in ("image_models", "image_reroutes")],
)
def test_older_versions_reject_new_maps_even_empty(version, field):
    with pytest.raises(ValidationError):
        ModelDestinationPolicy.model_validate(
            {**MODEL_POLICY, "contract_version": version, field: {}}, strict=True
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"attachment_modes": {"test": "process"}},
        {"attachment_modes": {"test": "extract"}},
        {
            "attachment_modes": {"test": "process"},
            "image_forwarding": {"test": "if-no-pii-detected"},
        },
        {"models": {"test": False}, "attachment_modes": {"test": "passthrough"}},
        {
            "attachment_modes": {"test": "process"},
            "image_forwarding": {"test": "pii-unchecked"},
            "face_protection": {"test": False},
            "local_models": {"test": True},
        },
    ],
)
def test_v2_valid_policy(extra):
    policy = ModelDestinationPolicy.model_validate(
        {**MODEL_POLICY, "contract_version": 2, **extra}, strict=True
    )
    assert policy.contract_version == 2


@pytest.mark.parametrize(
    "extra",
    [
        {"image_forwarding": {"unknown": "none"}},
        {"face_protection": {"unknown": False}},
        {"local_models": {"unknown": True}},
        {"local_models": {"test": "true"}},
        {"face_protection": {"test": 1}},
        {"image_forwarding": {"test": "if-no-pii-detected"}, "face_protection": {"test": False}},
        {"image_forwarding": {"test": "if-no-pii-detected"}, "models": {"test": False}},
        {"image_forwarding": {"test": "pii-unchecked"}, "face_protection": {"test": False}},
        {"image_forwarding": {"test": "pii-unchecked"}, "local_models": {"test": True}},
        {"image_forwarding": {"test": "if-no-pii-detected"}, "attachment_modes": {"test": "block"}},
        {
            "models": {"test": False},
            "attachment_modes": {"test": "passthrough"},
            "face_protection": {"test": True},
        },
        {
            "models": {"test": False},
            "attachment_modes": {"test": "passthrough"},
            "image_forwarding": {"test": "none"},
        },
        {"unknown": {}},
    ],
)
def test_v2_rejects_unsafe_policy(extra):
    with pytest.raises(ValidationError):
        ModelDestinationPolicy.model_validate(
            {
                **MODEL_POLICY,
                "contract_version": 2,
                "attachment_modes": {"test": "process"},
                **extra,
            },
            strict=True,
        )


@pytest.mark.parametrize(
    "extra,valid",
    [
        ({"image_models": {"test": True}}, True),
        ({"image_models": {"test": False}}, False),
        ({"image_models": {"test": "true"}}, False),
        ({"image_models": {"unknown": True}}, False),
        ({"image_reroutes": {"unknown": {"local": "vision"}}}, False),
        ({"image_models": {"test": True}, "image_reroutes": {"test": {"local": True}}}, False),
        ({"image_models": {"test": True}, "image_reroutes": {"test": {"local": "vision"}}}, True),
    ],
)
def test_v3_requires_explicit_image_capability_and_typed_bindings(extra, valid):
    payload = {
        **MODEL_POLICY,
        "contract_version": 3,
        "attachment_modes": {"test": "process"},
        "image_forwarding": {"test": "pii-unchecked"},
        "face_protection": {"test": False},
        "local_models": {"test": True},
        **extra,
    }
    if valid:
        assert ModelDestinationPolicy.model_validate(payload, strict=True).contract_version == 3
    else:
        with pytest.raises(ValidationError):
            ModelDestinationPolicy.model_validate(payload, strict=True)


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_real_model_load_and_inference_and_normalization(format):
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == MODEL_SHA256
    part = EngineAttachmentPart.model_validate(image_part(format=format))
    batch = ImageBatch()
    upload = preflight([part], DoclingSettings(inference_mode="private-vlm"), batch)[0]
    assert not has_faces(upload.data)
    assert upload.pages == 1 and upload.format == "img"
    assert b"DO-NOT-FORWARD" not in upload.data
    assert base64.b64decode(batch.images[0].split(",")[1]) == upload.data
    with Image.open(io.BytesIO(upload.data)) as image:
        assert image.info == {} and image.mode == "RGB" and image.size == (64, 64)


def test_image_format_does_not_expand_the_document_file_allowlist():
    part = EngineAttachmentPart(
        type="file",
        file={
            "filename": "upload.img",
            "file_data": image_part()["image_url"]["url"],
        },
    )
    with pytest.raises(DocumentError) as error:
        preflight([part], DoclingSettings(inference_mode="private-vlm"), ImageBatch())
    assert error.value.status == 400


def test_jpeg_orientation_and_missing_or_corrupt_detector(tmp_path):
    image = Image.new("RGB", (32, 48), "white")
    exif = image.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    normalized = normalize(buffer.getvalue(), "JPEG")
    with Image.open(io.BytesIO(normalized)) as rebuilt:
        assert rebuilt.size == (48, 32) and not rebuilt.getexif()
    corrupt = tmp_path / "invalid.onnx"
    with patch("agentgateway_extproc.lib.image_probe.MODEL_PATH", corrupt):
        with pytest.raises(FileNotFoundError):
            has_faces(normalized)
        corrupt.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="invalid face model"):
            has_faces(normalized)


@pytest.mark.parametrize(
    "case,status",
    [
        ("url", 400),
        ("file-id", 400),
        ("extra", 400),
        ("mime", 400),
        ("base64", 400),
        ("frames", 400),
        ("dimension", 413),
        ("pixels", 413),
        ("cpu", 403),
        ("timeout", 413),
        ("detector-error", 503),
        ("face", 403),
        ("empty", 400),
    ],
)
def test_image_preflight_rejections(case, status):
    part = image_part(
        size=(4097, 1) if case == "dimension" else (4000, 4000) if case == "pixels" else (64, 64),
        frames=case == "frames",
    )
    if case == "url":
        part["image_url"]["url"] = "https://never-fetch.test/private"
    if case in {"file-id", "extra"}:
        part["file_id" if case == "file-id" else "detail"] = "private"
    if case == "mime":
        part["image_url"]["url"] = part["image_url"]["url"].replace("image/png", "image/jpeg")
    if case == "base64":
        part["image_url"]["url"] += "!"
    if case == "empty":
        part["image_url"]["url"] = "data:image/png;base64,"
    settings = DoclingSettings(
        inference_mode="internal-standard" if case == "cpu" else "private-vlm"
    )
    with patch("agentgateway_extproc.lib.documents.subprocess.run", wraps=subprocess.run) as run:
        if case == "timeout":
            run.side_effect = subprocess.TimeoutExpired("fixed", 15)
        if case in {"detector-error", "face"}:
            run.return_value = subprocess.CompletedProcess(
                [], 3 if case == "detector-error" else 0, b"\x01"
            )
            run.side_effect = None
        with pytest.raises(DocumentError) as error:
            preflight([EngineAttachmentPart.model_validate(part)], settings, ImageBatch())
        assert error.value.status == status


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize(
    "case",
    [
        "clean",
        "pass-pii",
        "mask-pii",
        "block",
        "face",
        "empty",
        "unreadable",
        "detector-error",
        "unchecked",
        "unchecked-bypass",
        "unchecked-block",
        "unchecked-error",
        "ambiguous-scan",
        "none",
        "none-mask-pii",
        "none-bypass",
        "invalid-later",
        "final-limit",
        "v3-cpu-clean",
        "v3-cpu-no-text",
        "v3-cpu-failure",
    ],
)
async def test_image_dispatch(engine_reply, api, case):  # noqa: C901
    v3 = case.startswith("v3-")
    unchecked = case.startswith("unchecked") or case == "v3-cpu-no-text"
    text_only = case.startswith("none")
    bypass = case.endswith("bypass")
    forwarding = "pii-unchecked" if unchecked else "none" if text_only else "if-no-pii-detected"
    policy = {
        **MODEL_POLICY,
        "contract_version": 3 if v3 else 2,
        "models": {"test": not bypass},
        "attachment_modes": {"test": "process"},
        "image_forwarding": {"test": forwarding},
        "face_protection": {"test": not unchecked},
        "local_models": {"test": unchecked},
    }
    if v3:
        policy["image_models"] = {"test": unchecked}
    settings = Settings(
        docling=DoclingSettings(
            enabled=True,
            api_key="test-only",
            inference_mode="internal-standard" if v3 else "private-vlm",
        ),
        max_transformed_request_bytes=1024 if case == "final-limit" else 10 * 1_048_576,
    )
    field = "messages" if api == "chat" else "input"
    text_type = "text" if api == "chat" else "input_text"
    later = (
        {"type": "file", "file": _upload()}
        if api == "chat"
        else {"type": "input_file", **_upload()}
    )
    if case == "invalid-later":
        later = {**image_part(api), "file_id": "must-reject-before-conversion"}
    original = {
        "model": "test",
        field: [
            {
                "role": "user",
                "content": [
                    {"type": text_type, "text": "before" * (300 if case == "final-limit" else 1)},
                    image_part(api, face=text_only),
                    {"type": text_type, "text": "after"},
                ],
            },
            {
                "role": "user",
                "content": [later],
            },
        ],
    }
    native_calls, engine_calls, canonical = [], [], []
    current_format = ""
    doc = _document()
    if case in {"empty", "v3-cpu-no-text"}:
        doc = {
            **doc,
            "texts": [],
            "groups": [],
            "tables": [],
            "pictures": [],
            "body": {"self_ref": "#/body", "children": []},
            "furniture": {"self_ref": "#/furniture", "children": []},
        }

    def native(request):
        nonlocal current_format
        native_calls.append(request)
        if request.method == "POST":
            form = BytesParser(policy=email_policy.default).parsebytes(
                f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
                + request.content
            )
            fields = {
                part.get_param("name", header="content-disposition"): part
                for part in form.iter_parts()
            }
            current_format = fields["from_formats"].get_payload()
            if current_format == "image":
                if not v3:
                    assert fields["vlm_pipeline_preset"].get_payload() == "images"
                assert fields["files"].get_filename() == "upload.png"
                canonical.append(fields["files"].get_payload(decode=True))
                assert b"DO-NOT-FORWARD" not in canonical[-1]
            elif not v3:
                assert fields["vlm_pipeline_preset"].get_payload() == "default"
            assert fields["pipeline"].get_payload() == ("standard" if v3 else "vlm")
            if v3:
                assert fields["do_ocr"].get_payload() == "true"
            return httpx.Response(
                200, json={"task_id": "job", "task_type": "convert", "task_status": "success"}
            )
        return httpx.Response(
            200,
            json={
                "status": "failure" if case in {"unreadable", "v3-cpu-failure"} else "success",
                "errors": [],
                "document": {"json_content": doc if current_format == "image" else _document()},
            },
        )

    def engine(request):
        engine_calls.append(request)
        assert request.url.path == "/v1/adapter/analyze-document-request"
        assert b"base64" not in request.content and b"image_url" not in request.content
        if case == "unchecked-error":
            return httpx.Response(500, text="DO-NOT-FORWARD")
        sent = json.loads(request.content)
        if v3:
            assert sent["text_pii_enabled"] is True
            engine_reply["visual_findings"] = sent["visual_findings"]
            sent = sent["request"]
        engine_reply.update(
            request=copy.deepcopy(sent),
            decision="pass",
            entities=[],
            entity_counts={},
            applied_actions=[],
            reversal={},
            report={"rows": []},
        )
        if case == "ambiguous-scan":
            engine_reply["analysis"]["text_leaf_count"] = 0
        if case in {
            "pass-pii",
            "mask-pii",
            "none-mask-pii",
            "unchecked",
            "block",
            "unchecked-block",
        }:
            action = (
                "mask"
                if case.endswith("mask-pii")
                else "block"
                if case.endswith("block")
                else "pass"
            )
            engine_reply.update(
                entities=["PERSON"],
                entity_counts={"PERSON": 2},
                applied_actions=[action],
                decision="apply_actions" if action == "mask" else action,
            )
            engine_reply["report"] = {
                "rows": [
                    {
                        "entity_type": "PERSON",
                        "action": action,
                        "detected_count": 2,
                        "transformed_count": 2 if action == "mask" else 0,
                        "unique_transformed_count": 1 if action == "mask" else 0,
                    }
                ]
            }
            if action == "mask":
                for message in engine_reply["request"][field]:
                    for part in message["content"]:
                        part["text"] = part["text"].replace("Jane Doe", "***")
            if action == "block":
                engine_reply.update(request=None, remote_allowed=False)
        return httpx.Response(200, json=engine_reply)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(native)) as native_http,
        httpx.AsyncClient(transport=httpx.MockTransport(engine)) as engine_http,
    ):
        docling = DoclingClient(settings.docling, native_http)
        servicer = ExtProcServicer(EngineClient(EngineSettings(), engine_http), settings, docling)

        async def requests():
            yield header_request(policy=policy)
            yield body_request(json.dumps(original).encode(), policy=policy)

        native_run = subprocess.run

        def checked_probe(command, **kwargs):
            # A face-positive or failing detector must never gate text-only extraction.
            if text_only:
                assert command[-1] == "skip", "YuNet must not run for image_forwarding=none"
            return native_run(command, **kwargs)

        with (
            patch(
                "agentgateway_extproc.lib.documents.subprocess.run", side_effect=checked_probe
            ) as run,
            patch(
                "agentgateway_extproc.lib.image_probe.has_faces",
                side_effect=AssertionError("face detector must stay off the text-only path"),
            ),
        ):
            if case in {"face", "detector-error"}:
                run.side_effect = None
                run.return_value = subprocess.CompletedProcess(
                    [], 3 if case == "detector-error" else 0, b"\x01"
                )
            replies = [reply async for reply in servicer.Process(requests(), object())]
        result = replies[-1]
        if (
            case in {"clean", "unchecked", "unchecked-bypass", "v3-cpu-clean", "v3-cpu-no-text"}
            or text_only
        ):
            assert result.HasField("request_body")
            wire = result.request_body.response.body_mutation.body
            forwarded = json.loads(wire)
            content = forwarded[field][0 if api == "responses" or bypass else 1]["content"]
            assert content[0]["text"] == "before" and content[-1]["text"] == "after"
            if text_only:
                assert len(content) == 3 and canonical
                assert b"base64" not in wire and b"image_url" not in wire
                assert b"DO-NOT-FORWARD" not in wire
                assert "Document: image.png" in content[1]["text"]
                if case == "none-mask-pii":
                    assert b"Jane Doe" not in wire and "***" in content[1]["text"]
                else:
                    assert "Jane Doe" in content[1]["text"]
                assert all(call.args[0][-1] == "skip" for call in run.call_args_list)
            else:
                image = content[2]["image_url"]
                uri = image["url"] if api == "chat" else image
                assert base64.b64decode(uri.split(",")[1]) == canonical[0]
            assert len(engine_calls) == int(not bypass)
            if case == "v3-cpu-no-text":
                assert content[1]["text"] == "[Image: no text extracted]"
        else:
            assert result.HasField("immediate_response")
            assert not result.HasField("request_body")
            assert "DO-NOT-FORWARD" not in result.immediate_response.body
            if case in {"face", "detector-error", "invalid-later"}:
                assert not native_calls and not engine_calls
            if case == "v3-cpu-failure":
                assert native_calls and not engine_calls
                assert result.immediate_response.status.code == 503
            if case in {"pass-pii", "mask-pii", "face", "block"}:
                assert "text only" in result.immediate_response.body
        await docling.close()


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize(
    "case,face_action,forwarding,pii,has_text,status",
    [
        ("clean", None, "if-no-pii-detected", True, True, 200),
        ("text-only", "text-only", "if-no-pii-detected", True, True, 200),
        ("text-only-off", "text-only", "none", False, True, 200),
        ("clean-off", None, "none", False, True, 200),
        ("face-block", "block", "none", False, True, 403),
        ("reroute", "reroute", "if-no-pii-detected", True, True, 200),
        ("reroute-no-text", "reroute", "if-no-pii-detected", True, False, 200),
        ("missing-binding", "reroute", "if-no-pii-detected", True, True, 403),
        ("wrong-binding", "reroute", "if-no-pii-detected", True, True, 403),
        ("reroute-none", "reroute", "none", False, True, 403),
        ("text-block", "reroute", "if-no-pii-detected", True, True, 403),
        ("text-only-no-text", "text-only", "if-no-pii-detected", True, False, 403),
        ("none-no-text", None, "none", False, False, 403),
        ("external-no-text", None, "if-no-pii-detected", True, False, 403),
        ("text-pii", None, "if-no-pii-detected", True, True, 403),
        ("unchecked-no-text", None, "pii-unchecked", True, False, 200),
        ("bypass-no-text", None, "pii-unchecked", False, False, 200),
        ("structured-success", "text-only", "none", False, True, 200),
        ("structured-block", "block", "none", False, True, 403),
    ],
)
async def test_v3_image_dispatch(  # noqa: C901
    engine_reply, api, case, face_action, forwarding, pii, has_text, status
):
    unchecked = forwarding == "pii-unchecked"
    bypass = unchecked and not pii
    structured = case.startswith("structured")
    policy = {
        **MODEL_POLICY,
        "contract_version": 3,
        "models": {"test": pii},
        "attachment_modes": {"test": "process"},
        "image_forwarding": {"test": forwarding},
        "face_protection": {"test": not unchecked},
        "local_models": {"test": unchecked},
        "image_models": {"test": unchecked},
        "image_reroutes": {"test": {"local-faces": "approved-vision"}},
    }
    if case == "missing-binding":
        policy["image_reroutes"] = {}
    elif case == "wrong-binding":
        policy["image_reroutes"] = {"test": {"other-route": "approved-vision"}}
    field = "messages" if api == "chat" else "input"
    original = {
        "model": "test",
        field: [
            {
                "role": "user",
                "content": [
                    image_part(api),
                    {"type": "file", "file": _upload()}
                    if api == "chat"
                    else {"type": "input_file", **_upload()},
                    image_part(api),
                ],
            }
        ],
    }
    if structured:
        original.update(
            {"response_format": {"type": "json_object"}}
            if api == "chat"
            else {"text": {"format": {"type": "json_object"}}}
        )
    faces = {
        "scan_status": "not_scanned" if unchecked else "complete",
        "count": None if unchecked else 3 if face_action else 0,
    }
    normalized = "data:image/png;base64,bm9ybWFsaXplZA=="

    async def convert(parts, *, images):
        assert len(parts) == 3 and images.policy_version == 3
        assert images.protect_faces is not unchecked
        images.images = {0: normalized, 2: normalized}
        images.text_present = {0: True, 2: has_text}
        images.scan_status = faces["scan_status"]
        images.face_count = faces["count"] or 0
        return [
            "image text",
            "document text",
            "image text" if has_text else "[Image: no text extracted]",
        ]

    engine_calls = []

    def engine(request):
        engine_calls.append(request)
        assert not bypass
        assert request.url.path == "/v1/adapter/analyze-document-request"
        sent = json.loads(request.content)
        assert set(sent) == {"api_version", "request", "text_pii_enabled", "visual_findings"}
        assert sent["api_version"] == "v1" and sent["text_pii_enabled"] is pii
        assert sent["visual_findings"] == {"faces": faces}
        assert b"base64" not in request.content and b"image_url" not in request.content
        rows = []
        if face_action:
            rows.append(
                {
                    "entity_type": "FACE",
                    "action": face_action,
                    "detected_count": 3,
                    "transformed_count": 0,
                    "unique_transformed_count": 0,
                }
            )
        if case in {"text-block", "text-pii"}:
            rows.append(
                {
                    "entity_type": "PERSON",
                    "action": "block" if case == "text-block" else "pass",
                    "detected_count": 1,
                    "transformed_count": 0,
                    "unique_transformed_count": 0,
                }
            )
        decision = (
            "block"
            if case == "text-block" or face_action == "block"
            else (
                "reroute"
                if face_action == "reroute"
                else "apply_actions"
                if face_action
                else "pass"
            )
        )
        engine_reply.update(
            request=None if decision == "block" else sent["request"],
            visual_findings=sent["visual_findings"],
            entities=[row["entity_type"] for row in rows],
            entity_counts={row["entity_type"]: row["detected_count"] for row in rows},
            decision=decision,
            applied_actions=["block"]
            if decision == "block"
            else sorted({row["action"] for row in rows}),
            remote_allowed=decision not in {"block", "reroute"},
            route_class="local-faces" if decision == "reroute" else None,
            report={"rows": rows},
            reversal={},
            notices={"request": [], "response": []},
        )
        engine_reply["analysis"].update(scan_performed=pii, duration_ms=1 if pii else None)
        return httpx.Response(200, json=engine_reply)

    # Exercise both successful carriers without enabling text PII merely to render a report.
    sse = api == "responses"
    output_text = "answer" if pii else REVERSIBLE_TOKEN
    answer = (
        {"choices": [{"index": 0, "message": {"content": output_text}}]}
        if not sse
        else {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": output_text,
        }
    )
    wire = json.dumps(answer).encode()
    if sse:
        wire = (
            b"data: "
            + wire
            + b'\n\ndata: {"type":"response.completed","response":{"status":"completed"}}'
            b"\n\ndata: [DONE]\n\n"
        )
    settings = Settings()
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as http:
        docling = DoclingClient(settings.docling)
        servicer = ExtProcServicer(EngineClient(settings.engine, http), settings, docling)

        async def requests():
            yield header_request(policy=policy)
            yield body_request(json.dumps(original).encode(), policy=policy)
            yield response_headers(
                "text/event-stream" if sse else "application/json",
                policy=policy,
                extra_headers={"x-presidio-code": "P00"} if not bypass else None,
            )
            yield response_body(wire, policy=policy)

        with patch.object(docling, "convert", side_effect=convert):
            replies = [reply async for reply in servicer.Process(requests(), object())]
        await docling.close()
    assert len(engine_calls) == int(not bypass)
    assert replies[0].HasField("mode_override") is bypass
    if status == 403:
        blocked = replies[-1].immediate_response
        assert blocked.status.code == 403
        error = json.loads(blocked.body)
        assert error["error"]["message"].startswith("image withheld")
        assert ("PII Engine Notice" in error["error"]["message"]) is not structured
        assert error["pii_report"]["decision"] == "block"
        face_rows = [row for row in error["pii_report"]["rows"] if row["entity_type"] == "FACE"]
        assert face_rows == (
            [
                {
                    "entity_type": "FACE",
                    "action": "block",
                    "detected_count": 3,
                    "transformed_count": 0,
                    "unique_transformed_count": 0,
                }
            ]
            if face_action
            else []
        )
        if face_action and not structured:
            assert "| Face | `block`: 3 detected; images blocked |" in error["error"]["message"]
        assert "forwarded to" not in blocked.body and "forwarded without" not in blocked.body
        assert "Effective route" not in blocked.body
        return
    sent_body = replies[1].request_body.response.body_mutation.body
    sent = json.loads(sent_body)
    content = sent[field][int(pii and api == "chat")]["content"]
    restores_images = forwarding != "none" and face_action != "text-only"
    assert sum(part["type"] in {"image_url", "input_image"} for part in content) == (
        2 if restores_images else 0
    )
    if not has_text:
        assert b"[Image: no text extracted]" in sent_body
    response = replies[-1].response_body.response.body_mutation.streamed_response.body.decode()
    assert output_text in response
    if not pii and not bypass:
        headers = next(
            reply.response_headers.response.header_mutation
            for reply in replies
            if reply.HasField("response_headers")
        )
        assert "x-presidio-code" in headers.remove_headers
        assert not any(item.header.key == "x-presidio-code" for item in headers.set_headers)
    assert ("PII Engine Notice" in response) is (not structured and not bypass)
    if face_action and not structured:
        assert f"| Face | `{face_action}`: 3 detected" in response
        assert "masked" not in response and "restored" not in response
    if not pii and not structured and not bypass:
        assert "Text PII analysis was disabled." in response
        assert "PII scan completed" not in response


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("version", [1, 2, 3])
async def test_passthrough_preserves_original_bytes_without_docling(api, version):
    policy = {
        **MODEL_POLICY,
        "contract_version": version,
        "models": {"test": False},
        "attachment_modes": {"test": "passthrough"},
    }
    field = "messages" if api == "chat" else "input"
    url = "https://backend-owned.test/original?keep=exact"
    reference = (
        {"type": "image_url", "image_url": {"url": url, "detail": "original"}}
        if api == "chat"
        else {"type": "input_image", "image_url": url, "detail": "original"}
    )
    raw_file = (
        {"type": "file", "file": {"file_data": "RAW-UNTOUCHED", "file_id": "original"}}
        if api == "chat"
        else {"type": "input_file", "file_data": "RAW-UNTOUCHED", "file_id": "original"}
    )
    wire = (
        json.dumps(
            {
                "model": "test",
                field: [
                    {
                        "role": "user",
                        "content": [image_part(api), reference, raw_file],
                    }
                ],
            },
            indent=3,
        ).encode()
        + b"\n"
    )
    settings = Settings()

    def unexpected(_request):
        pytest.fail("passthrough must not call a reader or PII")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as http:
        servicer = ExtProcServicer(EngineClient(settings.engine, http), settings, docling=None)

        async def requests():
            yield header_request(policy=policy)
            yield body_request(wire, policy=policy)

        with (
            patch(
                "agentgateway_extproc.lib.documents.subprocess.run",
                side_effect=AssertionError("passthrough must not decode or detect"),
            ) as run,
            patch.object(
                DoclingClient, "convert", side_effect=AssertionError("passthrough must not convert")
            ) as convert,
            patch(
                "agentgateway_extproc.lib.pipeline.request.request_mutation", wraps=request_mutation
            ) as mutation,
        ):
            replies = [reply async for reply in servicer.Process(requests(), object())]
        assert replies[-1].HasField("request_body")
        assert not replies[-1].request_body.response.HasField("body_mutation")
        mutation.assert_called_once_with(wire, {}, False)
        run.assert_not_called()
        convert.assert_not_called()


@pytest.mark.parametrize("result", [(0, None), (1, []), (1, [[float("nan")] * 15])])
def test_ambiguous_face_detector_output_is_never_clean(result):
    import numpy as np

    status, rows = result
    image = normalize(base64.b64decode(image_part()["image_url"]["url"].split(",")[1]), "PNG")
    with patch("cv2.FaceDetectorYN.create") as create:
        create.return_value.detect.return_value = (status, None if rows is None else np.array(rows))
        with pytest.raises(ValueError):
            has_faces(image)


@pytest.mark.parametrize("mode", ["private-vlm", "remote"])
def test_image_preset_and_transition_alias_preserve_pdf_default(mode):
    client = DoclingClient(DoclingSettings(inference_mode=mode))
    assert client._options("img")["vlm_pipeline_preset"] == "images"
    assert client._options("pdf")["vlm_pipeline_preset"] == "default"


@pytest.mark.parametrize("size", [(2000, 1000), (1000, 2000), (2048, 64), (64, 2048), (64, 64)])
def test_real_protected_helper_at_supported_bounds(size):
    part = EngineAttachmentPart.model_validate(image_part(size=size))
    upload = preflight([part], DoclingSettings(inference_mode="private-vlm"), ImageBatch())[0]
    with Image.open(io.BytesIO(upload.data)) as image:
        assert image.size == size
        assert image.getextrema() == ((255, 255),) * 3


@pytest.mark.parametrize(
    "size",
    [
        (2000, 1001),
        (1001, 2000),
        (2049, 64),
        (64, 2049),
        (3000, 2000),
        (4000, 3000),
        (2048, 63),
        (63, 2048),
        (2048, 32),
        (32, 2048),
    ],
)
def test_real_protected_helper_rejects_overlimit_without_resizing(size):
    part = EngineAttachmentPart.model_validate(image_part(size=size))
    with pytest.raises(DocumentError) as error:
        preflight([part], DoclingSettings(inference_mode="private-vlm"), ImageBatch())
    assert error.value.status == 413


@pytest.mark.parametrize("size", [(4000, 3000), (3000, 4000), (4096, 32), (32, 4096)])
def test_real_text_only_helper_keeps_higher_normalization_bounds(size):
    part = EngineAttachmentPart.model_validate(image_part(size=size))
    upload = preflight(
        [part], DoclingSettings(inference_mode="private-vlm"), ImageBatch(protect_faces=False)
    )[0]
    with Image.open(io.BytesIO(upload.data)) as image:
        assert image.size == size


def test_opencv_identifiable_memory_error_is_a_size_failure():
    import cv2

    image = normalize(base64.b64decode(image_part()["image_url"]["url"].split(",")[1]), "PNG")
    error = cv2.error("private allocation error")
    error.code = cv2.Error.StsNoMem
    with patch("cv2.FaceDetectorYN.create", side_effect=error), pytest.raises(MemoryError):
        has_faces(image)


@pytest.mark.parametrize("case", ["cancel", "deadline"])
async def test_image_preflight_stops_between_helpers_and_retains_admission(case):
    entered, release = threading.Event(), threading.Event()
    settings = DoclingSettings(
        enabled=True,
        api_key="test-only",
        inference_mode="private-vlm",
        timeout=0.1 if case == "deadline" else 5,
    )
    parts = [EngineAttachmentPart.model_validate(image_part()) for _ in range(3)]
    native_run = subprocess.run

    def in_flight_helper(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return native_run(*args, **kwargs)

    def unexpected(_request):
        pytest.fail("abandoned preflight must not submit any native job")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as http:
        client = DoclingClient(settings, http)
        with patch(
            "agentgateway_extproc.lib.documents.subprocess.run", side_effect=in_flight_helper
        ) as run:
            caller = asyncio.create_task(client.convert(parts, images=ImageBatch()))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                retained = client._task
                assert retained is not None
                if case == "cancel":
                    caller.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await caller
                else:
                    with pytest.raises(DocumentError) as error:
                        await caller
                    assert error.value.status == 504
                assert isinstance(client._abandoned, threading.Event)
                assert client._abandoned.is_set() and not retained.done()
                with pytest.raises(DocumentError):
                    await client.convert(parts, images=ImageBatch())
                release.set()
                await asyncio.gather(retained, return_exceptions=True)
                await asyncio.sleep(0)
                assert run.call_count == 1 and client._task is None
                assert not client._poisoned
            finally:
                release.set()
                await asyncio.gather(caller, return_exceptions=True)
                await client.close()


def test_preflight_expired_monotonic_deadline_never_starts_helper():
    parts = [EngineAttachmentPart.model_validate(image_part())]
    with patch("agentgateway_extproc.lib.documents.subprocess.run") as run:
        with pytest.raises(DocumentError) as error:
            preflight(
                parts,
                DoclingSettings(inference_mode="private-vlm"),
                ImageBatch(),
                threading.Event(),
                time.monotonic() - 1,
            )
        assert error.value.status == 504
        run.assert_not_called()
