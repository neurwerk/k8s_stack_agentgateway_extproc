from __future__ import annotations

import base64
import io
import json
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image
from pydantic import SecretStr, ValidationError

from agentgateway_extproc.config.settings import DoclingSettings, Settings
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.documents import DocumentError, ImageBatch, preflight
from agentgateway_extproc.lib.image_probe import COUNT_HEADER, normalize
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.destination import ModelDestinationPolicy
from agentgateway_extproc.models.engine import EngineAttachmentPart

from .conftest import MODEL_POLICY, body_request, header_request
from .test_documents import _document, _upload
from .test_images import image_part


def _v4_policy(**extra: object) -> dict[str, object]:
    return {
        **MODEL_POLICY,
        "models": {"test": False},
        "contract_version": 4,
        "document_modes": {"test": "block"},
        "image_modes": {"test": "forward-normalized"},
        "image_forwarding": {"test": "pii-unchecked"},
        "face_protection": {"test": False},
        "local_models": {"test": True},
        "image_models": {"test": True},
        **extra,
    }


def _typed_policy(**extra: object) -> dict[str, object]:
    return {
        **MODEL_POLICY,
        "contract_version": 4,
        "document_modes": {"test": "block"},
        "image_modes": {"test": "block"},
        "image_forwarding": {"test": "none"},
        "face_protection": {"test": False},
        "local_models": {"test": False},
        "image_models": {"test": False},
        **extra,
    }


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("field", ["document_modes", "image_modes"])
def test_v1_v3_reject_v4_maps_even_when_empty(version, field):
    with pytest.raises(ValidationError):
        ModelDestinationPolicy.model_validate(
            {**MODEL_POLICY, "contract_version": version, field: {}}, strict=True
        )


def test_v4_rejects_legacy_or_incomplete_typed_modes():
    with pytest.raises(ValidationError):
        ModelDestinationPolicy.model_validate(
            {
                **MODEL_POLICY,
                "contract_version": 4,
                "attachment_modes": {},
                "document_modes": {"test": "block"},
                "image_modes": {"test": "block"},
            },
            strict=True,
        )
    for missing in ("document_modes", "image_modes"):
        payload = _typed_policy()
        del payload[missing]
        with pytest.raises(ValidationError):
            ModelDestinationPolicy.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    "extra",
    [
        {
            "document_modes": {"test": "extract-text"},
            "face_protection": {"test": True},
        },
        {
            "image_modes": {"test": "extract-text"},
            "image_forwarding": {"test": "none"},
            "face_protection": {"test": True},
        },
        {
            "image_modes": {"test": "forward-normalized"},
            "image_forwarding": {"test": "if-no-pii-detected"},
            "face_protection": {"test": True},
        },
        {
            "image_modes": {"test": "forward-normalized"},
            "image_forwarding": {"test": "if-policy-allows"},
            "face_protection": {"test": True},
        },
        {
            "image_modes": {"test": "forward-normalized"},
            "image_forwarding": {"test": "pii-unchecked"},
            "face_protection": {"test": False},
            "local_models": {"test": True},
            "image_models": {"test": True},
            "models": {"test": False},
        },
    ],
)
def test_v4_accepts_each_coherent_typed_mode(extra):
    policy = ModelDestinationPolicy.model_validate(
        _typed_policy(**extra),
        strict=True,
    )
    assert policy.contract_version == 4


@pytest.mark.parametrize(
    "extra",
    [
        {"image_modes": {"test": "forward-normalized"}},
        {"image_modes": {"test": "extract-text"}},
        {
            "image_modes": {"test": "extract-text"},
            "image_forwarding": {"test": "if-policy-allows"},
        },
        {
            "image_modes": {"test": "forward-normalized"},
            "image_forwarding": {"test": "pii-unchecked"},
            "face_protection": {"test": False},
            "local_models": {"test": True},
        },
    ],
)
def test_v4_rejects_incoherent_image_policy(extra):
    with pytest.raises(ValidationError):
        ModelDestinationPolicy.model_validate(
            _typed_policy(**extra),
            strict=True,
        )


async def test_v4_unchecked_image_is_ocr_face_and_pii_free(engine_client):
    source_part = image_part()
    original = {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    source_part,
                    {"type": "text", "text": "after"},
                ],
            }
        ],
    }

    docling = DoclingClient(DoclingSettings())
    handler = StreamHandler(engine_client, Settings(), docling)
    uri = source_part["image_url"]["url"]
    source = base64.b64decode(uri.split(",")[1])
    canonical = normalize(source, "PNG", policy_version=4)

    def normalization_only(command, **kwargs):
        assert command[-2:] == ["skip", "4"]
        assert kwargs["input"] == source
        output = COUNT_HEADER + b"\x00\x00" + canonical
        return type("Result", (), {"returncode": 0, "stdout": output})()

    with (
        patch.object(docling, "_http", new_callable=AsyncMock) as http,
        patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze,
        patch(
            "agentgateway_extproc.lib.documents.subprocess.run", side_effect=normalization_only
        ) as run,
    ):
        try:
            await handler.handle(header_request(policy=_v4_policy()))
            response = await handler.handle(
                body_request(json.dumps(original).encode(), policy=_v4_policy())
            )
        finally:
            await docling.close()

    http.assert_not_awaited()
    analyze.assert_not_awaited()
    run.assert_called_once()
    assert response is not None and response.HasField("request_body")
    content = json.loads(response.request_body.response.body_mutation.body)["messages"][0][
        "content"
    ]
    assert [part["type"] for part in content] == ["text", "image_url", "text"]
    uri = content[1]["image_url"]["url"]
    assert uri.startswith("data:image/png;base64,")
    assert b"DO-NOT-FORWARD" not in base64.b64decode(uri.partition(",")[2])


@pytest.mark.parametrize("family", ["chat", "responses"])
async def test_v4_mixed_documents_convert_while_unchecked_images_keep_position(
    engine_client, family
):
    source_part = image_part()
    uri = source_part["image_url"]["url"]
    png = normalize(base64.b64decode(uri.partition(",")[2]), "PNG", policy_version=4)
    canonical = "data:image/png;base64," + base64.b64encode(png).decode()
    policy = _v4_policy(
        document_modes={"test": "extract-text"},
    )
    chat = family == "chat"
    parts = (
        [source_part, {"type": "file", "file": _upload("xlsx")}]
        if chat
        else [{"type": "input_image", "image_url": uri}, {"type": "input_file", **_upload("xlsx")}]
    )
    original = {
        "model": "test",
        "messages" if chat else "input": [{"role": "user", "content": parts}],
    }
    docling = DoclingClient(DoclingSettings(enabled=True, api_key=SecretStr("test-only")))
    handler = StreamHandler(engine_client, Settings(), docling)
    converted_formats = []

    async def extract(upload, abandoned):
        converted_formats.append(upload.format)
        return _document()

    with (
        patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze,
        patch.object(docling, "_convert_one", side_effect=extract),
        patch(
            "agentgateway_extproc.lib.documents._normalize_image", return_value=(png, 0)
        ) as image,
    ):
        try:
            await handler.handle(header_request(policy=policy))
            response = await handler.handle(
                body_request(json.dumps(original).encode(), policy=policy)
            )
        finally:
            await docling.close()

    assert converted_formats == ["xlsx"]
    assert image.call_args.args[2:] == (False, 4)
    analyze.assert_not_awaited()
    assert response is not None and response.HasField("request_body")
    content = json.loads(response.request_body.response.body_mutation.body)[
        "messages" if chat else "input"
    ][0]["content"]
    assert len(content) == 2
    assert content[0] == (
        {"type": "image_url", "image_url": {"url": canonical}}
        if chat
        else {"type": "input_image", "image_url": canonical}
    )
    assert content[1]["type"] == ("text" if chat else "input_text")
    assert "Jane Doe" in content[1]["text"]
    assert "file_data" not in content[1]


async def test_v4_blocked_part_rejects_whole_batch_before_side_effects(engine_client):
    policy = {
        **_typed_policy(),
        "document_modes": {"test": "extract-text"},
        "face_protection": {"test": True},
    }
    original = {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "file", "file": _upload()},
                    image_part(),
                ],
            }
        ],
    }
    docling = AsyncMock()
    handler = StreamHandler(engine_client, Settings(), docling)
    with (
        patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze,
        patch("agentgateway_extproc.lib.documents.subprocess.run") as run,
    ):
        await handler.handle(header_request(policy=policy))
        response = await handler.handle(body_request(json.dumps(original).encode(), policy=policy))

    docling.convert.assert_not_awaited()
    docling.convert_selected.assert_not_awaited()
    analyze.assert_not_awaited()
    run.assert_not_called()
    assert response is not None
    assert response.immediate_response.status.code == 403


def _webp_part(*, animated: bool = False) -> EngineAttachmentPart:
    output = io.BytesIO()
    image = Image.new("RGBA", (32, 24), (255, 0, 0, 128))
    image.save(
        output,
        format="WEBP",
        save_all=animated,
        append_images=[Image.new("RGB", (32, 24), "blue")] if animated else [],
    )
    uri = "data:image/webp;base64," + base64.b64encode(output.getvalue()).decode()
    return EngineAttachmentPart.model_validate({"type": "image_url", "image_url": {"url": uri}})


def test_v4_webp_normalizes_to_metadata_free_rgb_png():
    part = _webp_part()
    source = base64.b64decode(part.model_dump()["image_url"]["url"].partition(",")[2])
    rebuilt = normalize(source, "WEBP", policy_version=4)
    with Image.open(io.BytesIO(rebuilt)) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (32, 24)
        assert image.info == {}


def test_v4_rejects_animated_webp_and_v3_rejects_all_webp():
    animated_part = _webp_part(animated=True)
    source = base64.b64decode(animated_part.model_dump()["image_url"]["url"].partition(",")[2])
    with pytest.raises(ValueError, match="frames"):
        normalize(source, "WEBP", policy_version=4)
    with pytest.raises(ValueError, match="unsupported"):
        normalize(source, "WEBP", policy_version=3)
    with pytest.raises(DocumentError) as legacy:
        preflight(
            [_webp_part()],
            DoclingSettings(),
            ImageBatch(protect_faces=False, policy_version=3),
        )
    assert legacy.value.status == 400 and legacy.value.reason == "unsupported_format"
