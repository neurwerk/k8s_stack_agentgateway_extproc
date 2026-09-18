"""Real codec regressions for version-three canonical image pixels and bounds."""

import base64
import io

import numpy as np
import pillow_heif
import pytest
from PIL import Image, ImageChops, ImageDraw, ImageOps

from agentgateway_extproc.config.settings import DoclingSettings
from agentgateway_extproc.lib.documents import DocumentError, ImageBatch, _image_data, preflight
from agentgateway_extproc.lib.image_probe import MAX_IMAGE_BYTES, MAX_SOURCE_BYTES
from agentgateway_extproc.models.engine import EngineAttachmentPart


def _encode(image, format="HEIF", **kwargs):
    # The dev-only encoder makes fixtures; the isolated runtime helper uses pi-heif.
    pillow_heif.register_heif_opener()
    stream = io.BytesIO()
    image.save(
        stream,
        format=format,
        enc_params={
            "preset": "ultrafast",
            "x265:pools": "none",
            "x265:frame-threads": "1",
            "x265:log-level": "none",
        },
        **kwargs,
    )
    return stream.getvalue()


def _part(data, mime="heic"):
    return EngineAttachmentPart(
        type="input_image", image_url=f"data:image/{mime};base64," + base64.b64encode(data).decode()
    )


@pytest.mark.parametrize(
    "api,fields,version,valid",
    [
        ("chat", {"detail": "auto"}, 3, True),
        ("responses", {"detail": "low"}, 3, True),
        ("chat", {"detail": "high"}, 3, True),
        ("chat", {"detail": "auto"}, 2, False),
        ("responses", {"detail": "auto"}, 2, False),
        ("chat", {"detail": "other"}, 3, False),
        ("responses", {"detail": None}, 3, False),
        ("chat", {"detail": ["auto"]}, 3, False),
        ("chat", {"detail": "auto", "file_id": "unknown"}, 3, False),
        ("responses", {"detail": "auto", "unknown": True}, 3, False),
    ],
)
def test_detail_is_an_optional_v3_hint_not_an_extra_field_allowance(api, fields, version, valid):
    data = _encode(Image.new("RGB", (64, 64), "white"), "PNG")
    value = _part(data, "png").model_dump()
    if api == "chat":
        value = {"type": "image_url", "image_url": {"url": value["image_url"], **fields}}
    else:
        value.update(fields)
    part = EngineAttachmentPart.model_validate(value)
    if valid:
        assert _image_data(part, MAX_IMAGE_BYTES, policy_version=version) == ("png", data)
        assert part.model_dump() == value
    else:
        with pytest.raises(DocumentError) as error:
            _image_data(part, MAX_IMAGE_BYTES, policy_version=version)
        assert error.value.status == 400


@pytest.mark.parametrize(
    "orientation,mime", [(2, "heic"), (6, "heif"), (7, "x-heic"), (8, "x-heif")]
)
def test_heic_orientation_alpha_auxiliary_and_metadata(orientation, mime):
    image = Image.new("RGBA", (96, 64), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((0, 0, 47, 31), fill=(240, 40, 20, 255))
    image.getexif()[274] = orientation
    image.getexif()[315] = "PRIVATE-METADATA"
    data = _encode(image, quality=-1, chroma=444, xmp=b"PRIVATE-METADATA")
    batch = ImageBatch(policy_version=3)
    upload = preflight([_part(data, mime)], DoclingSettings(), batch)[0]
    expected = ImageOps.exif_transpose(image)
    white = Image.new("RGBA", expected.size, "white")
    white.alpha_composite(expected)
    with Image.open(io.BytesIO(upload.data)) as canonical:
        assert canonical.mode == "RGB" and canonical.size == expected.size
        assert canonical.info == {} and not canonical.getexif()
        assert np.asarray(ImageChops.difference(canonical, white.convert("RGB"))).max() <= 3
    assert b"PRIVATE-METADATA" not in upload.data
    assert base64.b64decode(batch.images[0].split(",")[1]) == upload.data
    assert batch.scan_status == "complete" and batch.face_count == 0


@pytest.mark.parametrize(
    "format,size,protect",
    [
        ("PNG", (5712, 4284), False),
        ("HEIF", (10000, 5000), True),
    ],
)
def test_phone_pixels_use_one_bounded_canonical_image(format, size, protect):
    with Image.new("RGB", size, "white") as image:
        data = _encode(image, format, tile_size=512)
    with Image.open(io.BytesIO(data)) as source:
        assert source.size == size
    part = _part(data, "heic" if format == "HEIF" else format.lower())
    batch = ImageBatch(policy_version=3, protect_faces=protect)
    upload = preflight([part], DoclingSettings(), batch)[0]
    with Image.open(io.BytesIO(upload.data)) as canonical:
        width, height = canonical.size
        assert width * height <= 2_000_000 and max(width, height) <= 2048
        assert canonical.getextrema() == ((255, 255),) * 3 and canonical.info == {}
    assert batch.scan_status == ("complete" if protect else "not_scanned")
    assert base64.b64decode(batch.images[0].split(",")[1]) == upload.data
    assert upload.source_bytes == len(data) and len(batch.images[0]) <= MAX_IMAGE_BYTES


def test_real_phone_source_above_legacy_bytes_keeps_normalized_and_request_caps():
    noise = np.random.default_rng(0).integers(96, 160, (3000, 4000), dtype=np.uint8)
    with Image.fromarray(noise).convert("RGB") as image:
        data = _encode(image, "JPEG", quality=100)
    assert MAX_IMAGE_BYTES < len(data) < MAX_SOURCE_BYTES
    part = _part(data, "jpeg")
    batch = ImageBatch(policy_version=3)
    upload = preflight([part], DoclingSettings(), batch)[0]
    assert len(upload.data) < MAX_IMAGE_BYTES and len(batch.images[0]) < MAX_IMAGE_BYTES
    for parts, settings, version in [
        ([part], DoclingSettings(inference_mode="remote"), 2),
        ([part], DoclingSettings(file_bytes=MAX_IMAGE_BYTES), 3),
        ([part], DoclingSettings(total_bytes=len(data) - 1), 3),
        ([part, part], DoclingSettings(), 3),
        (
            [_part(data.ljust(MAX_SOURCE_BYTES + 1, b"\0"), "jpeg")],
            DoclingSettings(file_bytes=40 * 1_048_576),
            3,
        ),
    ]:
        with pytest.raises(DocumentError) as error:
            preflight(parts, settings, ImageBatch(policy_version=version))
        assert error.value.status == 413


@pytest.mark.parametrize("size", [(10001, 64), (10000, 5001)])
def test_source_dimensions_rejected_before_full_decode(size):
    with Image.new("RGB", size, "white") as image:
        part = _part(_encode(image, "JPEG"), "jpeg")
    with pytest.raises(DocumentError) as error:
        preflight([part], DoclingSettings(), ImageBatch(policy_version=3))
    assert error.value.status == 413


@pytest.mark.parametrize("kind", ["brand", "track"])
def test_heif_with_sequence_brand_or_movie_track_is_not_a_still(kind):
    data = _encode(Image.new("RGB", (64, 64), "white"))
    data = data[:8] + b"hevc" + data[12:] if kind == "brand" else data + b"\0\0\0\x08moov"
    with pytest.raises(DocumentError) as error:
        preflight([_part(data)], DoclingSettings(), ImageBatch(policy_version=3))
    assert error.value.status == 400


@pytest.mark.parametrize("format", ["HEIF", "PNG", "GIF", "WEBP", "AVIF", "TIFF"])
def test_no_sequence_selection_or_other_codecs(format):
    image = Image.new("RGB", (64, 64), "white")
    data = _encode(
        image,
        format,
        save_all=True,
        append_images=[Image.new("RGB", image.size, "black")],
        primary_index=1,
    )
    with pytest.raises(DocumentError) as error:
        preflight(
            [_part(data, "png" if format == "PNG" else "heif")],
            DoclingSettings(),
            ImageBatch(policy_version=3),
        )
    assert error.value.status == 400
