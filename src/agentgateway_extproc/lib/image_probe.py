"""Normalize images and detect faces in a killable, resource-limited process."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import resource
import sys
import warnings
from pathlib import Path

MAX_IMAGE_BYTES = 5 * 1_048_576
MAX_SOURCE_BYTES = 20 * 1_048_576
MAX_SOURCE_PIXELS = 50_000_000
MAX_SOURCE_DIMENSION = 10_000
MAX_PIXELS = 12_000_000
MAX_DIMENSION = 4096
MAX_DETECTION_PIXELS = 2_000_000
MAX_DETECTION_DIMENSION = 2048
MIN_DETECTION_DIMENSION = 64
MAX_FACES = 5000
COUNT_HEADER = b"IMG3"
MODEL_PATH = Path(__file__).resolve().parents[1] / "assets/face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def normalize(
    data: bytes, expected: str, *, protect_faces: bool = False, policy_version: int = 2
) -> bytes:
    """Rebuild single-frame, oriented RGB pixels without source metadata."""
    from PIL import Image, ImageOps

    # Enforce our own stricter dimensions before load(), with one stable limit error.
    Image.MAX_IMAGE_PIXELS = None
    modern = policy_version >= 3
    if expected == "HEIF":
        if not modern:
            raise ValueError("unsupported image format")  # noqa: TRY003
        _register_heif(data)
    elif expected not in {"JPEG", "PNG"} and (expected != "WEBP" or policy_version != 4):
        raise ValueError("unsupported image format")  # noqa: TRY003
    pixel_limit = (
        MAX_SOURCE_PIXELS if modern else MAX_DETECTION_PIXELS if protect_faces else MAX_PIXELS
    )
    dimension_limit = (
        MAX_SOURCE_DIMENSION
        if modern
        else MAX_DETECTION_DIMENSION
        if protect_faces
        else MAX_DIMENSION
    )
    with Image.open(io.BytesIO(data), formats=[expected]) as image:
        width, height = image.size
        if (
            width * height > pixel_limit
            or max(width, height) > dimension_limit
            or (protect_faces and min(width, height) < MIN_DETECTION_DIMENSION)
        ):
            raise MemoryError
        if getattr(image, "n_frames", 1) != 1 or min(width, height) < 1:
            raise ValueError("invalid image frames")  # noqa: TRY003
        if modern:
            # One canonical reduction, before full-size orientation/RGBA copies.
            # Every reader, detector and downstream destination gets these pixels.
            scale = min(
                1,
                MAX_DETECTION_DIMENSION / max(width, height),
                math.sqrt(MAX_DETECTION_PIXELS / (width * height)),
            )
            image.thumbnail(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
            if protect_faces and min(image.size) < MIN_DETECTION_DIMENSION:
                raise MemoryError
        image.load()
        # HEIF's plugin already applies irot/imir and clears descriptive EXIF
        # orientation; this therefore does not rotate HEIF a second time.
        ImageOps.exif_transpose(image, in_place=True)
        oriented = image.convert("RGBA")
        background = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
        background.alpha_composite(oriented)
        rgb = background.convert("RGB")
        clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
        output = io.BytesIO()
        clean.save(output, format="PNG")
    if output.tell() > MAX_IMAGE_BYTES:
        raise MemoryError
    return output.getvalue()


def _register_heif(data: bytes) -> None:
    from pi_heif import register_heif_opener

    box_size = int.from_bytes(data[:4], "big")
    if data[4:8] != b"ftyp" or not 16 <= box_size <= min(len(data), 1024) or box_size % 4:
        raise ValueError("invalid HEIF header")  # noqa: TRY003
    brands = {data[8:12], *(data[offset : offset + 4] for offset in range(16, box_size, 4))}
    still = {b"heic", b"heix", b"heim", b"heis"}
    if (
        data[8:12] not in still | {b"mif1"}
        or not brands & still
        or brands & {b"avif", b"avis", b"msf1", b"hevc", b"hevx", b"hevm", b"hevs"}
    ):
        raise ValueError("unsupported HEIF container")  # noqa: TRY003
    # libheif's top-level photo count excludes timed tracks. A still cover image
    # in a movie container must not bypass the no-animation rule.
    offset = 0
    for _ in range(1024):
        size = int.from_bytes(data[offset : offset + 4], "big")
        header_size = 16 if size == 1 else 8
        if data[offset + 4 : offset + 8] in {b"moov", b"moof"}:
            raise ValueError("HEIF tracks are unsupported")  # noqa: TRY003
        if size == 1:
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
        elif size == 0:
            size = len(data) - offset
        if size < header_size or offset + size > len(data):
            raise ValueError("invalid HEIF box")  # noqa: TRY003
        offset += size
        if offset == len(data):
            break
    else:
        raise MemoryError
    # n_frames counts independent top-level photos, not tiles, alpha, depth or
    # gain-map auxiliaries. Decode only the sole primary still, never auxiliaries.
    # The pinned plugin converts HDR 10/12-bit to 8-bit, without gain-map HDR.
    register_heif_opener(
        decode_threads=1,
        thumbnails=False,
        depth_images=False,
        aux_images=False,
        preferred_decoder={"HEIF": "libde265"},
    )


def count_faces(data: bytes) -> int:
    """Load only the checked-in model; ambiguous native output is an error."""
    import cv2
    import numpy as np

    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("invalid face model")  # noqa: TRY003
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid canonical image")  # noqa: TRY003
        height, width = image.shape[:2]
        if (
            width * height > MAX_DETECTION_PIXELS
            or max(width, height) > MAX_DETECTION_DIMENSION
            or min(width, height) < MIN_DETECTION_DIMENSION
        ):
            raise MemoryError
        detector = cv2.FaceDetectorYN.create(
            str(MODEL_PATH),
            "",
            (width, height),
            0.5,
            0.3,
            MAX_FACES,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        status, faces = detector.detect(image)
    except cv2.error as exc:
        if exc.code == cv2.Error.StsNoMem:
            raise MemoryError from None
        raise
    if status != 1:
        raise ValueError("invalid detector status")  # noqa: TRY003
    if faces is None:
        return 0
    if (
        faces.ndim != 2
        or faces.shape[1] != 15
        or not 1 <= len(faces) <= MAX_FACES
        or not np.isfinite(faces).all()
        or (faces[:, 2:4] <= 0).any()
        or (faces[:, 14] < 0).any()
        or (faces[:, 14] > 1).any()
    ):
        raise ValueError("invalid detections")  # noqa: TRY003
    return len(faces)


def has_faces(data: bytes) -> bool:
    """Keep the shipped boolean interface for version-two callers."""
    return bool(count_faces(data))


def main() -> int:
    """Emit a bounded count (v3) or legacy flag, followed by canonical PNG bytes."""
    output = os.dup(1)
    os.dup2(2, 1)
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1_048_576, 768 * 1_048_576))
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        logging.disable(logging.CRITICAL)
        warnings.simplefilter("error")
        version = int(sys.argv[3]) if len(sys.argv) == 4 else 2
        limit = MAX_SOURCE_BYTES if version >= 3 else MAX_IMAGE_BYTES
        data = sys.stdin.buffer.read(limit + 1)
        if len(data) > limit:
            return 2
        normalized = normalize(
            data, sys.argv[1], protect_faces=sys.argv[2] == "detect", policy_version=version
        )
        try:
            count = count_faces(normalized) if sys.argv[2] == "detect" else 0
        except MemoryError:
            return 2
        except Exception:  # noqa: BLE001
            return 3
        with os.fdopen(output, "wb", closefd=False) as stream:
            header = (
                COUNT_HEADER + count.to_bytes(2, "big") if version >= 3 else bytes([bool(count)])
            )
            stream.write(header + normalized)
    except MemoryError:
        return 2
    except Exception:  # noqa: BLE001
        return 1
    finally:
        os.close(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
