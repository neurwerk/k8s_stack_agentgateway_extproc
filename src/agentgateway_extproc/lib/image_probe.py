"""Normalize images and detect faces in a killable, resource-limited process."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import resource
import sys
import warnings
from pathlib import Path

MAX_IMAGE_BYTES = 5 * 1_048_576
MAX_PIXELS = 12_000_000
MAX_DIMENSION = 4096
MAX_DETECTION_PIXELS = 2_000_000
MAX_DETECTION_DIMENSION = 2048
MIN_DETECTION_DIMENSION = 64
MODEL_PATH = Path(__file__).resolve().parents[1] / "assets/face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def normalize(data: bytes, expected: str, *, protect_faces: bool = False) -> bytes:
    """Rebuild single-frame, oriented RGB pixels without source metadata."""
    from PIL import Image, ImageOps

    # Enforce our own stricter dimensions before load(), with one stable limit error.
    Image.MAX_IMAGE_PIXELS = None
    pixel_limit = MAX_DETECTION_PIXELS if protect_faces else MAX_PIXELS
    dimension_limit = MAX_DETECTION_DIMENSION if protect_faces else MAX_DIMENSION
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
        image.load()
        oriented = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
        background.alpha_composite(oriented)
        rgb = background.convert("RGB")
        clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
        output = io.BytesIO()
        clean.save(output, format="PNG")
    if output.tell() > MAX_IMAGE_BYTES:
        raise MemoryError
    return output.getvalue()


def has_faces(data: bytes) -> bool:
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
            5000,
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
        return False
    if (
        faces.ndim != 2
        or faces.shape[1] != 15
        or not len(faces)
        or not np.isfinite(faces).all()
        or (faces[:, 2:4] <= 0).any()
        or (faces[:, 14] < 0).any()
        or (faces[:, 14] > 1).any()
    ):
        raise ValueError("invalid detections")  # noqa: TRY003
    return len(faces) > 0


def main() -> int:
    """Emit one flag plus bounded PNG bytes, never native diagnostics or raw input."""
    output = os.dup(1)
    os.dup2(2, 1)
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1_048_576, 768 * 1_048_576))
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        logging.disable(logging.CRITICAL)
        warnings.simplefilter("error")
        data = sys.stdin.buffer.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            return 2
        normalized = normalize(data, sys.argv[1], protect_faces=sys.argv[2] == "detect")
        try:
            face = has_faces(normalized) if sys.argv[2] == "detect" else False
        except MemoryError:
            return 2
        except Exception:  # noqa: BLE001
            return 3
        with os.fdopen(output, "wb", closefd=False) as stream:
            stream.write(bytes([face]) + normalized)
    except MemoryError:
        return 2
    except Exception:  # noqa: BLE001
        return 1
    finally:
        os.close(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
