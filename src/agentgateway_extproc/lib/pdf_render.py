"""Render complete PDFs in a disposable, resource-limited process."""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import resource
import sys
from contextlib import closing

from agentgateway_extproc.lib.image_probe import (
    MAX_DETECTION_DIMENSION,
    MAX_DETECTION_PIXELS,
    MAX_IMAGE_BYTES,
    MAX_SOURCE_DIMENSION,
    MAX_SOURCE_PIXELS,
)

MAX_RENDERED_PAGES = 5
HARD_MAX_RENDERED_PAGES = 20


def render(data: bytes, max_pages: int = MAX_RENDERED_PAGES) -> list[str]:
    """Return all pages in order as clean RGB PNGs under a total wire budget."""
    import pypdfium2 as pdfium
    from PIL import Image

    if type(max_pages) is not int or not 1 <= max_pages <= HARD_MAX_RENDERED_PAGES:
        raise MemoryError
    result: list[str] = []
    size = 0
    with pdfium.PdfDocument(data) as document:
        if not 1 <= len(document) <= max_pages:
            raise MemoryError
        for index in range(len(document)):
            with closing(document[index]) as page:
                width, height = page.get_size()
                if (
                    not all(math.isfinite(value) and value > 0 for value in (width, height))
                    or max(width, height) * 2 > MAX_SOURCE_DIMENSION
                    or width * height * 4 > MAX_SOURCE_PIXELS
                ):
                    raise MemoryError
                scale = min(
                    2,
                    MAX_DETECTION_DIMENSION / max(width, height),
                    math.sqrt(MAX_DETECTION_PIXELS / (width * height)),
                )
                # Ceil rounding in PDFium must not exceed the canonical pixel budget.
                while math.ceil(width * scale) * math.ceil(height * scale) > MAX_DETECTION_PIXELS:
                    scale *= 0.999
                with closing(page.render(scale=scale)) as bitmap:
                    image = bitmap.to_pil().convert("RGB")
                    clean = Image.frombytes("RGB", image.size, image.tobytes())
                    output = io.BytesIO()
                    clean.save(output, format="PNG")
                    uri = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
                size += len(uri)
                if size > MAX_IMAGE_BYTES:
                    raise MemoryError
                result.append(uri)
    return result


def main() -> int:
    """Emit only bounded JSON; silence native and Python library diagnostics."""
    output = os.dup(1)
    os.dup2(2, 1)
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1_048_576, 512 * 1_048_576))
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        logging.disable(logging.CRITICAL)
        data = sys.stdin.buffer.read(40 * 1_048_576 + 1)
        if len(data) > 40 * 1_048_576:
            return 2
        max_pages = int(sys.argv[1]) if len(sys.argv) == 2 else MAX_RENDERED_PAGES
        payload = json.dumps(render(data, max_pages), separators=(",", ":")).encode()
        with os.fdopen(os.dup(output), "wb") as stream:
            stream.write(payload)
    except MemoryError:
        return 2
    except Exception:  # noqa: BLE001 - native parser errors never escape the process
        return 1
    finally:
        os.close(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
