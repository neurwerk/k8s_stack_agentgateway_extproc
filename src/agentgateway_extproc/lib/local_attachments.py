"""Independent admission for local unchecked image normalization and PDF rendering."""

from __future__ import annotations

import asyncio
import base64
import json
import signal
import subprocess
import sys
import threading
import time
from typing import cast

from agentgateway_extproc.config.settings import DoclingSettings
from agentgateway_extproc.lib.documents import (
    DocumentError,
    ImageBatch,
    Upload,
    _check_preflight,
    preflight,
)
from agentgateway_extproc.lib.image_probe import MAX_IMAGE_BYTES
from agentgateway_extproc.lib.pdf_render import HARD_MAX_RENDERED_PAGES, MAX_RENDERED_PAGES
from agentgateway_extproc.models.engine import EngineAttachmentPart


def render_pdf(
    upload: Upload, abandoned: threading.Event, deadline: float, max_pages: int = MAX_RENDERED_PAGES
) -> list[str]:
    """Reap a bounded child before releasing the local worker's admission slot."""
    _check_preflight(abandoned, deadline)
    if type(max_pages) is not int or not 1 <= max_pages <= HARD_MAX_RENDERED_PAGES:
        raise DocumentError(413)
    try:
        result = subprocess.run(  # noqa: S603 - fixed module and validated integer argument
            [
                sys.executable,
                "-I",
                "-B",
                "-m",
                "agentgateway_extproc.lib.pdf_render",
                str(max_pages),
            ],
            input=upload.data,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={},
            timeout=min(15, max(0.001, deadline - time.monotonic())),
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise DocumentError(504, reason="pdf_render_timeout") from None
    except OSError:
        raise DocumentError(reason="pdf_render_failed") from None
    _check_preflight(abandoned, deadline)
    if result.returncode in {2, -signal.SIGKILL, -signal.SIGXCPU}:
        raise DocumentError(413)
    if result.returncode or len(result.stdout) > MAX_IMAGE_BYTES + 64:
        raise DocumentError(400, reason="pdf_render_failed")
    return _rendered_pages(result.stdout, upload.pages, max_pages)


def _rendered_pages(data: bytes, expected: int, max_pages: int) -> list[str]:
    """Validate the isolated renderer's complete, bounded PNG response."""
    try:
        pages = json.loads(data)
    except (ValueError, UnicodeError):
        raise DocumentError(reason="pdf_render_failed") from None
    if (
        not isinstance(pages, list)
        or len(pages) != expected
        or not 1 <= len(pages) <= max_pages
        or any(
            not isinstance(uri, str) or not uri.startswith("data:image/png;base64,")
            for uri in pages
        )
        or sum(map(len, pages)) > MAX_IMAGE_BYTES
    ):
        raise DocumentError(reason="pdf_render_failed")
    try:
        for uri in pages:
            encoded = uri.removeprefix("data:image/png;base64,")
            png = base64.b64decode(encoded, validate=True)
            if (
                not png.startswith(b"\x89PNG\r\n\x1a\n")
                or base64.b64encode(png).decode() != encoded
            ):
                raise DocumentError(reason="pdf_render_failed")
    except ValueError:
        raise DocumentError(reason="pdf_render_failed") from None
    return cast(list[str], pages)


class LocalAttachments:
    """Hold a no-queue local slot until timed-out or cancelled workers have drained."""

    def __init__(self, settings: DoclingSettings) -> None:
        """Own local work independently of Docling HTTP capacity."""
        self.settings = settings
        self._task: asyncio.Task[set[int]] | None = None
        self._abandoned = threading.Event()
        self._closed = False

    async def prepare(
        self,
        parts: list[EngineAttachmentPart],
        images: ImageBatch,
        pdf_mode: str | None,
        max_pages: int = MAX_RENDERED_PAGES,
    ) -> set[int]:
        """Validate all original inputs before rendering; return extraction indexes."""
        if self._closed or self._task is not None:
            raise DocumentError(reason="extraction_unavailable")
        abandoned = self._abandoned = threading.Event()
        deadline = time.monotonic() + self.settings.timeout
        task = asyncio.create_task(
            asyncio.to_thread(
                self._prepare, parts, images, pdf_mode, abandoned, deadline, max_pages
            )
        )
        self._task = task
        task.add_done_callback(self._finished)
        try:
            async with asyncio.timeout(self.settings.timeout):
                return await asyncio.shield(task)
        except TimeoutError:
            abandoned.set()
            raise DocumentError(504) from None
        except asyncio.CancelledError:
            abandoned.set()
            raise

    def _finished(self, task: asyncio.Task[set[int]]) -> None:
        if not task.cancelled():
            task.exception()
        self._task = None

    async def close(self) -> None:
        """Stop admission and drain the bounded local worker."""
        self._closed = True
        self._abandoned.set()
        if self._task is not None:
            await asyncio.gather(asyncio.shield(self._task), return_exceptions=True)

    def _prepare(
        self,
        parts: list[EngineAttachmentPart],
        images: ImageBatch,
        pdf_mode: str | None,
        abandoned: threading.Event,
        deadline: float,
        max_pages: int = MAX_RENDERED_PAGES,
    ) -> set[int]:
        uploads: list[Upload] = []
        try:
            uploads = preflight(parts, self.settings, images, abandoned, deadline)
            pdfs = {index for index, upload in enumerate(uploads) if upload.format == "pdf"}
            if pdfs and pdf_mode == "block":
                raise DocumentError(403, reason="attachments_disabled")  # noqa: TRY301
            rendered = pdfs if pdf_mode == "render-images" else set()
            page_count = sum(uploads[index].pages for index in rendered)
            if rendered and page_count + len(images.images) > max_pages:
                raise DocumentError(413)  # noqa: TRY301
            for index in sorted(rendered):
                images.pdf_images[index] = render_pdf(
                    uploads[index], abandoned, deadline, max_pages
                )
                if (
                    sum(len(uri) for uri in images.images.values())
                    + sum(len(uri) for pages in images.pdf_images.values() for uri in pages)
                    > MAX_IMAGE_BYTES
                ):
                    raise DocumentError(413)  # noqa: TRY301
            _check_preflight(abandoned, deadline)
            return set(range(len(parts))) - set(images.images) - rendered
        except DocumentError:
            raise
        except Exception:  # noqa: BLE001 - same malformed parser boundary as Docling preflight
            raise DocumentError(400, diagnostic="preflight_failed") from None
        finally:
            for upload in uploads:
                upload.data = b""
