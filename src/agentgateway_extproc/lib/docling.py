"""Shared, fail-closed native Docling async client with no waiting queue."""

from __future__ import annotations

import asyncio
import logging
import re
import ssl
import threading
import time

import httpx

from agentgateway_extproc.config.settings import DoclingSettings
from agentgateway_extproc.lib.documents import (
    _MIMES,
    MAX_TEXT,
    DocumentError,
    ImageBatch,
    Upload,
    inspect_canonical_faces,
    preflight,
    project_document,
)
from agentgateway_extproc.lib.image_inspection import (
    ImageInspectionClient,
    ImageInspectionTimeoutError,
)
from agentgateway_extproc.lib.pipeline.mcp import strict_json_loads
from agentgateway_extproc.models.engine import EngineAttachmentPart

_STATUSES = {"pending", "started", "success", "partial_success", "failure", "skipped"}
_ACTIVE = {"pending", "started"}
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_logger = logging.getLogger(__name__)


class DoclingClient:
    """Admit one batch until all local work and its native job have stopped."""

    def __init__(
        self,
        settings: DoclingSettings,
        client: httpx.AsyncClient | None = None,
        image_inspection: ImageInspectionClient | None = None,
    ) -> None:
        """Own one verified HTTP client; an injected transport is for local tests."""
        self.settings = settings
        self._client = client
        self._owns_client = client is None
        if client is None and settings.enabled:
            self._client = httpx.AsyncClient(
                verify=ssl.create_default_context(cafile=settings.ca_cert),
                trust_env=False,
                follow_redirects=False,
                timeout=httpx.Timeout(30),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            )
        self._task: asyncio.Task[dict[int, str]] | None = None
        self._abandoned = threading.Event()
        self._poisoned = False
        self._closed = False
        self._active_stage = "extraction"
        self.image_inspection = image_inspection

    async def convert(
        self,
        parts: list[EngineAttachmentPart],
        *,
        images: ImageBatch | None = None,
        inspect_images: set[int] | None = None,
    ) -> list[str]:
        """Extract every attachment, preserving the legacy ordered result."""
        texts = await self.convert_selected(
            parts,
            set(range(len(parts))),
            images=images,
            inspect_images=inspect_images,
        )
        return list(texts.values())

    async def convert_selected(
        self,
        parts: list[EngineAttachmentPart],
        selected: set[int],
        *,
        images: ImageBatch | None = None,
        inspect_images: set[int] | None = None,
    ) -> dict[int, str]:
        """Admit one batch; an empty selection normalizes images without Docling."""
        if (
            (selected and not self.settings.enabled)
            or self._closed
            or self._poisoned
            or self._task is not None
        ):
            raise DocumentError(reason="extraction_unavailable")
        abandoned = threading.Event()
        self._abandoned = abandoned
        self._active_stage = "extraction"
        deadline = time.monotonic() + self.settings.timeout
        task = asyncio.create_task(
            self._batch(parts, selected, abandoned, deadline, images, inspect_images or set())
        )
        self._task = task
        task.add_done_callback(self._finished)
        try:
            async with asyncio.timeout(self.settings.timeout):
                return await asyncio.shield(task)
        except TimeoutError:
            abandoned.set()
            reason = (
                "image_inspection_timeout"
                if self._active_stage == "image_inspection"
                else "extraction_timeout"
            )
            raise DocumentError(504, reason=reason) from None
        except asyncio.CancelledError:
            abandoned.set()
            raise

    def _finished(self, task: asyncio.Task[dict[int, str]]) -> None:
        # Retrieve errors even after the requesting stream has gone away.
        if not task.cancelled():
            task.exception()
        self._task = None

    async def close(self) -> None:
        """Stop admission and allow at most five seconds of native-job draining."""
        self._closed = True
        self._abandoned.set()
        if self._task is not None:
            await asyncio.wait({self._task}, timeout=5)
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def _batch(  # noqa: C901
        self,
        parts: list[EngineAttachmentPart],
        selected: set[int],
        abandoned: threading.Event,
        deadline: float,
        images: ImageBatch | None = None,
        inspect_images: set[int] | None = None,
    ) -> dict[int, str]:
        """Validate every part before extracting the selected uploads in order."""
        try:
            uploads = await asyncio.to_thread(
                preflight, parts, self.settings, images, abandoned, deadline
            )
        except DocumentError:
            raise
        except Exception:  # noqa: BLE001
            # Third-party parsers have multiple malformed-input exception types.
            raise DocumentError(400) from None
        del parts
        # The retained task owns the thread and native job, not the caller timeout.
        output: dict[int, str] = {}
        pages = size = 0
        for index, upload in enumerate(uploads):
            if index not in selected:
                upload.data = b""
                continue
            if abandoned.is_set():
                break
            if time.monotonic() >= deadline:
                raise DocumentError(504, reason="extraction_timeout")
            canonical_image = (
                upload.data
                if index in (inspect_images or set()) and upload.format == "img"
                else None
            )
            self._active_stage = "extraction"
            document = await self._convert_one(upload, abandoned)
            upload.data = b""
            if abandoned.is_set():
                break
            text, count = await asyncio.to_thread(
                project_document,
                document,
                upload,
                self.settings.pages - pages,
                allow_empty_image=images is not None and images.policy_version >= 3,
            )
            if images is not None and upload.format == "img":
                if index in (inspect_images or set()):
                    if canonical_image is None:
                        raise DocumentError(reason="image_inspection_failed")
                    self._active_stage = "image_inspection"
                    await self._inspect_image(index, canonical_image, images, abandoned, deadline)
                    if images.protect_faces and images.defer_face_inspection:
                        if abandoned.is_set() or time.monotonic() >= deadline:
                            raise DocumentError(504, reason="image_inspection_timeout")
                        self._active_stage = "image_analysis"
                        await asyncio.to_thread(
                            inspect_canonical_faces, images, index, canonical_image
                        )
                inspection = images.inspections.get(index)
                transcription = (
                    inspection.transcription
                    if inspection is not None and inspection.outcome == "text_extracted"
                    else None
                )
                images.text_present[index] = bool(text.strip() or transcription)
                if transcription is not None:
                    text = (
                        f"{text}\n\n[Image inspection transcription]\n{transcription}"
                        if text
                        else transcription
                    )
                text = text or "[Image: no text extracted]"
            document = None
            pages += count
            size += len(text)
            if size > MAX_TEXT:
                raise DocumentError(413)
            output[index] = text
        return output

    async def _inspect_image(
        self,
        index: int,
        png: bytes,
        images: ImageBatch,
        abandoned: threading.Event,
        deadline: float,
    ) -> None:
        """Inspect one canonical image after its complete Docling projection."""
        if abandoned.is_set():
            return
        if time.monotonic() >= deadline:
            raise DocumentError(504, reason="image_inspection_timeout")
        inspector = self.image_inspection
        if inspector is None:
            raise DocumentError(reason="image_inspection_failed")
        try:
            async with asyncio.timeout(deadline - time.monotonic()):
                result = await inspector.inspect(png)
        except (TimeoutError, ImageInspectionTimeoutError):
            raise DocumentError(504, reason="image_inspection_timeout") from None
        if abandoned.is_set() or time.monotonic() >= deadline:
            raise DocumentError(504, reason="image_inspection_timeout")
        images.inspections[index] = result
        if result.outcome == "unreadable":
            raise DocumentError(403, reason="image_inspection_unreadable")
        if result.outcome == "failed":
            raise DocumentError(reason="image_inspection_failed")

    async def _convert_one(self, upload: Upload, abandoned: threading.Event) -> object:
        terminal = False
        try:
            value = await self._http("POST", "/v1/convert/file/async", limit=65_536, upload=upload)
            upload.data = b""
            task_id, status = _task_status(value)
            while status in _ACTIVE:
                # The native job is not cancelled by an HTTP/caller timeout.
                await asyncio.sleep(0.1)
                value = await self._http("GET", f"/v1/status/poll/{task_id}?wait=2", limit=65_536)
                _, status = _task_status(value, task_id)
            terminal = True
        finally:
            if not terminal:
                # A lost POST reply or status does not prove native capacity is free.
                self._poisoned = True
                _logger.warning("Docling admission disabled: native job completion is unknown")
        if status != "success":
            raise DocumentError
        if abandoned.is_set():
            return None
        value = await self._http(
            "GET", f"/v1/result/{task_id}", limit=self.settings.max_response_bytes
        )
        if (
            not isinstance(value, dict)
            or value.get("status") != "success"
            or value.get("errors") != []
            or not isinstance(value.get("document"), dict)
        ):
            raise DocumentError
        return value["document"].get("json_content")

    async def _http(
        self, method: str, path: str, *, limit: int, upload: Upload | None = None
    ) -> object:
        client = self._client
        key = self.settings.api_key
        if client is None or key is None:
            raise DocumentError(reason="extraction_unavailable")
        files = None
        options = None
        if upload is not None:
            source_format = "md" if upload.format == "txt" else upload.format
            extension = "png" if source_format == "img" else source_format
            mime = "image/png" if source_format == "img" else _MIMES[source_format]
            files = {"files": (f"upload.{extension}", upload.data, mime)}
            options = self._options(source_format)
        try:
            async with (
                asyncio.timeout(30),
                client.stream(
                    method,
                    self.settings.base_url.rstrip("/") + path,
                    headers={"X-Api-Key": key.get_secret_value(), "Accept-Encoding": "identity"},
                    files=files,
                    data=options,
                    follow_redirects=False,
                ) as response,
            ):
                if not response.is_success:
                    raise DocumentError(reason="extraction_unavailable")
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise DocumentError
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(content) + len(chunk) > limit:
                        raise DocumentError
                    content.extend(chunk)
                return strict_json_loads(content.decode("utf-8"))
        except (TimeoutError, httpx.TimeoutException):
            raise DocumentError(504) from None
        except httpx.HTTPError:
            raise DocumentError(reason="extraction_unavailable") from None
        except ValueError:
            raise DocumentError from None

    def _options(self, source_format: str) -> dict[str, str]:
        options = {
            "from_formats": "image" if source_format == "img" else source_format,
            "target_type": "inbody",
            "to_formats": "json",
            "image_export_mode": "placeholder",
            "include_images": "false",
            "include_page_images": "false",
            "abort_on_error": "true",
            "document_timeout": str(self.settings.document_timeout),
            "do_code_enrichment": "false",
            "do_formula_enrichment": "false",
            "do_chart_extraction": "false",
            "do_picture_classification": "false",
            "do_picture_description": "false",
            "do_pdf_heading_hierarchy": "false",
        }
        if self.settings.inference_mode in {"cpu", "internal-standard"}:
            options.update(
                pipeline="standard", ocr_preset="rapidocr", do_ocr="true", do_table_structure="true"
            )
        else:
            options.update(
                pipeline="vlm",
                vlm_pipeline_preset="images" if source_format == "img" else "default",
            )
        return options


def _task_status(value: object, expected_id: str | None = None) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise DocumentError
    task_id, status = value.get("task_id"), value.get("task_status")
    if (
        not isinstance(task_id, str)
        or not _TASK_ID.fullmatch(task_id)
        or (expected_id is not None and task_id != expected_id)
        or value.get("task_type") != "convert"
        or not isinstance(status, str)
        or status not in _STATUSES
    ):
        raise DocumentError
    return task_id, status
