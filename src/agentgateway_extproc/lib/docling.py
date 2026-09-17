"""Shared, fail-closed native Docling async client with no waiting queue."""

from __future__ import annotations

import asyncio
import logging
import re
import ssl

import httpx

from agentgateway_extproc.config.settings import DoclingSettings
from agentgateway_extproc.lib.documents import (
    _MIMES,
    MAX_TEXT,
    DocumentError,
    Upload,
    preflight,
    project_document,
)
from agentgateway_extproc.lib.pipeline.mcp import strict_json_loads
from agentgateway_extproc.models.engine import EngineAttachmentPart

_STATUSES = {"pending", "started", "success", "partial_success", "failure", "skipped"}
_ACTIVE = {"pending", "started"}
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_logger = logging.getLogger(__name__)


class DoclingClient:
    """Admit one batch until all local work and its native job have stopped."""

    def __init__(self, settings: DoclingSettings, client: httpx.AsyncClient | None = None) -> None:
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
        self._task: asyncio.Task[list[str]] | None = None
        self._abandoned = asyncio.Event()
        self._poisoned = False
        self._closed = False

    async def convert(self, parts: list[EngineAttachmentPart]) -> list[str]:
        """Fail busy immediately and shield admitted work from caller cancellation."""
        if not self.settings.enabled or self._closed or self._poisoned or self._task is not None:
            raise DocumentError
        abandoned = asyncio.Event()
        self._abandoned = abandoned
        deadline = asyncio.get_running_loop().time() + self.settings.timeout
        task = asyncio.create_task(self._batch(parts, abandoned, deadline))
        self._task = task
        task.add_done_callback(self._finished)
        try:
            async with asyncio.timeout_at(deadline):
                return await asyncio.shield(task)
        except TimeoutError:
            abandoned.set()
            raise DocumentError(504) from None
        except asyncio.CancelledError:
            abandoned.set()
            raise

    def _finished(self, task: asyncio.Task[list[str]]) -> None:
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

    async def _batch(
        self, parts: list[EngineAttachmentPart], abandoned: asyncio.Event, deadline: float
    ) -> list[str]:
        try:
            uploads = await asyncio.to_thread(preflight, parts, self.settings)
        except DocumentError:
            raise
        except Exception:  # noqa: BLE001
            # Third-party parsers have multiple malformed-input exception types.
            raise DocumentError(400) from None
        del parts
        # The retained task owns the thread and native job, not the caller timeout.
        output: list[str] = []
        pages = size = 0
        for upload in uploads:
            if abandoned.is_set():
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise DocumentError(504)
            document = await self._convert_one(upload, abandoned)
            upload.data = b""
            if abandoned.is_set():
                break
            text, count = await asyncio.to_thread(
                project_document, document, upload, self.settings.pages - pages
            )
            document = None
            pages += count
            size += len(text)
            if size > MAX_TEXT:
                raise DocumentError(413)
            output.append(text)
        return output

    async def _convert_one(self, upload: Upload, abandoned: asyncio.Event) -> object:
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
            raise DocumentError
        files = None
        options = None
        if upload is not None:
            source_format = "md" if upload.format == "txt" else upload.format
            files = {"files": (f"upload.{source_format}", upload.data, _MIMES[source_format])}
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
                if (
                    not response.is_success
                    or response.headers.get("content-encoding", "identity") != "identity"
                ):
                    raise DocumentError
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(content) + len(chunk) > limit:
                        raise DocumentError
                    content.extend(chunk)
                return strict_json_loads(content.decode("utf-8"))
        except (TimeoutError, httpx.HTTPError, ValueError):
            raise DocumentError from None

    def _options(self, source_format: str) -> dict[str, str]:
        options = {
            "from_formats": source_format,
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
        if self.settings.inference_mode == "cpu":
            options.update(
                pipeline="standard", ocr_preset="rapidocr", do_ocr="true", do_table_structure="true"
            )
        else:
            options.update(pipeline="vlm", vlm_pipeline_preset="default")
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
