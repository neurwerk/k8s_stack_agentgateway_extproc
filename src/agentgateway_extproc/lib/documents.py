"""In-memory upload preflight and an allowlisted Docling JSON text projection."""

from __future__ import annotations

import base64
import binascii
import csv
import io
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import cast
from xml.parsers import expat

from agentgateway_extproc.config.settings import MEBIBYTE, DoclingSettings
from agentgateway_extproc.lib.image_probe import (
    COUNT_HEADER,
    MAX_FACES,
    MAX_IMAGE_BYTES,
    MAX_SOURCE_BYTES,
)
from agentgateway_extproc.models.engine import EngineAttachmentPart

MAX_TEXT = 4_000_000
MAX_NODES = 20_000
MAX_CELLS = 100_000
_MIMES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "txt": "text/plain",
    "md": "text/markdown",
    "csv": "text/csv",
}
_OFFICE_PARTS = {
    "docx": "word/document.xml",
    "xlsx": "xl/workbook.xml",
    "pptx": "ppt/presentation.xml",
}


class DocumentError(Exception):
    """Expose only fixed, content-free errors at the caller boundary."""

    def __init__(self, status: int = 503, *, reason: str | None = None) -> None:
        """Select a fixed public error without retaining upstream exception text."""
        self.status = status
        self.reason = (
            reason
            or {
                400: "invalid_upload",
                403: "policy_blocked",
                413: "processing_limits",
                503: "extraction_failed",
                504: "extraction_timeout",
            }[status]
        )
        self.message = (
            "neurwerk: "
            + {
                "invalid_upload": "attachment could not be decoded or validated.",
                "unsupported_format": (
                    "attachment format is not supported by the configured processor."
                ),
                "attachments_disabled": "attachments are disabled for this model.",
                "policy_blocked": "request blocked by configured data policy.",
                "face_policy_blocked": "request blocked by configured face policy.",
                "processing_limits": "attachment exceeds configured processing limits.",
                "extraction_unavailable": "text extraction service unavailable.",
                "extraction_failed": "text extraction could not be completed.",
                "extraction_timeout": "text extraction timed out.",
                "image_analysis_failed": "required image safety analysis could not be completed.",
                "image_processing_unavailable": (
                    "configured extraction mode does not support images."
                ),
                "image_text_unavailable": (
                    "image text extraction only; no text extracted from an image. "
                    "Image forwarding is disabled for this model."
                ),
                "face_text_unavailable": (
                    "faces detected; configured policy permits sending only extracted text. "
                    "No text was extracted from an image."
                ),
                "image_analysis_text_unavailable": (
                    "image forwarding requires extracted text for PII analysis; "
                    "no text extracted from an image."
                ),
                "image_pii_detected": (
                    "image forwarding requires zero PII detections; PII was detected."
                ),
                "image_reroute_unavailable": (
                    "image rerouting required; no approved image route available."
                ),
            }[self.reason]
        )
        super().__init__(self.message)

    @property
    def no_text(self) -> bool:
        """Distinguish successful empty extraction from processing and policy failures."""
        return self.reason in {
            "image_text_unavailable",
            "face_text_unavailable",
            "image_analysis_text_unavailable",
        }


@dataclass
class Upload:
    """Retain only checked upload bytes and the safe display basename."""

    filename: str
    format: str
    data: bytes
    pages: int = 0
    source_bytes: int = 0


@dataclass
class ImageBatch:
    """Retain only normalized images under the existing batch admission slot."""

    protect_faces: bool = True
    images: dict[int, str] = dataclass_field(default_factory=dict)
    policy_version: int = 2
    face_count: int = 0
    scan_status: str = "not_scanned"
    text_present: dict[int, bool] = dataclass_field(default_factory=dict)


def preflight(  # noqa: C901 - publish scan provenance only after the complete batch passes
    parts: list[EngineAttachmentPart],
    settings: DoclingSettings,
    images: ImageBatch | None = None,
    abandoned: threading.Event | None = None,
    deadline: float = float("inf"),
) -> list[Upload]:
    """Validate every file before the first network call, under batch admission."""
    if len(parts) > settings.count:
        raise DocumentError(413)
    uploads: list[Upload] = []
    total = pages = 0
    for index, part in enumerate(parts):
        _check_preflight(abandoned, deadline)
        if part.type in {"image_url", "input_image"} and images is not None:
            upload = _decode_image(part, settings, images, abandoned, deadline)
            images.images[index] = "data:image/png;base64," + base64.b64encode(upload.data).decode()
            if sum(len(value) for value in images.images.values()) > MAX_IMAGE_BYTES:
                raise DocumentError(413)
        else:
            upload = _decode_upload(part, settings.file_bytes)
        total += max(len(upload.data), upload.source_bytes)
        if total > settings.total_bytes:
            raise DocumentError(413)
        if upload.format == "pdf":
            upload.pages = _pdf_pages(upload.data, abandoned, deadline)
        elif upload.format in _OFFICE_PARTS:
            _check_office(upload)
        elif upload.format != "img":
            _check_text(upload)
        pages += upload.pages
        if pages > settings.pages:
            raise DocumentError(413)
        uploads.append(upload)
    _check_preflight(abandoned, deadline)
    if images is not None:
        images.scan_status = "complete" if images.protect_faces and images.images else "not_scanned"
    return uploads


def _check_preflight(abandoned: threading.Event | None, deadline: float) -> None:
    if (abandoned is not None and abandoned.is_set()) or time.monotonic() >= deadline:
        raise DocumentError(504)


def _decode_image(
    part: EngineAttachmentPart,
    settings: DoclingSettings,
    images: ImageBatch,
    abandoned: threading.Event | None,
    deadline: float,
) -> Upload:
    modern = images.policy_version == 3
    if not modern and settings.inference_mode not in {"private-vlm", "remote"}:
        raise DocumentError(403, reason="image_processing_unavailable")
    limit = min(settings.file_bytes, MAX_SOURCE_BYTES if modern else MAX_IMAGE_BYTES)
    mime, data = _image_data(part, limit, policy_version=images.policy_version)
    _check_preflight(abandoned, deadline)
    normalized, count = _normalize_image(data, mime, images.protect_faces, images.policy_version)
    if len(normalized) > min(settings.file_bytes, MAX_IMAGE_BYTES):
        raise DocumentError(413)
    images.face_count += count
    return Upload("image.png", "img", normalized, pages=1, source_bytes=len(data))


def _image_data(
    part: EngineAttachmentPart, limit: int, *, policy_version: int = 2
) -> tuple[str, bytes]:
    value = part.model_dump()
    url_key = "image_url"
    if part.type == "image_url":
        if set(value) != {"type", "image_url"} or not isinstance(value["image_url"], dict):
            raise DocumentError(400)
        value, url_key = value["image_url"], "url"
    else:
        value.pop("type")
    # Caller detail is only a rendering hint, never a normalization/OCR setting.
    if policy_version == 3 and value.pop("detail", "auto") not in ("auto", "low", "high"):
        raise DocumentError(400)
    url = value.get(url_key)
    if set(value) != {url_key} or not isinstance(url, str):
        raise DocumentError(400)
    match = re.fullmatch(r"data:image/([a-z-]+);base64,([A-Za-z0-9+/]*={0,2})", url)
    allowed = (
        {"jpeg", "png", "heic", "heif", "x-heic", "x-heif"}
        if policy_version == 3
        else {"jpeg", "png"}
    )
    if match is None or match[1] not in allowed:
        raise DocumentError(400, reason="unsupported_format" if match else "invalid_upload")
    mime, encoded = match.groups()
    if len(encoded) > 4 * ((limit + 2) // 3):
        raise DocumentError(413)
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise DocumentError(400) from None
    if not data or base64.b64encode(data).decode() != encoded:
        raise DocumentError(400)
    if len(data) > limit:
        raise DocumentError(413)
    return cast(str, mime), data


def _normalize_image(
    data: bytes, mime: str, protect_faces: bool, policy_version: int = 2
) -> tuple[bytes, int]:
    try:
        result = subprocess.run(  # noqa: S603 - fixed executable/module and allowlisted arguments
            [
                sys.executable,
                "-I",
                "-B",
                "-m",
                "agentgateway_extproc.lib.image_probe",
                "JPEG" if mime == "jpeg" else "PNG" if mime == "png" else "HEIF",
                "detect" if protect_faces else "skip",
                *(["3"] if policy_version == 3 else []),
            ],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise DocumentError(413) from None
    except OSError:
        raise DocumentError(reason="image_analysis_failed") from None
    if result.returncode in {2, -signal.SIGKILL, -signal.SIGXCPU}:
        raise DocumentError(413)
    if result.returncode == 3:
        raise DocumentError(reason="image_analysis_failed")
    if result.returncode:
        raise DocumentError(400)
    if policy_version == 3:
        count = int.from_bytes(result.stdout[4:6], "big")
        valid = (
            result.stdout.startswith(COUNT_HEADER)
            and len(result.stdout) >= 6
            and count <= MAX_FACES
            and (protect_faces or count == 0)
        )
        normalized = result.stdout[6:]
    else:
        valid = result.stdout[:1] in {b"\x00", b"\x01"}
        count, normalized = int.from_bytes(result.stdout[:1], "big"), result.stdout[1:]
    if not valid:
        raise DocumentError(503 if policy_version == 3 else 400, reason="image_analysis_failed")
    if count and policy_version != 3:
        raise DocumentError(403, reason="face_policy_blocked")
    if not normalized.startswith(b"\x89PNG\r\n\x1a\n") or len(normalized) > MAX_IMAGE_BYTES:
        raise DocumentError(413)
    return normalized, count


def _check_text(upload: Upload) -> None:
    text = upload.data.decode("utf-8")
    if not text.strip() or "\x00" in text or upload.data.startswith((b"%PDF-", b"PK\x03\x04")):
        raise DocumentError(400)
    if upload.format == "csv":
        _check_csv(text)


def _pdf_pages(
    data: bytes, abandoned: threading.Event | None = None, deadline: float = float("inf")
) -> int:
    if not data.startswith(b"%PDF-") or not data.rstrip().endswith(b"%%EOF"):
        raise DocumentError(400)
    _check_preflight(abandoned, deadline)
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-m", "agentgateway_extproc.lib.pdf_probe"],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={},
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # run() kills and reaps the child before the admitted thread returns.
        raise DocumentError(413) from None
    except OSError:
        raise DocumentError from None
    if result.returncode in {2, -signal.SIGKILL, -signal.SIGXCPU}:
        raise DocumentError(413)
    if result.returncode or not re.fullmatch(rb"[1-9][0-9]{0,9}", result.stdout):
        raise DocumentError(400)
    return int(result.stdout)


def _filename(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 256
        or any(unicodedata.category(char).startswith("C") for char in value)
    ):
        raise DocumentError(400)
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _decode_upload(part: EngineAttachmentPart, limit: int) -> Upload:
    value = part.model_dump()
    if part.type == "file" and set(value) == {"type", "file"}:
        value = value["file"]
    elif part.type == "input_file":
        value.pop("type")
    else:
        raise DocumentError(400)
    if not isinstance(value, dict) or set(value) != {"filename", "file_data"}:
        raise DocumentError(400)
    filename, encoded = _filename(value["filename"]), value["file_data"]
    extension = filename.rsplit(".", 1)[-1].lower()
    mime = _MIMES.get(extension)
    if mime is None or "." not in filename:
        raise DocumentError(400, reason="unsupported_format")
    prefix = f"data:{mime};base64,"
    if not isinstance(encoded, str) or not encoded.startswith(prefix):
        raise DocumentError(400)
    encoded = encoded[len(prefix) :]
    if len(encoded) > 4 * ((limit + 2) // 3):
        raise DocumentError(413)
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise DocumentError(400) from None
    if not data or base64.b64encode(data).decode("ascii") != encoded:
        raise DocumentError(400)
    if len(data) > limit:
        raise DocumentError(413)
    return Upload(filename, extension, data)


def _check_office(upload: Upload) -> None:
    if not upload.data.startswith(b"PK\x03\x04"):
        raise DocumentError(400)
    with zipfile.ZipFile(io.BytesIO(upload.data)) as archive:
        entries = archive.infolist()
        if len(entries) > 10_000 or sum(item.file_size for item in entries) > 100 * MEBIBYTE:
            raise DocumentError(413)
        names: set[str] = set()
        for item in entries:
            name = item.filename
            if (
                name in names
                or name != item.orig_filename
                or name.startswith("/")
                or any(part in {".", "..", ""} for part in name.rstrip("/").split("/"))
                or "\\" in name
                or ":" in name
                or any(unicodedata.category(char).startswith("C") for char in name)
                or item.flag_bits & 1
                or stat.S_ISLNK(item.external_attr >> 16)
                or item.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            ):
                raise DocumentError(400)
            names.add(name)
            if item.file_size > 25 * MEBIBYTE:
                raise DocumentError(413)
            _check_zip_member(archive, item)
        if not {"[Content_Types].xml", "_rels/.rels", _OFFICE_PARTS[upload.format]} <= names or any(
            part in names for kind, part in _OFFICE_PARTS.items() if kind != upload.format
        ):
            raise DocumentError(400)
        if upload.format == "xlsx":
            _check_xlsx(archive)


def _check_zip_member(archive: zipfile.ZipFile, item: zipfile.ZipInfo) -> None:
    # Read to EOF in bounded chunks to verify the size and CRC, never extract.
    size = 0
    with archive.open(item) as member:
        while chunk := member.read(65_536):
            size += len(chunk)
            if size > min(item.file_size, 25 * MEBIBYTE):
                raise DocumentError(413)
    if size != item.file_size:
        raise DocumentError(400)


def _check_csv(text: str) -> None:
    # Bound delimiter/quote expansion before Sniffer or reader can allocate rows.
    if (
        sum(text.count(char) for char in ",;\t|:\r\n") > MAX_CELLS
        or sum(text.count(char) for char in "\"'") > 2 * MAX_CELLS
    ):
        raise DocumentError(413)
    stream = io.StringIO(text.removeprefix("\ufeff"))
    head = stream.readline()
    try:
        dialect = csv.Sniffer().sniff(head, ",;\t|:")
    except csv.Error:
        try:
            dialect = csv.Sniffer().sniff(stream.getvalue()[:4096], ",;\t|:")
        except csv.Error:
            dialect = csv.excel
    stream.seek(0)
    columns = 0
    for rows, row in enumerate(csv.reader(stream, dialect=dialect, strict=True), start=1):
        columns = max(columns, len(row))
        if rows * columns > MAX_CELLS:
            raise DocumentError(413)


def _range_bounds(value: str) -> tuple[int, int, int, int]:
    match = re.fullmatch(
        r"\$?([A-Z]{1,3})\$?([1-9][0-9]{0,6})(?::\$?([A-Z]{1,3})\$?([1-9][0-9]{0,6}))?",
        value,
        flags=re.ASCII | re.IGNORECASE,
    )
    if match is None:
        raise DocumentError(400)
    left, top, right, bottom = match.groups()
    cols: list[int] = []
    for letters in (left, right or left):
        col = 0
        for letter in letters.upper():
            col = col * 26 + ord(letter) - ord("A") + 1
        cols.append(col)
    r1, r2, c1, c2 = int(top), int(bottom or top), *cols
    if r1 > r2 or c1 > c2:
        raise DocumentError(400)
    if r2 > 1_048_576 or c2 > 16_384 or (r2 - r1 + 1) * (c2 - c1 + 1) > MAX_CELLS:
        raise DocumentError(413)
    return r1, c1, r2, c2


class _WorksheetBudget:
    """Bound the rectangle native iter_rows expands, not just populated cells."""

    def __init__(self) -> None:
        self.row = self.col = self.cells = self.merges = self.positions = 0
        self.bounds = (1_048_577, 16_385, 0, 0)

    def start(self, tag: str, attrs: dict[str, str]) -> None:
        """Check dimensions, real/implicit cell coordinates and merged ranges."""
        if tag == "row":
            value = attrs.get("r", str(self.row + 1))
            if not re.fullmatch(r"[1-9][0-9]{0,6}", value):
                raise DocumentError(400)
            self.row, self.col = int(value), 0
        elif tag in {"c", "dimension", "mergeCell", "hyperlink"}:
            reference = attrs.get("r" if tag == "c" else "ref")
            if reference is None and tag != "c":
                raise DocumentError(400)
            bounds = (
                _range_bounds(reference)
                if reference is not None
                else (self.row, self.col + 1, self.row, self.col + 1)
            )
            r1, c1, r2, c2 = bounds
            if tag == "c":
                self.col = c2
                self.cells += 1
            if tag in {"mergeCell", "hyperlink"}:
                self.merges += (r2 - r1 + 1) * (c2 - c1 + 1)
            if min(bounds) < 1 or r2 > 1_048_576 or c2 > 16_384:
                raise DocumentError(400)
            top, left, bottom, right = self.bounds
            self.bounds = min(top, r1), min(left, c1), max(bottom, r2), max(right, c2)
            top, left, bottom, right = self.bounds
            self.positions = (bottom - top + 1) * (right - left + 1)
            if max(self.positions, self.cells, self.merges) > MAX_CELLS:
                raise DocumentError(413)


def _check_xlsx(archive: zipfile.ZipFile) -> None:
    # Expat callbacks inspect XML without allocating an ElementTree. Scan all XML
    # parts, not just conventional sheet filenames, because relationships select them.
    nodes = depth = positions = 0
    sheet = _WorksheetBudget()
    chart = _ChartBudget()

    def start(name: str, attrs: dict[str, str]) -> None:
        nonlocal nodes, depth
        nodes += 1
        depth += 1
        if nodes > 2 * MAX_CELLS or depth > 32 or len(attrs) > 64:
            raise DocumentError(413)
        tag = name.rsplit("}", 1)[-1]
        if (
            tag == "Relationship"
            and attrs.get("Type", "").rsplit("/", 1)[-1] in {"worksheet", "chart"}
            and not attrs.get("Target", "").lower().endswith(".xml")
        ):
            raise DocumentError(400)
        sheet.start(tag, attrs)
        chart.start(name)

    def end(_name: str) -> None:
        nonlocal depth
        depth -= 1
        chart.end(_name)

    def reject_doctype(*_args: object) -> None:
        raise DocumentError(400)

    for name in archive.namelist():
        if not name.lower().endswith((".xml", ".rels")):
            continue
        sheet = _WorksheetBudget()
        parser = expat.ParserCreate(namespace_separator="}")
        parser.StartElementHandler = start
        parser.EndElementHandler = end
        parser.CharacterDataHandler = chart.text
        parser.StartDoctypeDeclHandler = reject_doctype
        with archive.open(name) as member:
            parser.ParseFile(member)
        positions += max(sheet.positions, sheet.cells, sheet.merges)
        if positions + chart.positions > MAX_CELLS:
            raise DocumentError(413)


class _ChartBudget:
    # The native XLSX backend resolves chart ranges even with chart enrichment off.
    def __init__(self) -> None:
        self.reference: str | None = None
        self.positions = 0

    def start(self, name: str) -> None:
        """Recognize native chart data-reference elements, not worksheet formulas."""
        if name in {
            "http://schemas.openxmlformats.org/drawingml/2006/chart}f",
            "http://purl.oclc.org/ooxml/drawingml/chart}f",
        }:
            self.reference = ""

    def text(self, value: str) -> None:
        """Retain only a bounded range reference while streaming XML."""
        if self.reference is not None:
            if len(self.reference) + len(value) > 1024:
                raise DocumentError(413)
            self.reference += value

    def end(self, name: str) -> None:
        """Bound native chart range expansion, including repeated ranges."""
        if self.reference is not None and name.endswith("}f"):
            r1, c1, r2, c2 = _range_bounds(self.reference.rsplit("!", 1)[-1])
            self.positions += (r2 - r1 + 1) * (c2 - c1 + 1)
            self.reference = None
            if self.positions > MAX_CELLS:
                raise DocumentError(413)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DocumentError
    return cast(dict[str, object], value)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise DocumentError
    return cast(list[object], value)


def project_document(
    value: object, upload: Upload, page_limit: int, *, allow_empty_image: bool = False
) -> tuple[str, int]:
    """Project a complete supported tree, never source metadata or raw copies."""
    doc = _object(value)
    if doc.get("schema_name") != "DoclingDocument" or doc.get("version") != "1.10.0":
        raise DocumentError
    allowed = {
        "schema_name",
        "version",
        "name",
        "origin",
        "body",
        "furniture",
        "pages",
        "texts",
        "groups",
        "tables",
        "pictures",
    }
    if any(value for key, value in doc.items() if key not in allowed):
        raise DocumentError
    pages = _page_count(doc.get("pages"), upload.pages, page_limit)
    nodes: dict[str, dict[str, object]] = {}
    for kind in ("texts", "groups", "tables", "pictures"):
        for index, item in enumerate(_list(doc.get(kind))):
            nodes[f"#/{kind}/{index}"] = _object(item)
    for root in ("body", "furniture"):
        nodes[f"#/{root}"] = _object(doc.get(root))
    if len(nodes) > MAX_NODES:
        raise DocumentError(413)
    projection = _Projection(nodes)
    output = "\n".join(projection.walk({"$ref": f"#/{root}"}, 0) for root in ("body", "furniture"))
    empty_image = allow_empty_image and upload.format == "img" and not projection.has_text
    if projection.seen != set(nodes) or (not output.strip() and not empty_image):
        raise DocumentError
    output = "" if empty_image else f"Document: {upload.filename}\n{output}"
    if len(output) > MAX_TEXT:
        raise DocumentError(413)
    return output, pages


def _page_count(value: object, expected: int, limit: int) -> int:
    pages = _object(value)
    if len(pages) > limit:
        raise DocumentError(413)
    if expected and len(pages) != expected:
        raise DocumentError
    for number, page in pages.items():
        if (
            not re.fullmatch(r"[1-9][0-9]{0,3}", number)
            or type(_object(page).get("page_no")) is not int
            or _object(page).get("page_no") != int(number)
        ):
            raise DocumentError
    if expected and set(pages) != {str(number) for number in range(1, expected + 1)}:
        raise DocumentError
    return len(pages)


class _Projection:
    def __init__(self, nodes: dict[str, dict[str, object]]) -> None:
        self.nodes = nodes
        self.seen: set[str] = set()
        self.active: set[str] = set()
        self.text_size = self.edges = self.cells = 0
        self.has_text = False

    def text(self, value: object) -> str:
        """Count only canonical text, not metadata or original copies."""
        if not isinstance(value, str) or "\x00" in value:
            raise DocumentError
        self.text_size += len(value) + 1
        self.has_text |= bool(value.strip())
        if self.text_size > MAX_TEXT:
            raise DocumentError(413)
        return value

    def text_item(self, node: dict[str, object]) -> str:
        """Preserve checkbox state and reject formulas without canonical text."""
        value = node.get("text")
        if not isinstance(value, str) or (node.get("label") == "formula" and not value.strip()):
            raise DocumentError
        marker = {"checkbox_selected": "[x]", "checkbox_unselected": "[ ]"}.get(
            node.get("label") if isinstance(node.get("label"), str) else ""
        )
        if marker is not None:
            value = f"{marker} {value}" if value else marker
        return self.text(value)

    def table(self, value: object, depth: int) -> str:
        """Render each canonical cell once in row order, including rich cell refs."""
        data = _object(value)
        rows, cols = data.get("num_rows"), data.get("num_cols")
        if type(rows) is not int or type(cols) is not int or rows < 1 or cols < 1:
            raise DocumentError
        self.cells += rows * cols
        if self.cells > MAX_CELLS:
            raise DocumentError(413)
        grid = [[""] * cols for _ in range(rows)]
        occupied: set[tuple[int, int]] = set()
        entries = _list(data.get("table_cells"))
        if not entries or len(entries) > rows * cols:
            raise DocumentError
        for raw in entries:
            cell = _object(raw)
            sr, er, sc, ec = _cell_offsets(cell, rows, cols)
            for row in range(sr, er):
                for col in range(sc, ec):
                    if (row, col) in occupied:
                        raise DocumentError
                    occupied.add((row, col))
            grid[sr][sc] = (
                self.walk(cell["ref"], depth + 1) if "ref" in cell else self.text(cell.get("text"))
            )
        return "\n".join(" | ".join(row) for row in grid)

    def walk(self, value: object, depth: int) -> str:
        """Walk reading-order refs, rejecting cycles and unresolved content."""
        ref = _object(value)
        pointer = ref.get("$ref")
        if set(ref) != {"$ref"} or not isinstance(pointer, str) or pointer not in self.nodes:
            raise DocumentError
        if pointer in self.active:
            raise DocumentError
        self.edges += 1
        if depth > 32 or self.edges > MAX_CELLS:
            raise DocumentError(413)
        if pointer in self.seen:
            return ""
        node = self.nodes[pointer]
        if node.get("self_ref") != pointer:
            raise DocumentError
        self.active.add(pointer)
        output: list[str] = []
        if pointer.startswith("#/texts/"):
            output.append(self.text_item(node))
        if pointer.startswith("#/tables/"):
            output.append(self.table(node.get("data"), depth))
        for field in ("captions", "footnotes", "references", "children"):
            output.extend(self.walk(ref, depth + 1) for ref in _list(node.get(field, [])))
        self.active.remove(pointer)
        self.seen.add(pointer)
        return "\n".join(part for part in output if part)


def _cell_offsets(cell: dict[str, object], rows: int, cols: int) -> tuple[int, int, int, int]:
    offsets = [
        cell.get(f"{edge}_{axis}_offset_idx")
        for axis in ("row", "col")
        for edge in ("start", "end")
    ]
    if any(type(offset) is not int for offset in offsets):
        raise DocumentError
    sr, er, sc, ec = cast(list[int], offsets)
    if not (0 <= sr < er <= rows and 0 <= sc < ec <= cols):
        raise DocumentError
    return sr, er, sc, ec
