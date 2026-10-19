"""Inspect PDF pages or extract native text in a resource-limited Linux process."""

from __future__ import annotations

import io
import logging
import os
import resource
import sys

MAX_TEXT = 4_000_000


def _native_text(reader: object, page_limit: int) -> bytes:
    """Read native text only; never render pages, decode pictures, or run OCR."""
    from pypdf import PdfReader

    if not isinstance(reader, PdfReader):
        raise TypeError
    if len(reader.pages) > page_limit:
        raise MemoryError
    chunks: list[str] = []
    size = 0
    has_text = False
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text(extraction_mode="plain")
        if "\x00" in text:
            raise ValueError
        has_text |= bool(text.strip())
        chunk = f"[Page {number}]\n{text}\n"
        size += len(chunk)
        if size > MAX_TEXT:
            raise MemoryError
        chunks.append(chunk)
    # Page markers must not disguise a successful but textless extraction.
    return "".join(chunks).encode("utf-8") if has_text else b""


def main() -> int:
    """Read bounded stdin and emit only the requested result, never parser diagnostics."""
    output = os.dup(1)
    os.dup2(2, 1)  # The parent connects stderr to DEVNULL; silence library stdout too.
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        native = len(sys.argv) == 3 and sys.argv[1] == "text"
        memory = (512 if native else 256) * 1_048_576
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        cpu = 20 if native else 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        logging.disable(logging.CRITICAL)
        # Apply limits before importing or running the third-party parser.
        from pypdf import PdfReader

        data = sys.stdin.buffer.read(40 * 1_048_576 + 1)
        if len(data) > 40 * 1_048_576:
            return 2
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted:
            return 1
        pages = len(reader.pages)
        if not pages:
            return 1
        result = _native_text(reader, int(sys.argv[2])) if native else str(pages).encode("ascii")
        # A pipe write can be short; retain the worker until all bounded output is sent.
        with os.fdopen(os.dup(output), "wb") as stream:
            stream.write(result)
    except Exception as exc:  # noqa: BLE001
        # pypdf can wrap MemoryError in a PDF error while reading compressed xrefs.
        cause: BaseException | None = exc
        for _ in range(8):
            if isinstance(cause, MemoryError):
                return 2
            if cause is None:
                break
            cause = cause.__cause__ or cause.__context__
        return 1
    finally:
        os.close(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
