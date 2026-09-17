"""Inspect PDF page count in a disposable, resource-limited Linux process."""

from __future__ import annotations

import io
import logging
import os
import resource
import sys


def main() -> int:
    """Read bounded stdin and emit only a page integer, never parser diagnostics."""
    output = os.dup(1)
    os.dup2(2, 1)  # The parent connects stderr to DEVNULL; silence library stdout too.
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1_048_576, 256 * 1_048_576))
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
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
        os.write(output, str(pages).encode("ascii"))
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
