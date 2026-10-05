"""Optional private rejection files: one JSON metadata line, then original bytes."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class RejectionCaptureConfig:
    """Bound an exclusive capture directory, with periodic retention cleanup."""

    directory: Path
    enabled: bool = False
    max_file_bytes: int = 6 * 1_048_576
    max_total_bytes: int = 30 * 1_048_576
    max_files: int = 20
    retention_seconds: int = 3600


class RejectionCapture:
    """Best-effort diagnostics, called only after a rejection has been decided."""

    def __init__(self, config: RejectionCaptureConfig) -> None:
        """Keep configuration without creating directories or worker tasks."""
        self.config = config

    async def cleanup(self) -> bool:
        """Remove expired/excess captures off-thread; return false on failure."""
        if not self.config.enabled:
            return True
        try:
            await asyncio.to_thread(self._cleanup)
        except Exception:  # noqa: BLE001 -- cleanup is best effort, without sensitive logs
            return False
        return True

    async def run_cleanup(self) -> None:
        """Run in a lifespan task; cancel and await the task during shutdown.

        Clean immediately, then at most every 60 seconds, even without requests.
        Files may exceed retention by one interval plus cleanup time.
        """
        if not self.config.enabled:
            return
        interval = min(60.0, max(1.0, self.config.retention_seconds / 2))
        while True:
            await self.cleanup()
            await asyncio.sleep(interval)

    def _cleanup(self) -> None:
        try:
            directory_fd = os.open(
                self.config.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
        except FileNotFoundError:
            return
        try:
            os.fchmod(directory_fd, 0o700)
            fcntl.flock(directory_fd, fcntl.LOCK_EX)
            self._prune(directory_fd, incoming_bytes=0, incoming_count=0)
        finally:
            os.close(directory_fd)

    async def capture_rejection(
        self,
        body: bytes,
        *,
        correlation_id: str,
        reason_code: str,
        complete: bool,
        declared_bytes: int | None = None,
        observed_bytes: int | None = None,
    ) -> str | None:
        """Store original bytes, never headers; failure must not change the rejection.

        ``complete`` says whether ``body`` contains the whole original request.
        Declared and observed lengths are independent optional caller facts.
        The returned opaque filename is safe to use as a diagnostic reference.
        """
        if not self.config.enabled:
            return None
        try:
            return await asyncio.to_thread(
                self._capture,
                body,
                {
                    "correlation_id": correlation_id,
                    "reason_code": reason_code,
                    "input_complete": complete,
                    "declared_bytes": declared_bytes,
                    "observed_bytes": observed_bytes,
                    "available_bytes": len(body),
                },
            )
        except Exception:  # noqa: BLE001 -- diagnostics must never replace a rejection
            return None

    def _capture(self, body: bytes, metadata: dict[str, object]) -> str | None:
        config = self.config
        if (
            min(
                config.max_file_bytes,
                config.max_total_bytes,
                config.max_files,
                config.retention_seconds,
            )
            <= 0
        ):
            return None
        # Reserve a bounded metadata line, including the completeness fields.
        budget = min(config.max_file_bytes, config.max_total_bytes)
        stored = min(len(body), max(0, budget - 1024))
        metadata.update(
            captured_bytes=stored,
            capture_truncated=stored < len(body),
            complete=bool(metadata["input_complete"]) and stored == len(body),
            created_at_unix=time.time(),
        )
        prefix = json.dumps(metadata, separators=(",", ":")).encode() + b"\n"
        if len(prefix) > min(1024, budget):
            return None
        config.directory.mkdir(mode=0o700, exist_ok=True)
        directory_fd = os.open(config.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fchmod(directory_fd, 0o700)
            # Lock the directory itself: no extra files, and safe across workers.
            fcntl.flock(directory_fd, fcntl.LOCK_EX)
            self._prune(directory_fd, len(prefix) + stored)
            return self._write(directory_fd, prefix, body[:stored])
        finally:
            os.close(directory_fd)

    def _prune(self, directory_fd: int, incoming_bytes: int, incoming_count: int = 1) -> None:
        files = []
        cutoff = time.time() - self.config.retention_seconds
        for name in os.listdir(directory_fd):
            if not re.fullmatch(r"rejection-[0-9a-f]{32}\.capture", name):
                continue
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                continue
            if info.st_mtime < cutoff:
                os.unlink(name, dir_fd=directory_fd)
            else:
                files.append((info.st_mtime, name, info.st_size))
        files.sort()
        total = sum(size for _, _, size in files)
        count = len(files)
        for _, name, size in files:
            if (
                count + incoming_count <= self.config.max_files
                and total + incoming_bytes <= self.config.max_total_bytes
            ):
                break
            os.unlink(name, dir_fd=directory_fd)
            total -= size
            count -= 1

    @staticmethod
    def _write(directory_fd: int, prefix: bytes, body: bytes) -> str:
        reference = f"rejection-{uuid4().hex}.capture"
        fd = os.open(
            reference,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(prefix)
                stream.write(body)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(reference, dir_fd=directory_fd)
            raise
        return reference
