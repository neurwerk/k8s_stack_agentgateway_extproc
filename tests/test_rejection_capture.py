import asyncio
import json
import os
import threading
import time

import pytest

from agentgateway_extproc.lib.rejection_capture import RejectionCapture, RejectionCaptureConfig


async def capture(helper, body=b"original\x00\xff\n", **kwargs):
    return await helper.capture_rejection(
        body, correlation_id="request-1", reason_code="body_limit", complete=True, **kwargs
    )


async def test_disabled_has_no_side_effects(tmp_path):
    directory = tmp_path / "disabled"
    assert await capture(RejectionCapture(RejectionCaptureConfig(directory))) is None
    assert not directory.exists()


async def test_original_bytes_metadata_permissions_and_worker(tmp_path, monkeypatch):
    helper = RejectionCapture(RejectionCaptureConfig(tmp_path, enabled=True))
    main_thread = threading.get_ident()
    original = helper._write

    def write(*args):
        assert threading.get_ident() != main_thread
        return original(*args)

    monkeypatch.setattr(helper, "_write", write)
    reference = await capture(helper, declared_bytes=900, observed_bytes=12)
    assert reference is not None
    path = tmp_path / reference
    metadata, body = path.read_bytes().split(b"\n", 1)
    data = json.loads(metadata)
    assert body == b"original\x00\xff\n"
    assert data["correlation_id"] == "request-1"
    assert data["declared_bytes"] == 900
    assert data["observed_bytes"] == 12
    assert data["complete"] is True
    assert data["capture_truncated"] is False
    assert "headers" not in data
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(("max_files", "max_total_bytes"), [(2, 10_000), (20, 1600)])
async def test_bounds_retention_and_concurrent_writers(tmp_path, max_files, max_total_bytes):
    config = RejectionCaptureConfig(
        tmp_path,
        enabled=True,
        max_file_bytes=1500,
        max_total_bytes=max_total_bytes,
        max_files=max_files,
    )
    expired = tmp_path / f"rejection-{'0' * 32}.capture"
    expired.write_bytes(b"expired")
    expired_at = time.time() - 7200
    os.utime(expired, (expired_at, expired_at))
    helpers = [RejectionCapture(config) for _ in range(6)]
    references = await asyncio.gather(*(capture(helper, b"x" * 3000) for helper in helpers))
    assert all(references)
    assert not expired.exists()
    files = list(tmp_path.iterdir())
    assert 0 < len(files) <= max_files
    assert sum(path.stat().st_size for path in files) <= max_total_bytes
    for path in files:
        assert path.stat().st_size <= 1500
        metadata, body = path.read_bytes().split(b"\n", 1)
        data = json.loads(metadata)
        assert data["complete"] is False
        assert data["capture_truncated"] is True
        assert data["captured_bytes"] == len(body)
        assert body == b"x" * len(body)


async def test_partial_input_and_failure_are_best_effort(tmp_path, monkeypatch, caplog):
    helper = RejectionCapture(RejectionCaptureConfig(tmp_path, enabled=True))
    reference = await helper.capture_rejection(
        b"prefix", correlation_id="request-2", reason_code="incomplete", complete=False
    )
    assert reference is not None
    metadata = json.loads((tmp_path / reference).read_bytes().split(b"\n", 1)[0])
    assert metadata["complete"] is False
    assert metadata["capture_truncated"] is False
    assert metadata["declared_bytes"] is None
    assert metadata["observed_bytes"] is None

    def fail(*args):
        raise OSError("sensitive content")

    monkeypatch.setattr(helper, "_write", fail)
    assert await capture(helper) is None
    assert not caplog.records


async def test_symlink_directory_is_not_used(tmp_path):
    target = tmp_path / "target"
    target.mkdir(mode=0o755)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    assert await capture(RejectionCapture(RejectionCaptureConfig(link, enabled=True))) is None
    assert list(target.iterdir()) == []
    assert target.stat().st_mode & 0o777 == 0o755


async def test_idle_cleanup_expires_only_captures_without_reserving_a_slot(tmp_path):
    helper = RejectionCapture(RejectionCaptureConfig(tmp_path, enabled=True, max_files=2))
    references = [await capture(helper) for _ in range(2)]
    expired = tmp_path / f"rejection-{'0' * 32}.capture"
    unrelated = tmp_path / "rejection-not-a-capture.capture"
    for path in (expired, unrelated):
        path.write_bytes(b"old")
        expired_at = time.time() - 7200
        os.utime(path, (expired_at, expired_at))

    assert await helper.cleanup() is True
    assert not expired.exists()
    assert unrelated.read_bytes() == b"old"
    assert all(reference and (tmp_path / reference).exists() for reference in references)
