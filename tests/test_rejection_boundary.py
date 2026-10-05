"""Exercise admission errors and original-body captures at the gRPC boundary."""

import json

import pytest

from agentgateway_extproc.config.settings import RejectionCaptureSettings, Settings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer

from .conftest import body_request, header_request


@pytest.mark.parametrize("mode", ["disabled", "enabled", "failure"])
async def test_admission_error_is_correlated_and_capture_cannot_change_rejection(
    engine_client, tmp_path, caplog, mode
):
    directory = tmp_path / "captures"
    if mode == "failure":
        directory.write_text("not a directory")
    settings = Settings(
        max_request_bytes=1024,
        rejection_capture=RejectionCaptureSettings(
            enabled=mode != "disabled", directory=directory, max_file_bytes=4096
        ),
    )
    servicer = ExtProcServicer(engine_client, settings)
    original = b"private request value " * 70

    async def requests():
        yield header_request()
        yield body_request(original)

    responses = [response async for response in servicer.Process(requests(), object())]
    immediate = responses[-1].immediate_response
    assert immediate.status.code == 413
    error = json.loads(immediate.body)["error"]
    assert error["code"] == "request_too_large"
    assert error["limit"]["measured"] == len(original)
    assert error["limit"]["maximum"] == 1024
    assert error["limit"]["exact"] is True
    assert error["request_id"] in caplog.text
    assert "private request value" not in immediate.body + caplog.text
    if mode == "enabled":
        (capture,) = directory.glob("*.capture")
        metadata, body = capture.read_bytes().split(b"\n", 1)
        assert body == original
        assert json.loads(metadata)["complete"] is True
        assert json.loads(metadata)["correlation_id"] == error["request_id"]
    elif mode == "disabled":
        assert not directory.exists()
