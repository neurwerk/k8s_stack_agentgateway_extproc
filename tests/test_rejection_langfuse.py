import asyncio
import json
import threading

import httpx
import pytest
from pydantic import ValidationError

from agentgateway_extproc.config.settings import RejectionLangfuseSettings, Settings
from agentgateway_extproc.lib.rejection_langfuse import RejectionLangfuse
from agentgateway_extproc.metrics import rejection_exports_total


def config(**overrides):
    return RejectionLangfuseSettings.model_validate(
        {
            "enabled": True,
            "endpoint": "https://debug.invalid/api/public/otel/v1/traces",
            "public_key": "debug-public",
            "secret_key": "debug-secret",
        }
        | overrides
    )


def submit(exporter, body=b'{"messages":[{"content":"private prose"}]}', **kwargs):
    exporter.submit(
        body,
        request_id="correlation-1",
        status=413,
        observed_bytes=len(body),
        complete=True,
        **kwargs,
    )


def attributes(request):
    span = json.loads(request.content)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    return span, {item["key"]: item["value"]["stringValue"] for item in span["attributes"]}


def project_response():
    return httpx.Response(
        200,
        json={
            "data": [
                {"id": "rejected-request-debug", "organization": {"id": "rejected-request-debug"}}
            ]
        },
    )


async def test_disabled_and_endpoint_validation(monkeypatch):
    exporter = RejectionLangfuse(RejectionLangfuseSettings())
    exporter.start()
    submit(exporter)
    await exporter.close()
    assert exporter._client is None
    assert exporter._task is None
    missing = RejectionLangfuse(config(secret_key=""))
    missing.start()
    submit(missing)
    await missing.close()
    assert missing._client is None
    assert missing._task is None
    for endpoint in [
        "http://debug.invalid/api/public/otel/v1/traces",
        "https://user:pass@debug.invalid/api/public/otel/v1/traces",
        "https://debug.invalid/api/public/ingestion",
    ]:
        with pytest.raises(ValidationError):
            config(endpoint=endpoint)
    monkeypatch.setenv("EXTPROC_REJECTION_LANGFUSE__MAX_CAPTURE_BYTES", "2048")
    assert Settings().rejection_langfuse.max_capture_bytes == 2048


async def test_otlp_redaction_bounds_and_safe_failure(caplog):
    requests = []

    def transport(request):
        if request.method == "GET":
            return project_response()
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ConnectError("sensitive exception text")
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        exporter = RejectionLangfuse(config(max_capture_bytes=1024), client)
        exporter.start()
        submit(exporter)
        body = json.dumps(
            {
                "messages": [{"content": "private prose", "api_key": "hidden"}],
                "headers": {"Authorization": "Bearer hidden"},
                "x-neurwerk-authorization": "Bearer hidden",
                "x-goog-api-key": "hidden",
                "nested": {"clientSecret": "hidden"},
            }
        ).encode()
        submit(exporter, body)
        submit(exporter, b'{"password":"hidden"')
        submit(exporter, b"x" * 2048)
        await exporter.close()
    assert len(requests) == 4
    span, attrs = attributes(requests[1])
    assert span["status"]["code"] == 2
    assert len(span["traceId"]) == 32 and len(span["spanId"]) == 16
    assert attrs["langfuse.observation.level"] == "ERROR"
    assert "hidden" not in requests[1].content.decode()
    assert "private prose" in attrs["langfuse.observation.input"]
    assert requests[1].headers["x-langfuse-ingestion-version"] == "4"
    assert (
        requests[1].headers["authorization"]
        == httpx.BasicAuth("debug-public", "debug-secret")._auth_header
    )
    for request in requests[2:]:
        _, attrs = attributes(request)
        assert attrs["langfuse.observation.input"] == ""
    assert "private prose" not in caplog.text
    assert "sensitive exception text" not in caplog.text
    assert "hidden" not in caplog.text


async def test_queue_is_nonblocking_bounded_and_shutdown_discards():
    started = asyncio.Event()
    forever = asyncio.Event()

    async def transport(request):
        if request.method == "GET":
            return project_response()
        started.set()
        await forever.wait()
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        exporter = RejectionLangfuse(config(queue_capacity=1, timeout_seconds=0.02), client)
        exporter.start()
        submit(exporter)
        await started.wait()
        submit(exporter)
        before = rejection_exports_total.labels(outcome="dropped")._value.get()
        submit(exporter)
        assert exporter._queue.qsize() == 1
        assert rejection_exports_total.labels(outcome="dropped")._value.get() == before + 1
        await asyncio.wait_for(exporter.close(), timeout=1)
        assert exporter._queue.empty()
        assert exporter._task is None


async def test_wrong_project_never_receives_content():
    requests = []

    def transport(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [{"id": "normal-project", "organization": {"id": "rejected-request-debug"}}]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        exporter = RejectionLangfuse(config(), client)
        exporter.start()
        submit(exporter)
        await exporter.close()
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/public/projects"
    assert not requests[0].content


async def test_redaction_expansion_is_byte_bounded_and_invalid_utf8_is_omitted():
    exported = []

    def transport(request):
        if request.method == "GET":
            return project_response()
        exported.append(request)
        return httpx.Response(200, json={})

    body = json.dumps(
        [{"token": "", "text": "é"}] * 30, ensure_ascii=False, separators=(",", ":")
    ).encode()
    assert len(body) <= 1024
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        exporter = RejectionLangfuse(config(max_capture_bytes=1024), client)
        exporter.start()
        submit(exporter, body)
        submit(exporter, b'{"content":"\xff"}')
        await exporter.close()
    _, attrs = attributes(exported[0])
    assert len(attrs["langfuse.observation.input"].encode()) <= 1024
    assert (
        json.loads(attrs["langfuse.observation.metadata"])["capture_state"] == "redacted_truncated"
    )
    _, attrs = attributes(exported[1])
    assert attrs["langfuse.observation.input"] == ""


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
async def test_nonfinite_json_is_metadata_only(constant):
    exported = []

    def transport(request):
        if request.method == "GET":
            return project_response()
        exported.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        exporter = RejectionLangfuse(config(), client)
        exporter.start()
        submit(exporter, ('{"content":"private prose","value":' + constant + "}").encode())
        await exporter.close()
    _, attrs = attributes(exported[0])
    assert attrs["langfuse.observation.input"] == ""
    assert json.loads(attrs["langfuse.observation.metadata"])["capture_state"] == "invalid_json"


async def test_preparation_leaves_loop_responsive_and_shutdown_bounds_threads(monkeypatch):
    from agentgateway_extproc.lib import rejection_langfuse

    entered = asyncio.Event()
    progressed = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()
    calls = []
    requests = []
    original = rejection_langfuse._encode_payload

    def prepare(event, maximum):
        calls.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(2)
            return original(event, maximum)
        finally:
            finished.set()

    def transport(request):
        requests.append(request)
        return project_response()

    monkeypatch.setattr(rejection_langfuse, "_encode_payload", prepare)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        exporter = RejectionLangfuse(config(timeout_seconds=0.02), client)
        exporter.start()
        try:
            submit(exporter)
            await asyncio.wait_for(entered.wait(), timeout=1)
            loop.call_soon(progressed.set)
            await asyncio.wait_for(progressed.wait(), timeout=1)
            for _ in range(10):
                submit(exporter)
            await asyncio.wait_for(exporter.close(), timeout=1)
            exporter.start()  # A closed exporter cannot create another executor.
            assert exporter._task is None and exporter._executor is None
            assert len(calls) == 1 and calls[0] != threading.get_ident()
            assert not requests
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 2)
            await exporter.close()
