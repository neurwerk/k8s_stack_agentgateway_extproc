import asyncio
import json
import threading

import httpx
import pytest
from pydantic import ValidationError

from agentgateway_extproc.config.settings import RejectionLangfuseSettings, Settings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.lib.rejection_langfuse import RejectionLangfuse
from agentgateway_extproc.metrics import rejection_exports_total

from .conftest import body_request, header_request


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


@pytest.mark.parametrize("enabled", [False, True])
async def test_grpc_rejection_unchanged_and_export_failure_isolated(engine_client, enabled, caplog):
    exported = []

    def transport(request):
        if request.method == "GET":
            return project_response()
        exported.append(request)
        return httpx.Response(503, text="sensitive remote content")

    settings = Settings(max_request_bytes=1024, rejection_langfuse=config(enabled=enabled))
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        exporter = RejectionLangfuse(settings.rejection_langfuse, client)
        exporter.start()
        servicer = ExtProcServicer(engine_client, settings, rejection_langfuse=exporter)
        original = json.dumps({"messages": [{"content": "private prose" * 100}]}).encode()

        async def requests():
            yield header_request()
            yield body_request(original)

        responses = [response async for response in servicer.Process(requests(), object())]
        error = json.loads(responses[-1].immediate_response.body)["error"]
        assert responses[-1].immediate_response.status.code == 413
        assert error["code"] == "request_too_large"
        assert error["limit"]["measured"] == len(original)
        await exporter.close()
    assert len(exported) == int(enabled)
    if enabled:
        _, attrs = attributes(exported[0])
        metadata = json.loads(attrs["langfuse.observation.metadata"])
        assert metadata["request_id"] == error["request_id"]
        assert json.loads(attrs["langfuse.observation.input"]) == json.loads(original)
    assert "private prose" not in caplog.text
    assert "sensitive remote content" not in caplog.text


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


@pytest.mark.parametrize("status", [400, 403, 503, 200])
async def test_request_status_scope_and_response_content_excluded(
    engine_client, monkeypatch, status
):
    from agentgateway_extproc.controllers import grpc_servicer
    from agentgateway_extproc.gen import ext_proc_pb2
    from agentgateway_extproc.lib.pipeline.request import immediate_response

    from .conftest import response_body

    async def handle(handler, request, kind):
        if kind == "request_body":
            if status != 200:
                return (immediate_response(status, '{"error":"fixed error"}'),)
            return (ext_proc_pb2.ProcessingResponse(request_body={}),)
        if kind == "response_body":
            return (immediate_response(400, '{"error":"provider failure"}'),)
        return (ext_proc_pb2.ProcessingResponse(request_headers={}),)

    monkeypatch.setattr(grpc_servicer, "_handle_message", handle)
    exported = []

    def transport(request):
        if request.method == "GET":
            return project_response()
        exported.append(request)
        return httpx.Response(200, json={})

    settings = Settings(rejection_langfuse=config())
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        exporter = RejectionLangfuse(settings.rejection_langfuse, client)
        exporter.start()
        servicer = ExtProcServicer(engine_client, settings, rejection_langfuse=exporter)

        async def requests():
            yield header_request()
            yield body_request(b'{"messages":[{"content":"private request"}]}')
            yield response_body(b"private provider response")

        responses = [response async for response in servicer.Process(requests(), object())]
        assert responses[-1].immediate_response.status.code == (400 if status == 200 else status)
        await exporter.close()
    assert len(exported) == int(status != 200)
    assert all(b"private provider response" not in request.content for request in exported)


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


async def test_request_capacity_rejection_exports_without_changing_retry(engine_client):
    exported = []

    def transport(request):
        if request.method == "GET":
            return project_response()
        exported.append(request)
        return httpx.Response(200, json={})

    settings = Settings(rejection_langfuse=config())
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        exporter = RejectionLangfuse(settings.rejection_langfuse, client)
        exporter.start()
        servicer = ExtProcServicer(engine_client, settings, rejection_langfuse=exporter)
        servicer._buffered_bytes = 4 * settings.max_request_bytes

        async def requests():
            yield header_request()
            yield body_request(b'{"content":"private request"}')

        responses = [response async for response in servicer.Process(requests(), object())]
        immediate = responses[-1].immediate_response
        error = json.loads(immediate.body)["error"]
        assert immediate.status.code == 503
        assert error["code"] == "capacity_unavailable" and error["retryable"] is True
        headers = [(item.header.key, item.header.value) for item in immediate.headers.set_headers]
        assert headers.count(("retry-after", "1")) == 1
        assert headers.count(("x-request-id", error["request_id"])) == 1
        assert servicer._buffered_bytes == 4 * settings.max_request_bytes
        await exporter.close()
    assert len(exported) == 1
    _, attrs = attributes(exported[0])
    metadata = json.loads(attrs["langfuse.observation.metadata"])
    assert metadata["request_id"] == error["request_id"]
    assert metadata["reason"] == "capacity_unavailable"
    assert "private request" in attrs["langfuse.observation.input"]
