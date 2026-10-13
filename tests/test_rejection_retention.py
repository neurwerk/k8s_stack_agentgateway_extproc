import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import SecretStr

from agentgateway_extproc.lib import rejection_retention as retention

NOW = datetime(2026, 10, 13, 12, tzinfo=UTC)
CUTOFF = NOW - timedelta(hours=72)


def settings(**overrides):
    return retention.RetentionSettings(
        base_url="http://langfuse-web:3000",
        public_key=SecretStr("test-public"),
        secret_key=SecretStr("test-secret"),
        expected_project_id="debug-project",
        expected_org_id="debug-org",
        **overrides,
    )


def project():
    return {
        "id": "debug-project",
        "name": "rejected-request-debug",
        "organization": {"id": "debug-org", "name": "rejected-request-debug"},
    }


def observation(trace="trace-one", **overrides):
    return {
        "id": "observation-not-trace",
        "traceId": trace,
        "projectId": "debug-project",
        "startTime": (CUTOFF - timedelta(seconds=1)).isoformat(),
        **overrides,
    }


async def execute(handler, counts=None, *, discover_oldest=False):
    counts = counts or retention.Counts()
    async with httpx.AsyncClient(
        base_url="http://langfuse-web:3000", transport=httpx.MockTransport(handler)
    ) as client:
        if discover_oldest:
            await retention.expire(client, settings(), counts, now=NOW)
        else:
            # Isolate cursor/identity/deletion cases from oldest-window discovery.
            with patch.object(retention, "oldest_window", AsyncMock(return_value=CUTOFF)):
                await retention.expire(client, settings(), counts, now=NOW)
    return counts


@pytest.mark.parametrize(
    "change",
    [
        "normal-project",
        "normal-org",
        "wrong-project-id",
        "wrong-org-id",
        "missing-org",
        "empty-projects",
        "multiple-projects",
    ],
)
async def test_identity_mismatch_never_reads_or_deletes_observations(change):
    identity = project()
    if change == "normal-project":
        identity["name"] = "normal"
    elif change == "normal-org":
        identity["organization"]["name"] = "normal"
    elif change == "wrong-project-id":
        identity["id"] = "other"
    elif change == "wrong-org-id":
        identity["organization"]["id"] = "other"
    elif change == "missing-org":
        del identity["organization"]
    data = [identity]
    if change == "empty-projects":
        data = []
    elif change == "multiple-projects":
        data.append(project())
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"data": data})

    with pytest.raises((retention.RetentionError, ValueError)):
        await execute(handler)
    assert calls == ["/api/public/projects"]


async def test_fixed_cutoff_cursor_and_deduplicated_trace_deletion_after_collection():
    calls = []

    def handler(request):
        calls.append(request.method)
        if request.url.path == "/api/public/projects":
            return httpx.Response(200, json={"data": [project()]})
        if request.method == "DELETE":
            assert calls == ["GET", "GET", "GET", "DELETE"]
            assert json.loads(request.content) == {"traceIds": ["trace-one", "trace-two"]}
            return httpx.Response(200, json={"message": "Traces deleted successfully"})
        assert request.url.path == "/api/public/v2/observations"
        assert request.url.params["toStartTime"] == CUTOFF.isoformat()
        assert request.url.params["fields"] == "time"
        assert request.url.params["limit"] == "100"
        cursor = request.url.params.get("cursor")
        if cursor is None:
            # A short page can still have a cursor; page length is not the stop signal.
            return httpx.Response(
                200, json={"data": [observation()], "meta": {"cursor": "opaque-cursor"}}
            )
        assert cursor == "opaque-cursor"
        return httpx.Response(
            200, json={"data": [observation(), observation("trace-two")], "meta": {}}
        )

    counts = await execute(handler)
    assert counts == retention.Counts(pages=2, observations=3, queued=2)


@pytest.mark.parametrize(
    "bad_observation",
    [
        observation(startTime=CUTOFF.isoformat()),
        observation(startTime=NOW.isoformat()),
        observation(startTime="2020-01-01T00:00:00"),
        observation(startTime=0),
        observation(startTime="0"),
        observation(startTime=123.5),
        observation(startTime="123.5"),
        observation(startTime=None),
        observation(startTime="2020-01-01"),
        observation(startTime="2020-01-01 00:00:00Z"),
        observation(startTime="2020-01-01T00:00:00+0000"),
        observation(startTime="2020-99-01T00:00:00Z"),
        observation(projectId="normal-project"),
        observation(traceId=None),
        observation(traceId=""),
    ],
)
async def test_invalid_or_unexpired_record_aborts_entire_snapshot(bad_observation):
    def handler(request):
        assert request.method == "GET"
        if request.url.path == "/api/public/projects":
            return httpx.Response(200, json={"data": [project()]})
        return httpx.Response(200, json={"data": [observation(), bad_observation], "meta": {}})

    with pytest.raises((retention.RetentionError, ValueError)):
        await execute(handler)


@pytest.mark.parametrize("empty", [False, True])
async def test_repeating_or_empty_page_cursor_fails_without_deleting(empty):
    calls = []

    def handler(request):
        calls.append(request.method)
        assert request.method == "GET"
        if request.url.path == "/api/public/projects":
            return httpx.Response(200, json={"data": [project()]})
        return httpx.Response(
            200,
            json={"data": [] if empty else [observation()], "meta": {"cursor": "repeated"}},
        )

    with pytest.raises(retention.RetentionError):
        await execute(handler)
    assert len(calls) <= 3


async def test_page_bound_reports_backlog_and_batches_at_most_1000(monkeypatch):
    monkeypatch.setattr(retention, "MAX_PAGES", 11)
    page = 0
    batches = []

    def handler(request):
        nonlocal page
        if request.url.path == "/api/public/projects":
            return httpx.Response(200, json={"data": [project()]})
        if request.method == "DELETE":
            assert page == 11
            batches.append(json.loads(request.content)["traceIds"])
            return httpx.Response(200, json={"message": "queued"})
        page += 1
        return httpx.Response(
            200,
            json={
                "data": [observation(f"trace-{page}-{item}") for item in range(100)],
                "meta": {"cursor": f"cursor-{page}"},
            },
        )

    counts = await execute(handler)
    assert counts.backlog
    assert counts.queued == 1100
    assert [len(batch) for batch in batches] == [1000, 100]
    assert len(set(batches[0] + batches[1])) == 1100


@pytest.mark.parametrize("status", [302, 401, 403, 429, 500])
async def test_delete_failure_is_not_counted_as_queued(status):
    def handler(request):
        if request.url.path == "/api/public/projects":
            return httpx.Response(200, json={"data": [project()]})
        if request.method == "DELETE":
            return httpx.Response(status, text="sensitive upstream response")
        return httpx.Response(200, json={"data": [observation()], "meta": {}})

    counts = retention.Counts()
    with pytest.raises(httpx.HTTPStatusError):
        await execute(handler, counts)
    assert counts.queued == 0


async def test_empty_history_queues_nothing():
    def handler(request):
        assert request.method == "GET"
        if request.url.path == "/api/public/projects":
            return httpx.Response(200, json={"data": [project()]})
        return httpx.Response(200, json={"data": [], "meta": {}})

    assert await execute(handler) == retention.Counts(pages=1)


async def test_invalid_success_response_is_not_a_deletion_acknowledgement():
    def handler(request):
        if request.url.path == "/api/public/projects":
            return httpx.Response(200, json={"data": [project()]})
        if request.method == "DELETE":
            return httpx.Response(200, text="<html>Sign in</html>")
        return httpx.Response(200, json={"data": [observation()], "meta": {}})

    counts = retention.Counts()
    with pytest.raises(ValueError):
        await execute(handler, counts)
    assert counts.queued == 0


async def test_execution_deadline(monkeypatch):
    monkeypatch.setattr(retention, "RUN_TIMEOUT_SECONDS", 0)

    async def handler(request):
        import asyncio

        await asyncio.sleep(1)
        pytest.fail("deadline should cancel the first request")

    with pytest.raises(TimeoutError):
        await execute(handler)


@pytest.mark.parametrize("failure", [True, False])
def test_cli_exit_and_output_never_include_exception_content(monkeypatch, capsys, failure):
    async def run(counts):
        counts.queued = 1
        if failure:
            raise RuntimeError("http://private.example secret-key trace-id response-body")

    monkeypatch.setattr(retention, "run", run)
    assert retention.main() == int(failure)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"debug_retention failed={int(failure)} pages=0 probes=0 observations=0 "
        "deletion_queued=1 overdue=0 backlog=0\n"
    )


@pytest.mark.parametrize("pin", ["", " debug-project", "debug-project "])
def test_explicit_empty_or_whitespace_identity_pins_fail(pin, monkeypatch):
    monkeypatch.setenv("LANGFUSE_DEBUG_EXPECTED_PROJECT_ID", pin)
    with pytest.raises(retention.RetentionError):
        retention.RetentionSettings(
            base_url="http://langfuse-web:3000",
            public_key=SecretStr("test-public"),
            secret_key=SecretStr("test-secret"),
        )


async def test_runtime_uses_only_debug_credentials_and_bounded_client(monkeypatch):
    monkeypatch.setenv("LANGFUSE_DEBUG_BASE_URL", "http://langfuse-web:3000")
    monkeypatch.setenv("LANGFUSE_DEBUG_PUBLIC_KEY", "debug-public")
    monkeypatch.setenv("LANGFUSE_DEBUG_SECRET_KEY", "debug-secret")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "normal-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "normal-secret")
    captured = {}
    real_client = httpx.AsyncClient

    def handler(request):
        assert request.headers["authorization"] == "Basic ZGVidWctcHVibGljOmRlYnVnLXNlY3JldA=="
        if request.url.path == "/api/public/projects":
            identity = project()
            identity["id"] = "rejected-request-debug"
            identity["organization"]["id"] = "rejected-request-debug"
            return httpx.Response(200, json={"data": [identity]})
        return httpx.Response(200, json={"data": [], "meta": {}})

    def client(**kwargs):
        captured.update(kwargs)
        return real_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(retention.httpx, "AsyncClient", client)
    counts = retention.Counts()
    await retention.run(counts)
    assert counts.probes == 1
    assert captured["timeout"] == 30
    assert captured["follow_redirects"] is False
    assert captured["trust_env"] is False


@pytest.mark.parametrize(
    "project_id,org_id",
    [
        ("unrelated-project", "rejected-request-debug"),
        ("rejected-request-debug", "unrelated-org"),
        ("unrelated-project", "unrelated-org"),
    ],
)
async def test_default_pins_reject_matching_names_with_unrelated_ids(
    monkeypatch, project_id, org_id
):
    monkeypatch.delenv("LANGFUSE_DEBUG_EXPECTED_PROJECT_ID", raising=False)
    monkeypatch.delenv("LANGFUSE_DEBUG_EXPECTED_ORG_ID", raising=False)
    config = retention.RetentionSettings(
        base_url="http://langfuse-web:3000",
        public_key=SecretStr("test-public"),
        secret_key=SecretStr("test-secret"),
    )
    identity = project()
    identity["id"] = project_id
    identity["organization"]["id"] = org_id
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"data": [identity]})

    async with httpx.AsyncClient(
        base_url=config.base_url, transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(retention.RetentionError):
            await retention.expire(client, config, retention.Counts(), now=NOW)
    assert calls == [("GET", "/api/public/projects")]


@pytest.mark.parametrize(
    "timestamp",
    ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00.123+01:00", "2020-01-01T00:00:00-05:30"],
)
async def test_rfc3339_timestamp_with_explicit_timezone_can_be_deleted(timestamp):
    deleted = []
    counts = await execute(
        history_handler([observation(startTime=timestamp)], deleted), discover_oldest=True
    )
    assert deleted == ["trace-one"]
    assert counts.queued == 1


def history_handler(records, deleted):
    def handler(request):
        if request.url.path == "/api/public/projects":
            return httpx.Response(200, json={"data": [project()]})
        if request.method == "DELETE":
            deleted.extend(json.loads(request.content)["traceIds"])
            # Deliberately leave records visible, like an asynchronous or skipped delete.
            return httpx.Response(200, json={"message": "Traces deleted successfully"})
        assert request.url.params["fields"] == "time"
        upper = datetime.fromisoformat(request.url.params["toStartTime"])
        lower = datetime.fromisoformat(
            request.url.params.get("fromStartTime", "0001-01-01T00:00:00+00:00")
        )
        matches = sorted(
            [
                item
                for item in records
                if lower <= datetime.fromisoformat(item["startTime"]) < upper
            ],
            key=lambda item: item["startTime"],
            reverse=True,
        )
        limit = int(request.url.params["limit"])
        return httpx.Response(
            200,
            json={
                "data": matches[:limit],
                "meta": {"cursor": "more"} if len(matches) > limit else {},
            },
        )

    return handler


async def test_oldest_window_prevents_newest_first_starvation_and_reports_partial_scan():
    records = [
        observation("oldest", startTime=(CUTOFF - timedelta(days=30)).isoformat()),
        observation("newer", startTime=(CUTOFF - timedelta(days=3)).isoformat()),
        observation("recent-expired"),
        observation("retained", startTime=NOW.isoformat()),
    ]
    deleted = []
    counts = await execute(history_handler(records, deleted), discover_oldest=True)
    assert deleted == ["oldest"]
    assert counts.backlog
    assert counts.overdue == 1
    assert 0 < counts.probes <= retention.MAX_PROBES
    # Simulate the worker completing the oldest deletion between hourly runs.
    records.pop(0)
    deleted.clear()
    counts = await execute(history_handler(records, deleted), discover_oldest=True)
    assert deleted == ["newer"]
    assert counts.backlog


@pytest.mark.parametrize("age_hours,unhealthy", [(73, False), (96, False), (97, True)])
def test_successful_noop_deletion_is_unhealthy_after_grace(
    monkeypatch, capsys, age_hours, unhealthy
):
    records = [observation("expired", startTime=(NOW - timedelta(hours=age_hours)).isoformat())]
    deleted = []

    async def run(counts):
        await execute(history_handler(records, deleted), counts, discover_oldest=True)

    monkeypatch.setattr(retention, "run", run)
    assert retention.main() == int(unhealthy)
    assert deleted == ["expired"]
    output = capsys.readouterr().err
    assert "deletion_queued=1" in output
    assert f"overdue={int(unhealthy)}" in output
    assert "expired" not in output


def test_partial_scan_is_unhealthy_even_without_overdue_data(monkeypatch, capsys):
    async def run(counts):
        counts.backlog = True
        counts.queued = 10000

    monkeypatch.setattr(retention, "run", run)
    assert retention.main() == 1
    assert "backlog=1" in capsys.readouterr().err
