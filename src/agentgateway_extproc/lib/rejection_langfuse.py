"""Bounded, best-effort OTLP/HTTP JSON export to an operator-selected debug project."""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, NoReturn
from uuid import uuid4

import httpx

from agentgateway_extproc.config.settings import RejectionLangfuseSettings
from agentgateway_extproc.metrics import rejection_exports_total
from agentgateway_extproc.models.exceptions import LimitDetail

_NAME = "extproc.rejected-request"
_CREDENTIAL_FIELDS = frozenset(
    {
        "authorization",
        "proxyauthorization",
        "apikey",
        "xapikey",
        "apitoken",
        "secret",
        "secretkey",
        "privatekey",
        "bearer",
        "publickey",
        "password",
        "passwd",
        "token",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "clientsecret",
        "credentials",
        "cookie",
        "setcookie",
        "headers",
        "httpheaders",
    }
)


@dataclass(frozen=True)
class RejectionEvent:
    """Retain bounded original bytes and content-free rejection facts."""

    body: bytes
    request_id: str
    status: int
    observed_bytes: int
    complete: bool
    limit: LimitDetail | None
    timestamp: int
    reason: Literal["capacity_unavailable"] | None = None


def _redact(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if "".join(c for c in key.lower() if c.isalnum()) in _CREDENTIAL_FIELDS
            else _redact(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact(child) for child in value]
    return value


def _invalid_constant(value: str) -> NoReturn:
    raise ValueError("nonfinite JSON constant")  # noqa: TRY003


def _input(event: RejectionEvent, maximum: int) -> tuple[str, str]:
    # A truncated/malformed document cannot establish structured credential boundaries.
    if not event.complete or len(event.body) != event.observed_bytes:
        return "", "incomplete_or_over_budget"
    try:
        parsed = json.loads(event.body, parse_constant=_invalid_constant)
        if not isinstance(parsed, (dict, list)):
            return "", "invalid_json"
        sanitized = json.dumps(
            _redact(parsed), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        encoded = sanitized.encode("utf-8")
        return encoded[:maximum].decode("utf-8", errors="ignore"), (
            "redacted_truncated" if len(encoded) > maximum else "redacted"
        )
    except (ValueError, RecursionError, UnicodeError):
        return "", "invalid_json"


def _payload(event: RejectionEvent, maximum: int) -> dict[str, object]:
    content, state = _input(event, maximum)
    metadata: dict[str, object] = {
        "request_id": event.request_id,
        "status": event.status,
        "reason": event.reason
        or (
            event.limit.reason
            if event.limit
            else {
                400: "invalid_request",
                413: "request_too_large",
                403: "policy_blocked",
                503: "processing_failed",
            }[event.status]
        ),
        "observed_bytes": event.observed_bytes,
        "captured_bytes": len(content.encode("utf-8")),
        "input_complete": event.complete,
        "capture_state": state,
    }
    if event.limit is not None:
        metadata["limit"] = event.limit.model_dump()
    attributes = {
        "langfuse.trace.name": _NAME,
        "langfuse.observation.type": "span",
        "langfuse.observation.level": "ERROR",
        "langfuse.observation.input": content,
        "langfuse.observation.metadata": json.dumps(metadata, separators=(",", ":")),
    }
    span = {
        "traceId": uuid4().hex,
        "spanId": uuid4().hex[:16],
        "name": _NAME,
        "kind": 1,
        "startTimeUnixNano": str(event.timestamp),
        "endTimeUnixNano": str(event.timestamp + 1),
        "status": {"code": 2},
        "attributes": [
            {"key": key, "value": {"stringValue": value}} for key, value in attributes.items()
        ],
    }
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "agentgateway-extproc"}}
                    ]
                },
                "scopeSpans": [{"scope": {"name": _NAME}, "spans": [span]}],
            }
        ]
    }


def _encode_payload(event: RejectionEvent, maximum: int) -> bytes:
    payload = json.dumps(
        _payload(event, maximum), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if len(payload) > 6 * maximum + 16384:
        raise ValueError("oversized OTLP envelope")  # noqa: TRY003
    return payload


class RejectionLangfuse:
    """One bounded queue and worker; requests never await network delivery."""

    def __init__(
        self, settings: RejectionLangfuseSettings, client: httpx.AsyncClient | None = None
    ) -> None:
        """Keep disabled construction free of network clients and worker tasks."""
        self.settings = settings
        self._client = client
        self._owns_client = client is None
        self._queue: asyncio.Queue[RejectionEvent] = asyncio.Queue(settings.queue_capacity)
        self._task: asyncio.Task[None] | None = None
        self._accepting = False
        self._verified = False
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    def start(self) -> None:
        """Start once on the serving loop, only when explicitly enabled."""
        if not self.settings.enabled or self._task is not None or self._closed:
            return
        if not (
            self.settings.endpoint
            and self.settings.public_key.get_secret_value().strip()
            and self.settings.secret_key.get_secret_value().strip()
        ):
            rejection_exports_total.labels(outcome="unconfigured").inc()
            return
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.settings.timeout_seconds, follow_redirects=False, trust_env=False
            )
        self._accepting = True
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rejection-debug")
        self._task = asyncio.create_task(self._run())

    def submit(
        self,
        body: bytes,
        *,
        request_id: str,
        status: int,
        observed_bytes: int,
        complete: bool,
        limit: LimitDetail | None = None,
        reason: Literal["capacity_unavailable"] | None = None,
    ) -> None:
        """Enqueue without waiting; drop on saturation without changing the rejection."""
        if not self.settings.enabled:
            return
        if not self._accepting or self._queue.full():
            rejection_exports_total.labels(outcome="dropped").inc()
            return
        bounded = body[: self.settings.max_capture_bytes]
        self._queue.put_nowait(
            RejectionEvent(
                bounded,
                request_id,
                status,
                observed_bytes,
                complete and len(bounded) == observed_bytes,
                limit,
                time.time_ns(),
                reason,
            )
        )

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._export(event)
            except asyncio.CancelledError:
                rejection_exports_total.labels(outcome="dropped").inc()
                raise
            except Exception:  # noqa: BLE001 -- telemetry cannot expose exceptions or stop worker
                rejection_exports_total.labels(outcome="failed").inc()
            else:
                rejection_exports_total.labels(outcome="exported").inc()
            finally:
                self._queue.task_done()
                del event

    async def _export(self, event: RejectionEvent) -> None:
        if self._client is None or self._executor is None:
            return
        # The sole consumer awaits one preparation before submitting another. A network
        # timeout never abandons preparation and queues another executor job. Shutdown
        # may leave at most this one bounded computation finishing, without network I/O.
        payload = await asyncio.get_running_loop().run_in_executor(
            self._executor, _encode_payload, event, self.settings.max_capture_bytes
        )
        async with asyncio.timeout(self.settings.timeout_seconds):
            if not self._verified:
                await self._verify_project()
            async with self._client.stream(
                "POST",
                self.settings.endpoint,
                content=payload,
                auth=httpx.BasicAuth(
                    self.settings.public_key.get_secret_value(),
                    self.settings.secret_key.get_secret_value(),
                ),
                headers={"x-langfuse-ingestion-version": "4", "Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                # Never retain or log upstream error bodies. Bound successful OTLP replies too.
                result = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(result) + len(chunk) > 4096:
                        raise ValueError("oversized OTLP reply")  # noqa: TRY003
                    result.extend(chunk)
                reply = json.loads(result) if result else {}
                if not isinstance(reply, dict) or reply.get("partialSuccess"):
                    raise ValueError("partial OTLP delivery")  # noqa: TRY003

    async def _verify_project(self) -> None:
        if self._client is None:
            return
        endpoint = self.settings.endpoint.removesuffix("/otel/v1/traces") + "/projects"
        async with self._client.stream(
            "GET",
            endpoint,
            auth=httpx.BasicAuth(
                self.settings.public_key.get_secret_value(),
                self.settings.secret_key.get_secret_value(),
            ),
        ) as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                if len(data) + len(chunk) > 16384:
                    raise ValueError("oversized project reply")  # noqa: TRY003
                data.extend(chunk)
        payload = json.loads(data)
        projects = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(projects, list) or len(projects) != 1:
            raise ValueError("debug project identity mismatch")  # noqa: TRY003
        project = projects[0]
        organization = project.get("organization") if isinstance(project, dict) else None
        if (
            not isinstance(project, dict)
            or project.get("id") != "rejected-request-debug"
            or not isinstance(organization, dict)
            or organization.get("id") != "rejected-request-debug"
        ):
            raise ValueError("debug project identity mismatch")  # noqa: TRY003
        self._verified = True

    async def close(self) -> None:
        """Bound shutdown drain to one export timeout, then discard pending content."""
        self._accepting = False
        self._closed = True
        if self._task is not None:
            with suppress(TimeoutError):
                async with asyncio.timeout(self.settings.timeout_seconds):
                    await self._queue.join()
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
            rejection_exports_total.labels(outcome="dropped").inc()
        if self._owns_client and self._client is not None:
            await self._client.aclose()
