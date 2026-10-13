"""Queue expiry of isolated debug traces using the Langfuse v4.28.1 public API.

Run hourly. Each run snapshots at most 10,000 observations from the oldest
expired 24-hour window before deletion. Partial scans and records older than
the fixed 24-hour deletion grace are unhealthy even when deletion is queued.
"""

import asyncio
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EXPECTED_NAME = "rejected-request-debug"
RETENTION = timedelta(hours=72)
DELETION_GRACE = timedelta(hours=24)
OLDEST_WINDOW = timedelta(hours=24)
MAX_PROBES = 32
PAGE_SIZE = 100
MAX_PAGES = 100
DELETE_BATCH_SIZE = 1000
RUN_TIMEOUT_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 30


class RetentionError(Exception):
    """Fail closed without including remote or configuration content."""


class RetentionSettings(BaseSettings):
    """Read dedicated credentials and mandatory identity pins with safe defaults."""

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_DEBUG_", extra="ignore")
    base_url: str
    public_key: SecretStr
    secret_key: SecretStr
    expected_project_id: str = EXPECTED_NAME
    expected_org_id: str = EXPECTED_NAME

    @field_validator("base_url")
    @classmethod
    def validate_origin(cls, value: str) -> str:
        """Require an explicit HTTP origin without embedded credentials."""
        url = httpx.URL(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.host
            or url.userinfo
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise RetentionError
        return str(url).rstrip("/")

    @field_validator("public_key", "secret_key")
    @classmethod
    def validate_key(cls, value: SecretStr) -> SecretStr:
        """Reject empty credentials before making a request."""
        if not value.get_secret_value().strip():
            raise RetentionError
        return value

    @field_validator("expected_project_id", "expected_org_id")
    @classmethod
    def validate_pin(cls, value: str) -> str:
        """An explicitly supplied pin must be nonempty and exact."""
        if not value or value != value.strip():
            raise RetentionError
        return value


class Identity(BaseModel):
    """Minimum authenticated project or organization identity."""

    model_config = ConfigDict(strict=True)
    id: str = Field(min_length=1)
    name: str


class Project(Identity):
    """Project-scoped identity returned by the public projects endpoint."""

    organization: Identity


class Projects(BaseModel):
    """Require exactly one authenticated project."""

    data: list[Project] = Field(min_length=1, max_length=1)


class Observation(BaseModel):
    """Read only identity and time; never request message fields."""

    trace_id: str = Field(alias="traceId", min_length=1)
    project_id: str = Field(alias="projectId", min_length=1)
    start_time: datetime = Field(alias="startTime")

    @field_validator("start_time", mode="before")
    @classmethod
    def validate_timestamp_string(cls, value: object) -> str:
        """Require a timezone-bearing RFC3339 string before datetime coercion."""
        if not isinstance(value, str) or not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}"
            r"(?:\.[0-9]+)?(?:[Zz]|[+-][0-9]{2}:[0-9]{2})",
            value,
        ):
            raise RetentionError
        return value


class PageMeta(BaseModel):
    """Opaque cursor; absent or null marks the end."""

    cursor: str | None = Field(default=None, min_length=1)


class ObservationPage(BaseModel):
    """Bound the API page independently of the query limit."""

    data: list[Observation] = Field(max_length=PAGE_SIZE)
    meta: PageMeta


class DeletionAcknowledgement(BaseModel):
    """Require the public API acknowledgement rather than an arbitrary HTTP 200."""

    message: str


@dataclass
class Counts:
    """Safe progress retained even if a later batch fails."""

    pages: int = 0
    probes: int = 0
    observations: int = 0
    queued: int = 0
    overdue: int = 0
    backlog: bool = False


async def verify_project(client: httpx.AsyncClient, settings: RetentionSettings) -> str:
    """Authorize deletion only in the exact dedicated project and organization."""
    response = await client.get("/api/public/projects")
    response.raise_for_status()
    project = Projects.model_validate(response.json()).data[0]
    if (
        project.name != EXPECTED_NAME
        or project.organization.name != EXPECTED_NAME
        or project.id != settings.expected_project_id
        or project.organization.id != settings.expected_org_id
    ):
        raise RetentionError
    return project.id


def validate_observation(observation: Observation, project_id: str, cutoff: datetime) -> None:
    """Reject unexpected identity or age before considering trace-wide deletion."""
    if (
        observation.project_id != project_id
        or observation.start_time.tzinfo is None
        or observation.start_time >= cutoff
    ):
        raise RetentionError


async def has_expired(
    client: httpx.AsyncClient,
    project_id: str,
    cutoff: datetime,
    counts: Counts,
    *,
    starts_at: datetime | None = None,
) -> bool:
    """Probe only time and identity to locate the oldest data and remaining backlog."""
    if counts.probes >= MAX_PROBES:
        raise RetentionError
    params = {"toStartTime": cutoff.isoformat(), "fields": "time", "limit": "1"}
    if starts_at is not None:
        params["fromStartTime"] = starts_at.isoformat()
    response = await client.get("/api/public/v2/observations", params=params)
    response.raise_for_status()
    counts.probes += 1
    page = ObservationPage.model_validate(response.json())
    if len(page.data) > 1 or (not page.data and page.meta.cursor is not None):
        raise RetentionError
    for observation in page.data:
        validate_observation(observation, project_id, cutoff)
        if starts_at is not None and observation.start_time < starts_at:
            raise RetentionError
    return bool(page.data)


async def oldest_window(
    client: httpx.AsyncClient, project_id: str, cutoff: datetime, counts: Counts
) -> datetime | None:
    """Bisect time to an oldest occupied window, since v4.28.1 has no sort option.

    Probe timestamps stay in the nonnegative Unix epoch used by the exporter
    and ClickHouse. The final scan has no lower bound, so even unexpectedly
    earlier records remain eligible. Following runs advance once asynchronous
    deletion removes the oldest data. Any remaining backlog is unhealthy.
    """
    if not await has_expired(client, project_id, cutoff, counts):
        return None
    lower = datetime(1970, 1, 1, tzinfo=UTC)
    upper = cutoff
    while upper - lower > OLDEST_WINDOW:
        midpoint = (lower + (upper - lower) / 2).replace(hour=0, minute=0, second=0, microsecond=0)
        midpoint = max(lower + OLDEST_WINDOW, midpoint)
        if await has_expired(client, project_id, midpoint, counts):
            upper = midpoint
        else:
            lower = midpoint
    counts.backlog = await has_expired(client, project_id, cutoff, counts, starts_at=upper)
    return upper


async def collect_expired(
    client: httpx.AsyncClient,
    project_id: str,
    cutoff: datetime,
    counts: Counts,
    *,
    overdue_before: datetime,
) -> list[str]:
    """Snapshot bounded complete pages before mutation; never restart a cursor."""
    params = {"toStartTime": cutoff.isoformat(), "fields": "time", "limit": str(PAGE_SIZE)}
    cursors: set[str] = set()
    traces: dict[str, None] = {}
    for _ in range(MAX_PAGES):
        response = await client.get("/api/public/v2/observations", params=params)
        response.raise_for_status()
        page = ObservationPage.model_validate(response.json())
        counts.pages += 1
        for observation in page.data:
            validate_observation(observation, project_id, cutoff)
            if observation.start_time < overdue_before:
                counts.overdue += 1
            traces[observation.trace_id] = None
        counts.observations += len(page.data)
        cursor = page.meta.cursor
        if cursor is None:
            return list(traces)
        if not page.data or cursor in cursors:
            raise RetentionError
        cursors.add(cursor)
        params["cursor"] = cursor
    counts.backlog = True
    return list(traces)


async def expire(
    client: httpx.AsyncClient,
    settings: RetentionSettings,
    counts: Counts,
    *,
    now: datetime | None = None,
) -> None:
    """Verify identity, snapshot expired observations, then queue trace deletion.

    The dedicated exporter must give each request its own trace: deletion is
    trace-wide, not observation-wide. No retries or deletion-completion polling
    occur inside this run; the next hourly run can retry remaining records.
    """
    async with asyncio.timeout(RUN_TIMEOUT_SECONDS):
        cutoff = (now or datetime.now(UTC)) - RETENTION
        project_id = await verify_project(client, settings)
        scan_cutoff = await oldest_window(client, project_id, cutoff, counts)
        if scan_cutoff is None:
            return
        traces = await collect_expired(
            client, project_id, scan_cutoff, counts, overdue_before=cutoff - DELETION_GRACE
        )
        for offset in range(0, len(traces), DELETE_BATCH_SIZE):
            batch = traces[offset : offset + DELETE_BATCH_SIZE]
            response = await client.request(
                "DELETE", "/api/public/traces", json={"traceIds": batch}
            )
            response.raise_for_status()
            DeletionAcknowledgement.model_validate(response.json())
            counts.queued += len(batch)


async def run(counts: Counts) -> None:
    """Construct a bounded client without proxy inheritance or redirects."""
    settings = RetentionSettings()
    async with httpx.AsyncClient(
        base_url=settings.base_url,
        auth=(settings.public_key.get_secret_value(), settings.secret_key.get_secret_value()),
        timeout=REQUEST_TIMEOUT_SECONDS,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        await expire(client, settings, counts)


def main() -> int:
    """Exit nonzero on failure and emit only safe progress counts."""
    counts = Counts()
    failed = False
    try:
        asyncio.run(run(counts))
        failed = counts.backlog or counts.overdue > 0
    except Exception:  # noqa: BLE001 -- CLI must not leak transport/configuration exceptions.
        failed = True
    sys.stderr.write(
        f"debug_retention failed={int(failed)} pages={counts.pages} probes={counts.probes} "
        f"observations={counts.observations} deletion_queued={counts.queued} "
        f"overdue={counts.overdue} backlog={int(counts.backlog)}\n"
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
