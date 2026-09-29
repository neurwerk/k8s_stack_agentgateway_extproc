"""Best-effort, bounded lookup of effective notice preferences from Studio."""

from __future__ import annotations

import asyncio
import json
import ssl
import time
from collections import OrderedDict

import httpx
from pydantic import BaseModel, ConfigDict

from agentgateway_extproc.config.settings import NoticePreferencesSettings
from agentgateway_extproc.models.destination import ModelDestinationPolicy

_PATH = "/internal/v1/notice-preferences"
_TTL = 30.0
_CAPACITY = 1024
_MAX_BYTES = 2048


class NoticePreferences(BaseModel):
    """Effective display flags; missing and invalid replies use all-on defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    show_no_pii: bool = True
    show_pass: bool = True
    show_changes: bool = True
    show_reroutes: bool = True
    show_timing: bool = True


DEFAULT_PREFERENCES = NoticePreferences()


class NoticePreferencesClient:
    """Share a short-lived, principal-scoped cache across request streams."""

    def __init__(
        self, settings: NoticePreferencesSettings, client: httpx.AsyncClient | None = None
    ) -> None:
        """Create the private mTLS client only when lookup is enabled."""
        self._settings = settings
        self._cache: OrderedDict[tuple[str, str | None], tuple[float, NoticePreferences]] = (
            OrderedDict()
        )
        self._owns_client = client is None and settings.enabled
        self._client = client
        if settings.enabled and client is None:
            if not settings.ca_cert or not settings.client_cert or not settings.client_key:
                raise ValueError("notice preferences require mTLS files")  # noqa: TRY003
            context = ssl.create_default_context(cafile=settings.ca_cert)
            context.load_cert_chain(certfile=settings.client_cert, keyfile=settings.client_key)
            self._client = httpx.AsyncClient(
                base_url=settings.base_url.rstrip("/"),
                verify=context,
                timeout=httpx.Timeout(settings.timeout),
            )

    async def close(self) -> None:
        """Release the owned connection pool."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def get(self, policy: ModelDestinationPolicy) -> NoticePreferences:
        """Fetch at most once per stream; errors never delay model traffic beyond the timeout."""
        if not self._settings.enabled or self._client is None:
            return DEFAULT_PREFERENCES
        personal_id = policy.credential_id if policy.credential_kind == "personal" else None
        key = (policy.principal_id, personal_id)
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached is not None and cached[0] > now:
            self._cache.move_to_end(key)
            return cached[1]
        self._cache.pop(key, None)
        params = {"principal_id": policy.principal_id}
        if personal_id is not None:
            params["credential_id"] = personal_id
        try:
            async with asyncio.timeout(self._settings.timeout):
                async with self._client.stream("GET", _PATH, params=params) as response:
                    response.raise_for_status()
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(content) + len(chunk) > _MAX_BYTES:
                            return DEFAULT_PREFERENCES
                        content.extend(chunk)
                # Reject absent fields, strings masquerading as booleans, and extra data.
                payload = json.loads(content)
                if not isinstance(payload, dict) or set(payload) != set(
                    NoticePreferences.model_fields
                ):
                    return DEFAULT_PREFERENCES
                preferences = NoticePreferences.model_validate(payload, strict=True)
        except (TimeoutError, httpx.HTTPError, ValueError):
            return DEFAULT_PREFERENCES
        self._cache[key] = (time.monotonic() + _TTL, preferences)
        self._cache.move_to_end(key)
        if len(self._cache) > _CAPACITY:
            self._cache.popitem(last=False)
        return preferences
