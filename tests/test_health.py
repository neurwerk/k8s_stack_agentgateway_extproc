"""Test health and metrics endpoints."""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from agentgateway_extproc.config.settings import EngineSettings
from agentgateway_extproc.controllers.health import create_http_app
from agentgateway_extproc.lib.engine.client import EngineClient


def test_health_endpoint() -> None:
    """Health responds without authentication."""
    response = TestClient(create_http_app(_engine_client(503))).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_checks_engine_readiness() -> None:
    """Readiness follows the PII engine while liveness remains independent."""
    ready = TestClient(create_http_app(_engine_client(200))).get("/ready")
    unavailable = TestClient(create_http_app(_engine_client(503))).get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ok"}
    assert unavailable.status_code == 503
    assert unavailable.json() == {"status": "unavailable"}


def _engine_client(status: int) -> EngineClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/adapter/ready"
        return httpx.Response(status, request=request)

    return EngineClient(
        EngineSettings(base_url="https://pii-engine.test"),
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://pii-engine.test"
        ),
    )
