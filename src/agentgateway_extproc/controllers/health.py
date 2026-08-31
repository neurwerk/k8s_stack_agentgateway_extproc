"""Health and Prometheus metrics HTTP endpoints."""

from __future__ import annotations

from importlib.metadata import version as get_version

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest

from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.models.exceptions import EngineUnavailableError


def create_http_app(client: EngineClient) -> FastAPI:
    """Create the unauthenticated health and metrics sidecar."""
    app = FastAPI(title="agentgateway-extproc", version=get_version("agentgateway-extproc"))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        try:
            await client.check_ready()
        except EngineUnavailableError:
            return JSONResponse(status_code=503, content={"status": "unavailable"})
        return JSONResponse(content={"status": "ok"})

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest().decode())

    return app
