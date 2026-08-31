"""Start the gRPC ext_proc server and HTTP health sidecar."""

from __future__ import annotations

import asyncio
import logging
from typing import cast

import grpc
import grpc.aio
import uvicorn

from agentgateway_extproc.config.settings import Settings, get_settings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.controllers.health import create_http_app
from agentgateway_extproc.gen import ext_proc_pb2_grpc
from agentgateway_extproc.lib.engine.client import EngineClient


def main() -> None:
    """Run both adapter servers until interrupted."""
    settings = get_settings()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=logging.DEBUG if settings.debug else logging.INFO,
    )
    try:
        asyncio.run(_run(settings))
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Shutting down")


async def _run(settings: Settings) -> None:
    """Create clients and serve gRPC plus HTTP endpoints."""
    client = EngineClient(settings.engine)
    server = create_grpc_server(settings, client)
    server.add_insecure_port("[::]:9000")
    await server.start()
    http_server = uvicorn.Server(uvicorn.Config(create_http_app(client), host="0.0.0.0", port=8000))
    try:
        await asyncio.gather(server.wait_for_termination(), http_server.serve())
    finally:
        await client.close()
        await server.stop(grace=1)


def create_grpc_server(settings: Settings, client: EngineClient) -> grpc.aio.Server:
    """Create the extProc server with an explicit finite receive-message limit."""
    server = cast(
        grpc.aio.Server,
        grpc.aio.server(
            options=[("grpc.max_receive_message_length", settings.grpc_max_receive_message_bytes)]
        ),
    )
    ext_proc_pb2_grpc.add_ExternalProcessorServicer_to_server(
        ExtProcServicer(client, settings), server
    )
    return server


if __name__ == "__main__":
    main()
