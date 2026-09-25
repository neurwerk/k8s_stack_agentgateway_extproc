"""Start the gRPC ext_proc server and HTTP health sidecar."""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import cast

import grpc
import grpc.aio
import uvicorn

from agentgateway_extproc.config.settings import Settings, get_settings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.controllers.health import create_http_app
from agentgateway_extproc.gen import ext_proc_pb2_grpc
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.image_inspection import ImageInspectionClient


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
    image_inspection = ImageInspectionClient(settings.image_inspection)
    docling = DoclingClient(settings.docling, image_inspection=image_inspection)
    server = create_grpc_server(settings, client, docling)
    server.add_insecure_port("[::]:9000")
    await server.start()
    http_server = uvicorn.Server(uvicorn.Config(create_http_app(client), host="0.0.0.0", port=8000))
    # Uvicorn re-raises captured signals after its HTTP shutdown. Keep those
    # signals graceful until the gRPC drain and client cleanup have completed.
    handlers = {
        sig: signal.signal(sig, lambda *_: setattr(http_server, "should_exit", True))
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        await http_server.serve()
    finally:
        try:
            await server.stop(grace=settings.shutdown_grace_seconds)
            await docling.close()
            await image_inspection.close()
            await client.close()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def create_grpc_server(
    settings: Settings, client: EngineClient, docling: DoclingClient | None = None
) -> grpc.aio.Server:
    """Create the extProc server with an explicit finite receive-message limit."""
    server = cast(
        grpc.aio.Server,
        grpc.aio.server(
            options=[("grpc.max_receive_message_length", settings.grpc_max_receive_message_bytes)],
            # Reserve transport headroom so application admission can return a
            # retryable HTTP response rather than a bare gRPC RESOURCE_EXHAUSTED.
            maximum_concurrent_rpcs=2 * settings.grpc_maximum_concurrent_rpcs,
        ),
    )
    ext_proc_pb2_grpc.add_ExternalProcessorServicer_to_server(
        ExtProcServicer(client, settings, docling), server
    )
    return server


if __name__ == "__main__":
    main()
