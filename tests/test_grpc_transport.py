"""Test the concrete aio gRPC server transport limits."""

from __future__ import annotations

import asyncio

import grpc
import grpc.aio

from agentgateway_extproc.config.settings import Settings
from agentgateway_extproc.gen import ext_proc_pb2_grpc
from agentgateway_extproc.main import create_grpc_server

from .conftest import body_request, header_request


async def test_aio_server_receives_processing_message_above_four_mibibytes(
    engine_client,
) -> None:
    settings = Settings(grpc_maximum_concurrent_rpcs=1)
    server = create_grpc_server(settings, engine_client)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(
        f"127.0.0.1:{port}",
        options=[("grpc.max_send_message_length", settings.grpc_max_receive_message_bytes)],
    )

    async def requests(large=True):
        yield header_request()
        if large:
            yield body_request(b"x" * (4 * 1_048_576 + 1), end_of_stream=False)

    try:
        stub = ext_proc_pb2_grpc.ExternalProcessorStub(channel)
        responses = [response async for response in stub.Process(requests())]
        active = stub.Process()
        await active.write(header_request())
        assert (await active.read()).HasField("request_headers")
        busy = stub.Process()
        overload = await busy.read()
        assert overload.immediate_response.status.code == 503
        assert any(
            item.header.key == "retry-after"
            for item in overload.immediate_response.headers.set_headers
        )
        assert await busy.read() is grpc.aio.EOF
        await busy.done_writing()
        active.cancel()
        await active.code()
        # Cancellation must release admission without restarting the process.
        admitted = False
        for _ in range(50):
            await asyncio.sleep(0.01)
            try:
                recovered = [response async for response in stub.Process(requests(large=False))]
            except grpc.aio.AioRpcError as error:
                assert error.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
                continue
            if recovered[0].HasField("request_headers"):
                admitted = True
                break
        assert admitted
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert responses[0].HasField("request_headers")
    assert responses[-1].immediate_response.status.code == 400
