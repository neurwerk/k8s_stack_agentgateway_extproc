# Third-Party Notices

This project includes Envoy-compatible protobuf definitions sourced from the
AgentGateway project:

- `protos/ext_proc.proto`
- `protos/shared_envoy.proto`

The checked-in sources are byte-for-byte copies from AgentGateway `v1.4.1`
(commit `163ea2146acb7b82082acea30ed691b29079095f`):

- <https://github.com/agentgateway/agentgateway/blob/v1.4.1/crates/protos/proto/ext_proc.proto>
- <https://github.com/agentgateway/agentgateway/blob/v1.4.1/crates/protos/proto/shared_envoy.proto>

These definitions implement and adapt Envoy API types. Canonical Envoy API
sources are maintained at <https://github.com/envoyproxy/data-plane-api>.
AgentGateway is maintained at <https://github.com/agentgateway/agentgateway>.

The following files are generated derivatives of those definitions using the
protobuf and gRPC Python tooling recorded in `uv.lock`:

- `src/agentgateway_extproc/gen/ext_proc_pb2.py`
- `src/agentgateway_extproc/gen/ext_proc_pb2.pyi`
- `src/agentgateway_extproc/gen/ext_proc_pb2_grpc.py`
- `src/agentgateway_extproc/gen/shared_envoy_pb2.py`
- `src/agentgateway_extproc/gen/shared_envoy_pb2.pyi`
- `src/agentgateway_extproc/gen/shared_envoy_pb2_grpc.py`

The protobuf definitions and their generated derivatives are distributed under
the Apache License, Version 2.0. Copyright is held by their respective Envoy,
AgentGateway, and other contributors. No endorsement by those projects is
implied.

The complete Apache-2.0 license text is included at
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt). It is also available in the
AgentGateway source at
<https://github.com/agentgateway/agentgateway/blob/v1.4.1/LICENSE>.
