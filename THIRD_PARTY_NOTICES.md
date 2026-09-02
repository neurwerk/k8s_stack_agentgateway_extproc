# Third-Party Notices

This project includes Envoy-compatible protobuf definitions sourced from the
AgentGateway project:

- `protos/ext_proc.proto`
- `protos/shared_envoy.proto`

The checked-in sources are byte-for-byte copies from AgentGateway `v1.5.0`
(commit `fe6732474a96a0363dfb9822859af4e9bab360fa`):

- [`ext_proc.proto`](https://github.com/agentgateway/agentgateway/blob/fe6732474a96a0363dfb9822859af4e9bab360fa/crates/protos/proto/ext_proc.proto), Git blob `4d618ceed76fd1c147511e9bfb86fecf0cbc592a`
- [`shared_envoy.proto`](https://github.com/agentgateway/agentgateway/blob/fe6732474a96a0363dfb9822859af4e9bab360fa/crates/protos/proto/shared_envoy.proto), Git blob `4d679ea327c46d9c3d5e86dd5334640385543469`

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
<https://github.com/agentgateway/agentgateway/blob/fe6732474a96a0363dfb9822859af4e9bab360fa/LICENSE>.
