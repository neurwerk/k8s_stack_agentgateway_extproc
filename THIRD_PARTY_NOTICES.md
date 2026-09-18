# Third-Party Notices

## YuNet Face Detector

The service bundles OpenCV Zoo's `face_detection_yunet_2023mar.onnx`, copyright
2020 Shiqi Yu, under the MIT license. The model, complete license and immutable
source/checksum record ship together in `agentgateway_extproc/assets/` in the
wheel and runtime. See `src/agentgateway_extproc/assets/YUNET-SOURCE.txt` and
`YUNET-LICENSE.txt`. No model is downloaded at runtime.

## HEIF Decoder

Runtime uses the wheel-only `pi-heif==1.4.0` decoder, whose Python bindings are
BSD-3-Clause; its binary-wheel license is LGPLv3, not BSD alone. The Linux amd64
wheel bundles LGPLv3 libheif `1.23.0` and libde265 `1.1.0`, without x265 or a
HEVC/AV1 encoder. License notices remain in `pi_heif-1.4.0.dist-info/licenses/`.
Sources: [bindings](https://github.com/bigcat88/pillow_heif/tree/v1.4.0),
[libheif](https://github.com/strukturag/libheif/tree/v1.23.0), and
[libde265](https://github.com/strukturag/libde265/tree/v1.1.0).
`pillow-heif==1.5.0` and its GPLv2 binary-wheel encoder are used only by the `dev`
extra to generate synthetic test images; they are not installed in the runtime.

## Envoy Protobuf

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
