# AgentGateway extProc

`agentgateway-extproc` is the Envoy External Processing (`ext_proc`) adapter
between [AgentGateway](https://github.com/agentgateway/agentgateway) and the
Neurwerk PII Engine. It validates trusted routing metadata, adapts supported
model and MCP traffic to the engine API, applies approved request mutations,
and safely processes responses. PII detection, policy, and routing decisions
remain owned by the PII Engine.

## Architecture

```text
AgentGateway
    | Envoy ext_proc gRPC (:9000)
    v
agentgateway-extproc ---- mTLS HTTP ----> PII Engine
    |
    +---- health, readiness, and Prometheus metrics HTTP (:8000)
```

The adapter is fail-closed at malformed protocol, metadata, engine-response,
and placeholder boundaries. `/health` checks the process; `/ready` verifies the
PII Engine path; `/metrics` exposes bounded operational metrics. Deployment
network policy and workload identity are outside this repository.

## Packages

- Repository: `neurwerk/k8s_stack_agentgateway_extproc`
- Python distribution and command: `agentgateway-extproc`
- Python import package: `agentgateway_extproc`
- Container image: `ghcr.io/neurwerk/k8s-stack-agentgateway-extproc`

## Configuration

Settings are supplied through `EXTPROC_` environment variables. Nested engine
settings use `__`, for example `EXTPROC_ENGINE__BASE_URL`. Configuration falls
into these categories:

- Engine connectivity: `ENGINE__BASE_URL`
- Workload trust paths: `ENGINE__CA_CERT`, `ENGINE__CLIENT_CERT`, and
  `ENGINE__CLIENT_KEY`
- Engine deadlines: `ENGINE__TIMEOUT` and `ENGINE__READINESS_TIMEOUT`
- Request and response bounds: `MAX_REQUEST_BYTES`, `MAX_RESPONSE_BYTES`,
  `MAX_TRANSFORMED_REQUEST_BYTES`, `GRPC_MAX_RECEIVE_MESSAGE_BYTES`, and
  `ENGINE__MAX_RESPONSE_BYTES`
- Diagnostics: `DEBUG`

Certificate and private-key settings are filesystem paths. Inject sensitive
material at runtime; do not put credentials, private keys, certificates, or
real request payloads in repository files or environment examples.

## Development

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev
make check
```

`make check` verifies the lockfile, Ruff lint and formatting, strict `ty` type
checking, tests with the configured coverage threshold, and deterministic
protobuf generation. Regenerate checked-in bindings after an intentional proto
change with:

```bash
make proto
```

A local container can be built with `make build`.

The Dockerfile keeps version tags for readability and pins their OCI image
indexes by digest. When updating the Dockerfile frontend, uv, or Python image,
inspect the authoritative registry manifest and confirm that the selected index
contains a `linux/amd64` manifest before replacing both the version and digest:

```bash
docker buildx imagetools inspect docker/dockerfile:<version>
docker buildx imagetools inspect ghcr.io/astral-sh/uv:<version>
docker buildx imagetools inspect python:<version>-slim
docker build --check .
docker build --platform linux/amd64 -t agentgateway-extproc:validation .
```

## Releases

`pyproject.toml` and `uv.lock` are the version sources. Pull requests and pushes
to `main` run quality gates. An explicit `v<version>` tag additionally publishes
the Linux AMD64 container only when the tag exactly matches `project.version`.
The image receives exact SemVer and moving major/minor tags; no `latest` tag is
published. Deployments should pin the full version-specific tag.

## Security

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
Do not include secrets, certificates, credentials, or sensitive request content
in a report.

## License

Neurwerk-authored code is licensed under the [MIT License](LICENSE). Vendored
Envoy/AgentGateway protobuf definitions and their generated derivatives remain
under Apache-2.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
