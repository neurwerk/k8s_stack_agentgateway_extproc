# Development

## Local checks

Use Python 3.12, `uv`, GNU Make, and Git. Run these commands from this repository's root, including when using a worktree:

```bash
uv sync --frozen --extra dev
make check
```

The [Makefile](../../Makefile) checks the lockfile, Ruff lint and formatting, `ty` types, pytest, regenerated protobuf bindings, and package distributions. Tests use synthetic fixtures and injected clients; these checks do not require a running gateway or PII Engine. `make test` runs the same pytest target. That target excludes `tests/test_image_normalization.py`, `tests/test_images.py`, and `tests/test_documents.py`; it is not the complete pytest collection.

`make check` regenerates `src/agentgateway_extproc/gen/` and requires no Git diff there. Edit protocol definitions under `protos/` when changing bindings, then run `make proto`; do not maintain generated bindings by hand. Package checks build both distributions in a temporary directory and validate licensing metadata.

## Source map

| Path | Responsibility |
| --- | --- |
| `src/agentgateway_extproc/main.py` | Client creation, gRPC/HTTP startup, graceful shutdown. |
| `src/agentgateway_extproc/config/settings.py` | Environment settings and validation. |
| `src/agentgateway_extproc/controllers/` | Transport dispatch, fail-closed handling, health endpoints. |
| `src/agentgateway_extproc/models/` | Admission, destination policy, and Engine data validation. |
| `src/agentgateway_extproc/lib/pipeline/` | Request/response profiles, mutations, streaming, and MCP. |
| `packages/request_segments/` | Shared text extraction and reconstruction. |

The [request-segment package](../../packages/request_segments/README.md) is canonical here; Engine and Studio consume source snapshots for independent builds. Changes to extraction require coordination with those consumers.

Representative tests are [admission](../../tests/test_admission.py), [fail-closed errors](../../tests/test_fail_closed.py), [native profiles](../../tests/test_native_profiles.py), [MCP processing](../../tests/test_mcp_modern.py), [typed attachments](../../tests/test_typed_attachments_v4.py), and [response reflections](../../tests/test_response_reflections.py). They cover permission resolution, analysis bypasses, split SSE tokens, attachment decisions, and request-local restoration.

## Run and build

With the engine HTTPS endpoint and certificate file settings from [Operations](../operations/README.md) already supplied, start the service from the repository root:

```bash
uv run --frozen agentgateway-extproc
```

Default settings alone cannot start the real Engine client: it requires HTTPS, a CA file, and a client certificate/key pair. Unit tests inject an HTTP client instead. The service binds ports `9000` and `8000`; both must be available.

With Docker available, `make build` builds the local image `agentgateway-extproc:local`. The [Dockerfile](../../Dockerfile) verifies the bundled face-detection model and decoder-only HEIF runtime, then runs as a non-root user. Release publication is owned by the [tag workflow](../../.github/workflows/build-push.yaml), not `make build`.
