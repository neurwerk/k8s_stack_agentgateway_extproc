# Operations

## Configuration and deployment

Base owns the [extProc chart](https://github.com/neurwerk/k8s_stack_base/tree/main/charts/agentgateway-extproc) and [gateway chart](https://github.com/neurwerk/k8s_stack_base/tree/main/charts/agentgateway). Its [deployment template](https://github.com/neurwerk/k8s_stack_base/blob/main/charts/agentgateway-extproc/templates/deployment.yaml) maps values to environment variables, mounts certificates, and defines probes. Use the selected release's charts for effective image pins, access restrictions, and settings.

[Settings](../../src/agentgateway_extproc/config/settings.py) are loaded at process startup. Variables use `EXTPROC_` with `__` for nested fields; `ERROR_LOG_LEVEL` is the separate error-detail setting.

| Settings | Purpose and constraints |
| --- | --- |
| `EXTPROC_ENGINE__BASE_URL`, `__CA_CERT`, `__CLIENT_CERT`, `__CLIENT_KEY` | Required real-client HTTPS endpoint and mutual-TLS file paths. The HTTP default URL is not a working production configuration. |
| `EXTPROC_ENGINE__TIMEOUT`, `__READINESS_TIMEOUT`, `__MAX_RESPONSE_BYTES` | Bound analysis duration, readiness calls, and Engine replies; readiness timeout is at most one second. |
| `EXTPROC_MAX_REQUEST_BYTES`, `EXTPROC_MAX_TRANSFORMED_REQUEST_BYTES`, `EXTPROC_MAX_RESPONSE_BYTES` | Independent bounds for input, rebuilt requests, and responses. |
| `EXTPROC_GRPC_MAX_RECEIVE_MESSAGE_BYTES` | Must exceed the accepted request-body size to leave room for the gRPC envelope. |
| `EXTPROC_GRPC_MAXIMUM_CONCURRENT_RPCS` | Application admission limit; transport reserves extra capacity to return controlled overload responses. |
| `EXTPROC_DOCLING__*` | Optional conversion service, HTTPS origin, API key, timeouts, and document limits. |
| `EXTPROC_IMAGE_INSPECTION__*` | Optional fixed image-inspection endpoint, API key, model, and timeout; HTTP requires explicit `ALLOW_HTTP`. |
| `EXTPROC_NOTICE_PREFERENCES__*` | Studio display preferences over mutual TLS. Optional in the service; required by the inspected Base deployment template. |
| `EXTPROC_COMPATIBILITY__CONTROLS` | JSON list of reviewed scalar extensions described in the [shared package guide](../../packages/request_segments/README.md). |
| `EXTPROC_MAX_REASONING_FINGERPRINTS` | Bound retained reasoning fingerprints. |
| `EXTPROC_LOG_LEVEL`, `ERROR_LOG_LEVEL` | General logging and client-facing error detail respectively. General DEBUG logging can include request content. |
| `EXTPROC_SHUTDOWN_GRACE_SECONDS` | Time allowed to drain active gRPC calls before closing clients. |

For rows with abbreviated `__` names, retain the full prefix from the first variable. Exact defaults and allowed ranges live in the settings source. Body limits, gateway buffering, service timeouts, and deployment termination grace must agree across the Base charts; increasing only the processor limit does not increase the gateway's accepted body size.

## Verify service health

For an already running local instance, use `curl` from any directory:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --fail --silent --show-error http://127.0.0.1:8000/ready
curl --fail --silent --show-error http://127.0.0.1:8000/metrics
```

`/health` reports process liveness. `/ready` calls the Engine's `/v2/adapter/ready` through the configured mutual-TLS client and returns 503 when unavailable. It does not probe Docling, image inspection, Studio, or a provider. `/metrics` exposes Prometheus counters and timings defined in [metrics.py](../../src/agentgateway_extproc/metrics.py). A successful readiness probe is not an end-to-end inspection test; use the approved gateway acceptance procedure in the shared [authentication guide](https://github.com/neurwerk/docs/blob/main/dev/authentication/README.md).

## Diagnose and recover

| Symptom | Check and responsible owner |
| --- | --- |
| Startup rejects Engine configuration | Check the HTTPS URL and configured TLS file paths against the Base deployment; the real client does not support plaintext fallback. |
| `/health` succeeds but `/ready` returns 503 | Check Engine readiness, connectivity, and the workload TLS configuration through the platform operator. |
| `invalid_metadata` or unsupported operation | Compare gateway admission and profile settings with the adapter's supported schema; coordinate gateway and adapter changes in Base. |
| `capacity_unavailable` | Check active-stream metrics and request size/concurrency settings. The response is retryable after the supplied delay. |
| Stream terminates after headers | Correlate the request ID with processor failure phase/reason; committed responses cannot be replaced with a new HTTP error. |
| Stateful MCP request returns 404 | The controller rejects `Mcp-Session-Id`; use a supported stateless flow. |

Correct configuration through its owning chart and the selected release's recovery path. A restart drains active calls for the configured grace period but cannot recover an interrupted response or its request-local reversal map. Deployment verification and recovery follow the shared [upgrade guide](https://github.com/neurwerk/docs/blob/main/dev/operations/upgrades.md).

## Image publication

The [release workflow](../../.github/workflows/build-push.yaml) runs quality checks on pull requests and main. Version tags additionally validate the declared package version, publish the `linux/amd64` image, verify its digest and source revision, and publish release notes. Follow the shared [image-release procedure](https://github.com/neurwerk/docs/blob/main/dev/operations/image-releases.md) for publication and deployment-pin changes; publishing this image does not deploy it.
