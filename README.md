# neurwerk.base - AgentGateway External Processor

AgentGateway External Processor adapts gateway requests and responses for the PII Engine in neurwerk.base. See the [neurwerk.base website](https://base.neurwerk.com/) for more information.

| Repository | Description |
| --- | --- |
| [Base chart](https://github.com/neurwerk/k8s_stack_base) | Shared platform charts and release packages that form the foundation of the stack. |
| [Studio](https://github.com/neurwerk/k8s_stack_studio) | Web dashboard and API for operating AI platform services. |
| [Tooling](https://github.com/neurwerk/k8s_stack_tooling) | One container image plus separate CLI tools for setup and operations. |
| [PII Engine](https://github.com/neurwerk/k8s_stack_pii_engine) | Service that uses Presidio to evaluate PII and apply safety policies. |
| [AgentGateway External Processor](https://github.com/neurwerk/k8s_stack_agentgateway_extproc) | Adapter that processes gateway requests and responses with the PII Engine (**this repo**). |
| [Keycloak API Key Bridge](https://github.com/neurwerk/k8s_stack_keycloak_api_key_bridge) | Separate service that issues and validates API keys using Keycloak permissions. |
| [Keycloak Theme](https://github.com/neurwerk/k8s_stack_keycloak_theme) | Customized Keycloak login pages and emails. |
|  |  |
| [Example client chart](https://github.com/neurwerk/k8s_stack_client_example_com) | Reference client configuration and Flux deployment setup to adapt for a new client. |
|  |  |
| [Dify Add-on](https://github.com/neurwerk/k8s_stack_addon_dify) | Optional Dify package with API and web customizations, including single-workspace enforcement. |

## Contributing and support

### Optional rejection debug project

`EXTPROC_REJECTION_LANGFUSE__ENABLED` defaults to `false`. When enabled, configure
`ENDPOINT` (full `/api/public/otel/v1/traces` URL), `PUBLIC_KEY` and `SECRET_KEY`
under the same prefix, using credentials belonging only to a separate Langfuse
debug project. HTTPS uses system certificate trust; private HTTP requires explicit
`ALLOW_HTTP=true` and isolated network access. Keys belong in runtime Secrets.
Before sending any content, the worker verifies `/api/public/projects` returns
exactly the project and organization IDs `rejected-request-debug`. Failed or wrong
identity checks prevent export; successful verification is cached for the process.
Missing endpoint/keys leaves export inactive without failing extProc startup.

Defaults: `MAX_CAPTURE_BYTES=65536` (1024–1048576), `QUEUE_CAPACITY=32` (1–128),
`TIMEOUT_SECONDS=2` (greater than zero, at most 10). A single worker sends
OTLP/HTTP JSON with v4 ingestion semantics. Queue saturation drops events;
delivery errors never replace client rejections. Shutdown drains for at most one
export timeout. `extproc_rejection_exports_total` reports exported/failed/dropped
outcomes without content labels. Delivery is best effort, without retries.
Missing configuration increments `unconfigured`; inactive submissions are dropped.
Each queued record retains at most `MAX_CAPTURE_BYTES` original bytes; the worker's
encoded envelope is capped at `6 * MAX_CAPTURE_BYTES + 16384` bytes.
JSON parsing, credential redaction and serialization run on one dedicated thread,
with at most one submitted preparation. Shutdown may leave that single bounded
computation finishing without network delivery; pending captures are discarded.

Only immediate HTTP 400/413/403/503 rejections during processing of a nonempty
request-body message are captured. Request-body capacity rejection is included,
preserving its retry response. Successful requests, provider response content,
headers and transport-level rejections are excluded. Trace metadata includes the
generated request ID, status, limit facts and capture completeness. Valid complete
JSON within the byte budget is recursively scrubbed of known structured credential
and header fields; malformed, incomplete and oversized JSON produces metadata-only
traces because credential boundaries cannot be established safely. Sanitized input
is byte-bounded and may be truncated. Arbitrary secrets in ordinary prose cannot
be reliably removed. Treat the debug project as sensitive and configure access and
retention separately; existing temporary file capture remains independent.

This is an explicit operator override independent of destination
`contentTracingEnabled`. It must not use normal tracing or Studio project keys.

- **Contributions:** Read [CONTRIBUTING.md](.github/CONTRIBUTING.md) before proposing a change.
- **Bug reports and feature requests:** Use [GitHub Issues](https://github.com/neurwerk/k8s_stack_agentgateway_extproc/issues) for reproducible bugs and clearly scoped feature requests.

## Security

Report vulnerabilities privately by following the instructions in [SECURITY.md](SECURITY.md).

## Licensing

Project-owned content is licensed under the [MIT License](LICENSE). Third-party content retains its upstream license.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and licensing information for bundled models, decoders, and protobuf files.
