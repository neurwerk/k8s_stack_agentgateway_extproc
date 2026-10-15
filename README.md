# AgentGateway External Processor

extProc adapts approved API and proxy traffic for PII Engine, validates provider
responses and restores protected text within the current request.

## Request contract

The clean cutover requires the version-5 `neurwerk.destination_policy` envelope.
See [`models/admission.py`](src/agentgateway_extproc/models/admission.py) and the
[wire fixture](tests/fixtures/admission-v5.json). Unwrapped metadata is rejected.
Send identical admission on every callback; only an empty final response-body
callback may omit all metadata after accepted response headers.

API model paths require `/bindings/<verified binding_id>` before the provider
operation. Proxy paths retain their exact provider path. MCP uses
`/bindings/mcp-<server-id>/mcp` and stateless exchanges.

Supported model profiles are OpenAI Chat/Responses, Anthropic Messages and Google
generateContent. Requests use JSON and complete buffered bodies. Responses use
closed JSON/SSE schemas with explicit completion; EOF alone is not completion.
See [`profiles.py`](src/agentgateway_extproc/lib/pipeline/profiles.py) for paths and
[`openai_response.py`](src/agentgateway_extproc/models/openai_response.py) for OpenAI shapes.

Opaque reasoning continuation, stored prior Responses, background execution,
website adapters and unqualified transports are rejected.

## Identity and dispatch

- Employee identity and permissions come from trusted gateway admission.
- Personal provider credentials remain outside retained state and Engine requests.
  Invalid personal credentials never trigger company fallback.
- Company mode requires its company-use grant. The gateway injects the company
  credential after processing; credential-free local targets receive none.
- A company request may reroute before dispatch only to an explicitly approved
  local target whose invocation grant is also present. Personal credentials cannot.
- `neurwerk.request_context` records the selected model and actual dispatch target.
  Credentials, cookies and restoration maps are excluded. Raw request capture is disabled.
- PII and content tracing are independent. Restoration is request-scoped; usage
  completeness remains unknown until the gateway has sufficient provider evidence.

## Development

```sh
uv sync --frozen --extra dev
make check
```

Deployment belongs to the [Base chart](https://github.com/neurwerk/k8s_stack_base/tree/main/charts/agentgateway).
Read [CONTRIBUTING.md](.github/CONTRIBUTING.md) for contribution workflow and
[SECURITY.md](SECURITY.md) for private vulnerability reports.

## Licensing

Project-owned content uses the [MIT License](LICENSE). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for bundled third-party licenses.
