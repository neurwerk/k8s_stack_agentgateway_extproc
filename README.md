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

### AI Gateway admission, version 5

The `neurwerk.destination_policy` metadata namespace accepts the immutable v5
envelope in [`tests/fixtures/admission-v5.json`](tests/fixtures/admission-v5.json).
Send the same complete envelope on every callback. Its model catalog must contain
only models on the admitted `binding_id` and `provider_id`. The only metadata-less
exception is an empty final response-body callback after validated response headers;
partial or changed metadata is rejected even on that callback.

Supported native operations (all `POST`):

| Profile | Operation | Exact path | Caller credential |
| --- | --- | --- | --- |
| `openai-chat-completions-v1` | `create` | `/v1/chat/completions` | Bearer `authorization` |
| `openai-responses-v1` | `create` | `/v1/responses` | Bearer `authorization` |
| `anthropic-messages-v1` | `create` | `/v1/messages` | `x-api-key` |
| `google-generate-content-v1` | `generateContent` | `/v1beta/models/<model>:generateContent` | `x-goog-api-key` |
| `google-generate-content-v1` | `streamGenerateContent` | `/v1beta/models/<model>:streamGenerateContent?alt=sse` | `x-goog-api-key` |

For `entry_mode: api`, the exact prefix `/bindings/<verified binding_id>` may
precede one of these operation paths. This mount is removed for validation only;
extProc retains the incoming path and the native gateway owns its upstream
rewrite. `forward_proxy` accepts only the exact operation path. Other bindings,
arbitrary prefixes, encoded separators and binding dot segments are rejected.

Requests require JSON, identity encoding and one complete buffered body without
trailers. Other profiles/operations, opaque reasoning input, background Responses,
stored prior Responses and legacy raw attachment passthrough are rejected.
Anthropic requires `anthropic-version: 2023-06-01`; beta headers are unqualified.
Google query credentials are rejected; an optional body model must match the URL.
Anthropic/Google support text and client function tools, native JSON and SSE
restoration, including split text tokens and Anthropic tool-JSON fragments.
Google streamed calls must contain complete, nonrepeated function objects per
candidate/part position. Native media, hosted tools, cached context, thinking and
thought signatures are unqualified and rejected, as are all website operations.
Provider response fields/events use closed schemas; unknown shapes fail closed.
OpenAI output support covers assistant text/refusals, client function `tool_calls`,
and known usage/control fields. Opaque reasoning, nonempty citations/logprobs,
hosted tools and media output are unqualified. JSON responses require complete
envelopes. Responses SSE requires its created/item/part/done lifecycle and an
explicit `response.completed` or `response.incomplete` terminal with closed items.
Chat SSE requires a finish reason for every choice followed by `[DONE]`. Function
arguments must close as a valid JSON object, and repeated snapshots must agree.
EOF is not protocol completion; a failure after response headers aborts the stream.
OpenAI response media type must match the admitted request's `stream` flag.
Responses reflections of `instructions`, function descriptions/parameter schemas,
and `text.format` reuse canonical request extraction for restoration in JSON and
lifecycle snapshots. Only those text leaves use the current request's reversal
map; schema keys, enums, refs and other protocol controls remain protected.

One body-resolved selection controls all-of permissions, PII, content-tracing
selection and the final provider model rewrite, including PII-disabled requests.
extProc keeps the canonical ID in the body `model`, so the native
gateway selects that exact configured Model and then rewrites to the approved
binding's `upstream_model`. Only the canonical ID and unique `upstream_model` alias
are accepted. The gateway must retain the same approved binding. All v5 reroutes
are rejected.
In output metadata, `destination_id` is that canonical ID and `resolved_model` is
the selected binding's upstream model name.

Successful buffered request-body replies emit the separate resolved context under
`neurwerk.request_context`; see the exact
[output fixture](tests/fixtures/resolved-context-v5.json). It contains admitted
identity IDs, selection, policy and tracing facts, never credential values or
reversal material. `billing_classification` is `company_api` for company mode and
`unknown` otherwise: caller credentials cannot prove personal API or subscription
billing. `usage_completeness` is `unknown` at admission. The native gateway extracts
usage from provider responses; late full-duplex metadata is not relied on. Normal
content tracing remains the gateway's responsibility.

`caller_supplied` preserves the profile's approved credential header unchanged
upstream while excluding its value from processing state. `company` strips caller
credentials and requires `provider:<provider-id>:company-use`; the gateway supplies
its approved company credential afterward. `none` rejects provider credentials.
Neurwerk credentials and reserved identity headers are removed upstream. Provider
error replies have fixed safe content; credential response headers and trailers,
including `Set-Cookie`, are stripped. V5 requests never enter raw rejection capture
or the rejection debug exporter, including malformed admission and capacity errors.
Existing unwrapped policy versions keep their wire contracts and routing behavior.

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
