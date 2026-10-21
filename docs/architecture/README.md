# Architecture

## Traffic and ownership

AgentGateway sends request headers, one buffered request body, response headers, and streamed response bodies to the gRPC listener on port `9000`. Base configures `FailClosed`, buffered requests, and full-duplex streamed responses in its [model routes](https://github.com/neurwerk/k8s_stack_base/blob/main/charts/agentgateway/templates/ai-models.yaml) and [MCP routes](https://github.com/neurwerk/k8s_stack_base/blob/main/charts/agentgateway/templates/mcp-routes.yaml). The processor returns mutations and resolved routing metadata; AgentGateway performs upstream dispatch and provider translation.

The listener uses plaintext gRPC and does not authenticate callers itself. Its trust boundary depends on deployment access restrictions and gateway-supplied admission metadata, not on user-supplied HTTP headers. The separate HTTP listener on port `8000` serves health and metrics without authentication.

Each stream receives an isolated [StreamHandler](../../src/agentgateway_extproc/lib/pipeline/stream_handler.py). It requires immutable admission metadata in `neurwerk.destination_policy`, checks the gateway request ID, resolves an allowed model and its permissions, and rejects metadata changes between callbacks. The only metadata-free callback accepted is an empty final response body after response headers have been accepted. [Admission models](../../src/agentgateway_extproc/models/admission.py) define the accepted shape and reroute grants.

The [engine client](../../src/agentgateway_extproc/lib/engine/client.py) calls `/v2/adapter/analyze-segments` over HTTPS with mutual TLS. It sends text segments with opaque IDs, not the full provider request or provider credentials. The adapter checks reply structure, segment identity, permitted mutations, and reversal mappings before rebuilding the request. PII Engine owns detection and policy decisions; the adapter applies blocks, transformations, and explicitly authorized reroutes.

## Request formats and inspection

[Profile definitions](../../src/agentgateway_extproc/lib/pipeline/profiles.py) select exact operations. API paths are validated after removing the trusted binding mount, normally `/bindings/<binding-id>`. Forward-proxy paths are validated directly.

| Profile | Operation path | Inspected content |
| --- | --- | --- |
| OpenAI Chat Completions | `/v1/chat/completions` | Supported message text, tool content, arguments, and schema prose. |
| OpenAI Responses | `/v1/responses` | Supported input text, tool content, arguments, and schema prose. |
| OpenAI Codex Responses | `/v1/responses`; `/backend-api/codex/responses` in forward-proxy mode | The supported Responses subset. |
| Anthropic Messages | `/v1/messages` | Native text, client function calls/results, tool descriptions, and schema prose. |
| Google generateContent | `/v1beta/models/<model>:generateContent` or `:streamGenerateContent?alt=sse` | Native text, client function calls/results, tool descriptions, and schema prose. |
| MCP Streamable HTTP | `/mcp/<canonical-server-id>` | String leaves in `tools/call` arguments when PII processing is enabled. |

Model requests require POST, JSON, and no request compression. Unknown content features are rejected. Recognized web-profile identifiers do not imply implemented web inspection. Native Anthropic and Google profiles reject media, hosted tools, stored context, and opaque thinking/signature blocks. Responses background execution and stored-history continuation are unsupported. The [request-segment guide](../../packages/request_segments/README.md) defines extraction and reviewed scalar compatibility extensions.

OpenAI attachment handling is selected separately from text PII processing. Trusted document and image policies can block, extract text, or permit normalized images under explicit forwarding rules. [Document processing](../../src/agentgateway_extproc/lib/documents.py), [Docling integration](../../src/agentgateway_extproc/lib/docling.py), and [image policy](../../src/agentgateway_extproc/lib/image_policy.py) enforce format and resource limits. Optional image inspection contributes visual findings; the Engine validates the combined policy input. An explicitly allowed `pii-unchecked` local image route does not perform privacy inspection on that image.

Disabling text PII processing skips text analysis, not admission, protocol validation, limits, or attachment policy. MCP lifecycle messages and tool calls without text arguments do not trigger Engine analysis. MCP batches, resumable streams, and task extensions are rejected. Requests carrying `Mcp-Session-Id` receive HTTP 404; stateful MCP sessions are not supported by the controller.

## Responses and failures

JSON and server-sent events (SSE) are parsed according to the selected profile. [Response processing](../../src/agentgateway_extproc/lib/pipeline/response.py) and [native response handling](../../src/agentgateway_extproc/lib/pipeline/native_response.py) restore only eligible semantic fields using the current request's reversal map. They preserve protocol structure and do not run general PII detection over generated output. SSE processing handles split placeholders and validates event order and completion. Supported reasoning replay uses Engine-backed fingerprints rather than treating arbitrary opaque history as inspected.

Reversal state is cleared on completion, failure, or cancellation. Eligible OpenAI responses can include notices; native profiles disable notice injection. Studio's optional notice-preference lookup affects display only and falls back to all-on display defaults on lookup failure.

The [gRPC controller](../../src/agentgateway_extproc/controllers/grpc_servicer.py) returns a local error before response headers are committed. After commitment, processing failure aborts the gRPC stream rather than forwarding unprocessed content. Engine transport failures and invalid replies fail closed; validated policy errors retain their defined HTTP status. Concurrency or retained-body exhaustion returns retryable HTTP 503 with `Retry-After: 1`.
