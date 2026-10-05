# Request segments

Shared Chat, Responses and MCP text extraction; Pydantic is its only dependency.
AgentGateway owns provider translation.

```python
from neurwerk_request_segments import TextSegment, parse_request, extract_request

request = parse_request(payload, kind="chat")  # chat, responses, or mcp
extracted = extract_request(request, max_depth=32)
segments = extracted.segments  # independent TextSegment(id, text)
request = extracted.rebuild(segments)  # exact IDs and order required
wire = request.model_dump(by_alias=True, exclude_unset=True)
```

`diagnostic_path(id)` resolves locations locally; Engine receives opaque IDs.
Encoded JSON arguments are inspected as separate string leaves; unchanged strings
keep their original encoding. Duplicate keys and non-finite numbers are rejected.
Schema prose, defaults and examples are inspected; identifiers and enums are not.
Schema extensions remain open. Extraction limits raise `ExtractionLimitError`.

## Compatibility

Controls retain omitted/null/false distinctions. Chat tools are nested; Responses
tools are flat. MCP inspects argument strings, not metadata or identifiers.
The adapter processes attachments; this package preserves them and reports presence.
Non-null `previous_response_id` is unsupported because stored history lacks verified
inspection provenance. Unknown protocol fields raise `UnsupportedFeatureError`.

Add reviewed scalar controls with `CompatibilitySettings(controls=[...])`:

```json
{"controls":[{"endpoint":"chat","location":"request","field":"vendor_mode",
"kinds":["string"],"enum":["fast","safe"]}]}
```

Rules select `chat`/`responses` and a fixed location (`request`, `message`, `tool`,
`function`, `stream_options`, `text`, `text_format`, `input_item`). They cannot override
known fields or content. Kinds are `string`, `boolean`, `integer`, `number`, `null`;
strings require an enum. Limits: 128 rules, 64 enum values, 256 characters per string.
New content features require inspection code.

## Independent builds and source snapshots

Canonical source: `neurwerk/k8s_stack_agentgateway_extproc/packages/request_segments`.
Build with `uv build`; extProc uses a local path dependency. Engine and Studio vendor
the exact source for independent builds. Refresh both snapshots after changes and
record version, source and SHA-256 in their `ORIGIN.md`; exclude caches/build output.
