# Request segments

Canonical source: `neurwerk/k8s_stack_agentgateway_extproc`,
`packages/request_segments/`. Package: `neurwerk-request-segments` version `0.1.0`;
Python import: `neurwerk_request_segments`. Pydantic is the only runtime dependency.
The package does not translate providers; AgentGateway owns that boundary.

```python
from neurwerk_request_segments import TextSegment, parse_request, extract_request

request = parse_request(payload, kind="chat")  # chat, responses, or mcp
extracted = extract_request(request, max_depth=32)
segments = extracted.segments  # independent TextSegment(id, text)
request = extracted.rebuild(segments)  # exact IDs and order required
wire = request.model_dump(by_alias=True, exclude_unset=True)
```

`request_kind` and `attachments_present` contain only aggregate protocol facts.
`diagnostic_path(id)` resolves a segment locally. IDs/paths never encode user values
on the Engine wire. JSON-string function arguments are parsed into independent
string leaves, retaining keys and non-string values. Unchanged argument strings
retain their original encoding. Duplicate keys and non-finite numbers fail closed.
JSON Schema prose/defaults/examples are analyzed; identifiers and enum values are
not. Schema objects intentionally remain open, including extension keywords.
Extraction depth errors carry measured, content-free `ExtractionLimitError` facts.

## Compatibility

Authoritative provider models live in `models.py`; extProc aliases these models.
Known current/deprecated controls are retained without adding omitted defaults or
dropping explicit null/false. Responses function tools are flat; Chat tools are
nested. Deprecated Chat `functions` and `function_call` use the same extraction.
MCP analyzes argument string values only; metadata and tool identifiers stay local.
Provider metadata is a bounded string map (64 entries, 64-character keys,
512-character values), not model-visible text. Sampling, routing, cache, identity,
and stream controls are not inspection content. Attachment processing belongs to
the adapter; the package reports their presence and preserves their bodies.
Non-null Responses `previous_response_id` is unsupported: provider-stored history
has no verified local inspection provenance. Omitted/null values stay distinct,
and ordinary request/session policy caching is unchanged. Function choices and
Chat response-format envelopes are closed protocol shapes; only their inner user
JSON schemas remain open.

Unknown fields raise `UnsupportedFeatureError`. Operators may explicitly review a
scalar control with `CompatibilitySettings(controls=[...])`:

```json
{"controls":[{"endpoint":"chat","location":"request","field":"vendor_mode",
"kinds":["string"],"enum":["fast","safe"]}]}
```

Endpoints are `chat`/`responses`. Locations are the fixed `request`, `message`,
`tool`, `function`, `stream_options`, `text`, `text_format`, and `input_item`
shapes. Rules cannot override existing controls, structural fields, content, or
reserved unsupported content features. Kinds are strict `string`, `boolean`,
`integer`, `number`, or `null`; containers/arbitrary paths are not supported.
Rules are limited to 128, enums to 64 values, and extension strings to 256
characters. String extension controls require explicit enums; free-form text
cannot be marked safe by configuration. New content features require extraction
code and regression tests.

## Independent builds and source snapshots

Build this directory with `uv build`. extProc uses this local path dependency;
its Docker build installs both packages. Other repositories may vendor this exact
directory as an independently built dependency. Record owner, package version,
source path, and a SHA-256 source digest in the consumer. Copy exact source, not a
handwritten adapter; refresh snapshots after source changes. Source digests should
cover sorted relative paths and file bytes, excluding caches/build output.
