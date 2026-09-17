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

Typed attachment content parts in Chat Completions and Responses messages,
including history, follow the selected model's trusted attachment mode:

| Mode | Behavior |
| --- | --- |
| `block` (default) | Reject attachments with HTTP 403, independently of PII. |
| `extract` | Convert allowed inline documents to complete text parts through Docling, then apply PII when enabled; other attachment types return 403. |
| `passthrough` | Forward the original request and provider response unchanged; valid only when that model has PII disabled. |

The trusted version-1 metadata retains its `models` map of model IDs to PII
booleans and optionally adds an `attachment_modes` map using those same IDs.
Omitted modes default to `block`. Unknown modes/model IDs and `passthrough` with
PII enabled are invalid platform metadata and fail closed. Request bodies and
caller headers cannot override this trusted policy. Upgrade the extProc consumer
before configuring modes in the platform producer; older consumers reject the
new metadata field.

Passthrough is an explicit grant, never a fallback after extraction or PII fails.
It does not imply the destination is local, disable tracing, fetch referenced
files, or guarantee the provider supports the attachment. Existing request-size
and protocol bounds still apply.
Ordinary text-only bypass, arbitrary tool JSON and MCP handling are unchanged.

### Document Conversion

Version `0.8.0` supports PDF, DOCX, XLSX, PPTX, UTF-8 TXT, Markdown and CSV.
Chat uploads use `{"type":"file","file":{"filename":"report.pdf","file_data":"data:application/pdf;base64,..."}}`;
Responses use `{"type":"input_file","filename":"report.pdf","file_data":"data:application/pdf;base64,..."}`.
Only exact MIME/extension-matched, strict base64 data URIs are accepted, not raw
base64, URLs, file IDs or extra references. Filenames are limited to 256 characters,
reject controls and retain only the display basename. Docling receives a constant
`upload.ext` name. All files, including history, are checked before any submission.
PDFs must be valid, unencrypted and within the aggregate page limit; Office ZIPs
are checked in memory (100 MiB expanded, 25 MiB per entry, 10,000 entries, safe
paths, no encryption, required OOXML parts), never extracted to disk.
PDF page/encryption inspection runs in a disposable Python subprocess with 256 MiB
address-space and five-second CPU limits, a ten-second parent timeout, bounded
stdin (40 MiB hard ceiling), and no parser output or temporary files. The parent
kills and reaps a timed-out helper before releasing preflight admission.
XLSX XML is inspected without building a tree: DTDs are rejected, and actual cell
coordinates, dimensions, merged/hyperlink ranges and native chart references must
fit the 100,000-position budget. XML is limited to 200,000 elements and depth 32. CSV
delimiter/quote counts are bounded before parsing; its row/column product must
also fit 100,000 positions.

The verified-HTTPS client targets native docling-serve `1.33.0` revision
`27fa2aa9638e449d7fcd4364ffcde8d9a47bc4eb` (jobkit `3.6.0`, Docling `2.127.0`,
DoclingDocument schema `1.10.0`). It submits one multipart file to
`/v1/convert/file/async`, polls `/v1/status/poll/{task_id}`, then reads
`/v1/result/{task_id}`. It never retries submissions or forwards caller credentials.
CPU uses standard/RapidOCR/table extraction; remote uses the administrator's
`default` VLM preset. Enrichments and image exports are off. TXT uses the Markdown
backend. Caller options, URL sources and callbacks are never forwarded.

Only successful, error-free, nonempty results are accepted. A bounded reference
walk projects body/furniture text, table rows and picture captions, excluding
images, `orig`, source URLs and metadata. Unsupported structures, broken references,
cycles, incomplete PDF page sets and orphaned content fail closed. Limits include
20,000 nodes, 100,000 table positions/reference visits and depth 32. Each document
becomes one text part, including its filename. The complete converted request is
limited to 5 MiB serialized and 4,000,000 text characters, even with PII disabled.
There is no truncation, raw fallback, line/cell PII reconstruction or embedding path.
Formula items with missing/empty canonical text fail closed, never using `orig`.
Checkbox labels retain their state as `[x]` or `[ ]`, counted in output limits.
Incoming requests, Docling JSON and PII replies are checked before JSON tree
allocation: maximum depth 64 and 200,000 lexical tokens, with quoted strings
counted once regardless of base64/text length. Existing byte limits and strict
duplicate-key/non-finite-number rejection still apply.

PII-enabled converted chats call `POST /v1/adapter/analyze-document-request` once
with the existing Chat/Responses request and `EngineReply` contracts, using a fresh
session scope. Mutation/reversal checks use the converted request. PII-disabled
destinations receive that converted body, without guard injection or PII state.
Opaque assistant reasoning and protocol controls are preserved in both paths.

Each extProc process admits one conversion batch with no waiting queue. Cancellation
or the shared batch deadline stops subsequent files but retains admission until
the in-flight native job (or preflight thread) ends. Lost submission/status replies
disable conversion in that process until an operator checks native jobs and then
restarts it. This does **not** change readiness or trigger automatic restarts.
Two fixed replicas therefore admit two batches in steady state, not a global or
persistent lock. Shutdown drains for at most five seconds; process death/restarts
can leave native jobs running. Deployment termination grace and operator checks
remain necessary. Errors are fixed 400/413/503/504 messages without content.

PDF resource isolation is not a full security sandbox, and OCR/model completeness
cannot be proved by schema validation. Other parsers and native Docling still need
worker resource limits. Remote inference sees unredacted pages before PII. Embedded
images and arbitrary picture/chart metadata are not forwarded. Keep Docling's content-bearing
logs suppressed and backend resource fetching disabled, and enforce network and
worker resource limits. PII-disabled extracted text is not PII-sanitized.

All gateway MCP traffic is stateless. At the trusted request-header stage,
any `Mcp-Session-Id` header (including empty or duplicate headers, regardless of
case) is rejected with HTTP 404 and the fixed JSON error below, before upstream
dispatch or PII Engine processing, even when PII is disabled:

```json
{"error":"Stateful MCP sessions are currently unsupported pending an AgentGateway session-ownership fix. Reinitialize without Mcp-Session-Id."}
```

There is no compatibility toggle. Activation of this binary requires a stateless
AgentGateway configuration. Model conversation headers and JSON `session_id`
fields are not transport-session headers and are unaffected by this rejection.

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

Document settings use `EXTPROC_DOCLING__` plus these names (byte units are bytes):

| Name | Default | Allowed Bound |
| --- | --- | --- |
| `ENABLED` | `false` | Boolean |
| `BASE_URL` | `https://docling.docling.svc` | HTTPS origin, no credentials/path/query |
| `CA_CERT` | unset | CA file path; otherwise system trust |
| `API_KEY` | unset | Required secret when enabled, sent only as `X-Api-Key` |
| `INFERENCE_MODE` | `cpu` | `cpu` or `remote` |
| `TIMEOUT` | `360` | Whole batch, >0 through 3660 seconds |
| `DOCUMENT_TIMEOUT` | `300` | Native per-document budget, >0 through 3600 seconds |
| `FILE_BYTES` | `20971520` | 1 through 41943040 |
| `TOTAL_BYTES` | `41943040` | 1 through 41943040 |
| `COUNT` | `5` | 1 through 20 |
| `PAGES` | `200` | Aggregate, 1 through 1000 |
| `MAX_RESPONSE_BYTES` | `16777216` | 1024 through 16777216 |

Docling HTTP calls have a fixed 30-second bound; task status replies are capped at
64 KiB. All files share the Docling batch deadline. `EXTPROC_ENGINE__TIMEOUT` bounds
the whole PII HTTP call, including reading its response, not just individual I/O
operations. Foreground conversion/PII waits are therefore bounded by the batch
deadline plus the PII whole-call deadline (360 + 615 = 975 seconds when the platform
configures a 615-second engine timeout; the standalone engine default remains 5).
AgentGateway 1.5's backend HTTP `requestTimeout` covers only initial gRPC response
headers, not end-to-end conversion/PII processing. Native jobs may continue after
the caller deadline; admission remains held while they drain as described above.
`EXTPROC_MAX_REQUEST_BYTES` defaults to 5242880 and allows up to 67108864 (64 MiB).
`EXTPROC_GRPC_MAX_RECEIVE_MESSAGE_BYTES` defaults to 6356992 and allows up to
68222976 (65 MiB + 64 KiB); it must exceed the request limit.
`EXTPROC_GRPC_MAXIMUM_CONCURRENT_RPCS` defaults to 4, bounded 1 through 16,
independently of the HTTP health listener. Transformed model requests stay capped
at 10 MiB. Increasing upload transport limits does not raise converted-output limits.

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
