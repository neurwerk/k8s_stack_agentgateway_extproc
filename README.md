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
| `extract` / `process` | Convert allowed inline documents and version-two/three images to text parts through Docling, then apply PII when enabled. Forwarding image pixels additionally requires an explicit image policy. `process` requires version two or three, not an extra permission. |
| `passthrough` | Preserve the original request bytes and protocol parts unchanged in all versions, without normalization, Docling or face detection. PII and face protection must be disabled, and no image-forwarding policy may be set. |

The trusted version-1 metadata retains its `models` map of model IDs to PII
booleans and optionally adds an `attachment_modes` map using those same IDs.
Omitted modes default to `block`. Unknown modes/model IDs and `passthrough` with
PII enabled are invalid platform metadata and fail closed. Request bodies and
caller headers cannot override this trusted policy. Upgrade the extProc consumer
before configuring modes in the platform producer; older consumers reject the
new metadata field.

Version two adds optional sparse `image_forwarding`, `face_protection` and
`local_models` maps. All keys must exist in `models`; booleans are strict and
unknown fields are rejected. Version one rejects these maps even when empty.
Face protection defaults to true except for legacy passthrough; forwarding
defaults to `none`. MCP metadata remains version one.

Passthrough is an explicit grant, never a fallback after extraction or PII fails.
It does not imply the destination is local, disable tracing, fetch referenced
files, or guarantee the provider supports the attachment. Existing request-size
and protocol bounds still apply. Inline image bytes and URL references remain
untouched; the passthrough backend owns URL handling.
Ordinary text-only bypass, arbitrary tool JSON and MCP handling are unchanged.

### Document Conversion

The current source supports PDF, DOCX, XLSX, PPTX, UTF-8 TXT, Markdown and CSV.
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
`internal-standard` uses standard/RapidOCR/table extraction; `private-vlm` uses the administrator's
`default` VLM preset for PDFs and `images` for native images. Enrichments and image exports are off. TXT uses the Markdown
backend. Caller options, URL sources and callbacks are never forwarded.

Only successful, error-free results are accepted; text must be nonempty except
for the version-three no-text image marker described below. A bounded reference
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
session scope. Mutation/reversal checks use the converted request. Without v3
face protection, PII-disabled destinations receive that converted body without
guard injection or PII state; v3 face reporting uses the envelope described below.
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

### Private Images

Version `0.10.0` adds version-three face policy and CPU/HEIC image processing.
Compatible extProc, PII Engine and Base must precede v3 metadata.
Publication, verified image pinning
and activation remain separate approvals; v1/v2 defaults and behavior are preserved.

#### Version Two

Version-two processed images accept only inline JPEG/PNG in
`{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`
(Chat) or `{"type":"input_image","image_url":"data:image/png;base64,..."}`
(Responses). URLs, file IDs, extra fields, wrong MIME types, animation and
multi-frame images are rejected. Image parts anywhere in history are processed
in their original order, alongside ordinary documents and text.

| Image forwarding | Requirements and result |
| --- | --- |
| `none` (default) | Accept supported images, privately extract their text, apply ordinary text PII when enabled, and forward only text. Never restore image pixels or run YuNet, even when face protection is enabled or the image contains faces. Documents retain their existing text-only path. |
| `if-no-pii-detected` | Requires `process`/`extract`, PII enabled and face protection enabled. Forward only after a complete private-VLM extraction, successful YuNet check and one fresh document PII scan of the whole converted request. Any entity, including a policy `pass` or masked entity, face, cached/ambiguous result or non-pass decision withholds all images and rejects the request with a text-only retry message. |
| `pii-unchecked` | Requires `process`/`extract`, face protection false and `local_models[selected] == true`. This is trusted producer proof of a concrete local route, never a model-name guess. Images still need private-VLM extraction. Ordinary text PII settings, policy blocks, errors and output limits still apply. |

`private-vlm` must be an operator-qualified private reader compatible with the
pinned Docling output contract; a mode label alone cannot prove model quality.
Images cannot use the standard pipeline, and no automatic pipeline fallback exists.
`cpu` and `remote` remain accepted aliases for `internal-standard` and `private-vlm`.
The selected pipeline also applies to PDFs; Office formats keep their native parsers.
Native image conversion sends `from_formats=image`, a constant `upload.png`, the
administrator-owned `images` VLM preset and the same validated DoclingDocument
projection used for documents. This named preset must set `VlmConvertOptions.scale`
to `1` and `max_size` to `null`; both fields are supported by pinned Docling
`2.127.0`. PDFs retain the `default` preset. The `remote` transition alias uses
the same per-format preset selection. A missing `images` preset is an error,
never a fallback to `default`. One complete image page and nonempty text are required.

Before any conversion submission, a disposable helper checks dimensions and
single-frame decoding, applies EXIF orientation, composites transparency on white,
and rebuilds a clean RGB PNG. It strips metadata and trailing bytes. The private
reader, detector (when required) and downstream (when allowed) receive identical
decoded visible pixels at the normalized dimensions, not necessarily identical
PNG bytes: Docling's internal VLM backend may re-encode as opaque RGBA without
changing the visible pixels. The image preset must not resize them. The `none`
path skips detection.
PII Engine receives extracted text, not pixels. Forwarded pixels are inserted after
their scanned text while preserving all existing text and history order. Original
documents, embedded images and PDF page pixels are never forwarded.

Image limits are 5 MiB encoded-file bytes (or the smaller configured file limit),
4096 pixels per dimension and 12 million pixels for text-only or unchecked processing.
Protected images additionally require **at most 2,000,000 pixels and 64 through 2048
pixels per dimension**. The minimum avoids unreliable native detections for tiny
or extremely thin inputs. Protected images outside these bounds return HTTP 413 before
decoding/detection, never a detector-only resize. Real isolated-helper tests cover
the 2-million-pixel landscape/portrait boundary and skinny images in both
orientations under the unchanged resource cap. Python memory failures and OpenCV
`StsNoMem` allocation errors also map to HTTP 413; other detector errors fail closed.
Byte limits remain 5 MiB normalized PNG bytes per image
and 5 MiB total normalized data URIs per request. Existing batch count, aggregate
source/normalized byte and page limits also apply. The helper has 768 MiB address
space, ten CPU seconds and a fifteen-second parent deadline; timeout kills and
reaps it. Preflight checks a thread-safe cancellation event and monotonic batch
deadline before each file and native helper. Cancellation/deadline stops later
helpers; the active bounded helper finishes and is reaped before admission is
released. Existing native asynchronous-job draining and poisoning rules are unchanged.
Passthrough does not use this processing path or require a Docling client.
Final base64 plus text must fit `MAX_TRANSFORMED_REQUEST_BYTES`; no limits are raised.

The packaged MIT OpenCV YuNet `face_detection_yunet_2023mar.onnx` runs only on CPU,
with score threshold 0.5. Its SHA256 is
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
Immutable source and license are in `src/agentgateway_extproc/assets/`; distribution
checks verify model and license contents, and the Dockerfile verifies the installed
model and makes assets read-only. There are no runtime downloads, model mounts,
face identities, embeddings, storage, blur or image/face redaction. Missing/corrupt
models and detector failures fail protected image requests closed without affecting
ordinary text readiness. Native diagnostics and content are never logged.
Detection and OCR can miss content: these checks reduce risk, not prove that an
image contains no personal information. A real-model blank-image smoke test proves
load/inference only, not detection quality on every kind of photograph.

#### Version Three (0.10.0)

V3 accepts inline still JPEG, PNG and iPhone HEIC/HEIF images. Docling may use
`internal-standard` CPU OCR or the qualified `private-vlm` reader. CPU YuNet scans
whenever face protection is enabled, including `image_forwarding: none`. One
oriented, white-composited RGB PNG is shared by OCR, detection and allowed output:
sources are bounded to 20 MiB (or the smaller file limit), 50 million pixels and
10,000 pixels per dimension; the canonical image is at most 2 million pixels and
2048 per dimension, with a 64-pixel minimum per dimension when protected. The
5 MiB PNG and aggregate normalized-data-URI caps, helper limits and configured
request/batch/output limits still apply; phone uploads may need an explicit
transport-limit increase.

Current source also corrects the native Docling format from `img` to `image`
for both v2 and v3; published `0.9.0` still contains the incorrect format value.

The existing document-analysis endpoint receives a v1 envelope containing the
converted Chat/Responses `request`, trusted `text_pii_enabled` and aggregate
`visual_findings.faces`, never pixels. Its exact fresh findings echo is required.
Central FACE actions are `block`, `text-only` (withhold **all** request images) and
`reroute` (only with forwarding enabled and an exact approved local binding).
Text blocks win; real text PII findings still prohibit conditional remote pixels.
Successful extraction with no text uses `[Image: no text extracted]`, not a clean
PII claim: such images require an approved local reroute or unchecked local path;
text-only handling blocks them. Failed extraction or detection remains fatal.

V3 retains the old maps and adds `image_models` (explicit image support plus
concrete locality proof) and `image_reroutes` (`source -> route_class -> local
image destination`). Unchecked forwarding requires `image_models[source] == true`;
a face reroute requires its exact binding, with no fallback. Bindings do not grant
permissions: Gateway still authorizes each target. V1/v2 reject these new maps,
and older consumers must not receive v3 metadata. MCP remains v1 and explicit
passthrough is unchanged.

The existing PII Engine Notice/table reports FACE counts without claiming masking.
Face-only requests do not enable text scanning, guard injection or response
reversal, and omit `x-presidio-code` rather than claim clean text with `P00`.
Normal JSON/SSE notice placement and structured-output/MCP suppression remain.
Blocks stay HTTP 403 with a short `error.message`, a table when suitable and a
bounded `pii_report`. The unreleased no-text correction changes only this case:
when face policy selects `text-only` but an image has no
readable text, the 403 instead uses code `image_text_unavailable` and a short
plain-language message without the Markdown table. Its structured report retains
FACE `text-only` and the original counts, with `decision: block` and
`reason: no_readable_text`. Other image denials retain their existing reporting.
The pinned LibreChat error display is plain text and may truncate this message,
so it cannot reliably display the full Markdown table. This is not hidden by
returning a fake HTTP 200, and a zero face count never proves text PII is absent.

### Stateless MCP

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
| `INFERENCE_MODE` | `internal-standard` | `internal-standard` or `private-vlm`; transition aliases `cpu` or `remote` |
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
