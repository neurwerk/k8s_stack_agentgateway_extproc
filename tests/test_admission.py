"""Exercise admission, credential isolation and a single canonical dispatch decision."""

import json
import operator
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from google.protobuf.json_format import MessageToDict
from pydantic import ValidationError

from agentgateway_extproc.config.settings import (
    EngineSettings,
    NoticePreferencesSettings,
    Settings,
)
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.notice.preferences import DEFAULT_PREFERENCES, NoticePreferencesClient
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.admission import AdmissionEnvelope, metadata_from_request
from agentgateway_extproc.models.exceptions import GatewayPolicyError, TrustedMetadataError

from .conftest import (
    MODEL_POLICY,
    REVERSIBLE_TOKEN,
    add_policy,
    body_request,
    header_request,
    request_json,
    response_body,
    response_headers,
    segment_reply,
)
from .openai_fixtures import event_bytes, response_payload, stream_events
from .test_fail_closed import FakeAioContext, StreamAbortedError

SECRET = "credential-sentinel"


@pytest.fixture
def admission():
    return json.loads((Path(__file__).parent / "fixtures/admission-v5.json").read_text())


def headers(**changes):
    return {
        ":method": "POST",
        ":path": "/bindings/openai-api/v1/chat/completions",
        "content-type": "application/json",
        "authorization": f"Bearer {SECRET}",
    } | changes


def body(model="example-chat", *, responses=False, stream=False):
    return json.dumps(
        {"model": model, "stream": stream}
        | (
            {"input": "Jane Doe"}
            if responses
            else {"messages": [{"role": "user", "content": "Jane Doe"}]}
        )
    ).encode()


async def exchange(client, messages, *, settings=None):
    async def requests():
        for message in messages:
            yield message

    return [
        reply async for reply in ExtProcServicer(client, settings).Process(requests(), object())
    ]


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("requested", ["example-chat", "provider-chat"])
@pytest.mark.parametrize("pii", [True, False])
async def test_one_resolution_drives_permissions_policy_and_final_provider_model(
    admission, engine_reply, responses, requested, pii
):
    engine_reply["route_class"] = "general"
    if responses:
        admission["request_context"]["protocol_profile"] = "openai-responses-v1"
    admission["destination_policy"]["models"]["example-chat"] = pii
    calls = []

    def analyze(request):
        calls.append(json.loads(request.content))
        assert SECRET not in str(request.headers) + request.content.decode()
        return httpx.Response(200, json=segment_reply(engine_reply))

    async with httpx.AsyncClient(transport=httpx.MockTransport(analyze)) as http:
        client = EngineClient(EngineSettings(base_url="https://engine.example.test"), http)
        handler = StreamHandler(client)
        original = headers(
            **{
                ":path": "/bindings/openai-api/v1/"
                + ("responses" if responses else "chat/completions")
            }
        )
        header_reply = await handler.handle(header_request(original, policy=admission))
        assert header_reply is not None
        assert (
            "authorization"
            not in header_reply.request_headers.response.header_mutation.remove_headers
        )
        assert SECRET not in repr(vars(handler))
        result = await handler.handle(
            body_request(body(requested, responses=responses), policy=admission)
        )
        assert result is not None
        mutation = result.request_body.response.body_mutation
        forwarded = json.loads(mutation.body or body(requested, responses=responses))
        assert forwarded["model"] == "example-chat"
        context = handler.request_context
        assert context is not None
        assert context.destination_id == "example-chat"
        assert context.requested_model == requested
        assert context.resolved_model == "provider-chat"
        metadata = MessageToDict(result.dynamic_metadata)["neurwerk.request_context"]
        assert metadata == context.model_dump(mode="json")
        assert metadata["billing_classification"] == "unknown"
        assert metadata["usage_completeness"] == "unknown"
        assert metadata["request_text_inspection"] == ("scanned" if pii else "disabled")
        assert metadata["target_credential_mode"] == "caller_supplied"
        if not responses and requested != "provider-chat" and pii:
            expected = json.loads(
                (Path(__file__).parent / "fixtures/resolved-context-v5.json").read_text()
            )
            assert MessageToDict(result.dynamic_metadata) == expected
        assert (context.binding_id, context.provider_id) == ("openai-api", "openai")
        assert context.pii_policy.enabled is pii
        assert context.content_tracing_enabled is False
        assert len(calls) == int(pii)
        assert handler.admission is not None
        assert (
            handler.admission.request_context.model_dump(mode="json")
            == admission["request_context"]
        )
        with pytest.raises(ValidationError):
            context.resolved_model = "another-provider-model"  # ty: ignore[invalid-assignment]
        response = json.dumps(response_payload("responses" if responses else "chat")).encode()
        await handler.handle(response_headers("application/json", policy=admission))
        reply = await handler.handle(response_body(response, policy=admission))
        assert reply is not None
        restored = reply.response_body.response.body_mutation.streamed_response.body
        assert (b"Jane Doe" in restored) is pii
        assert handler.reversal_map == {}


@pytest.mark.parametrize(
    "changed",
    [
        "principal",
        "credential",
        "kind",
        "coverage",
        "collision",
        "canonical_collision",
        "permissions",
        "mode",
        "provider",
        "profile",
        "unpaired",
    ],
)
async def test_invalid_admission_fails_before_engine(admission, engine_client, changed):  # noqa: C901
    context = admission["request_context"]
    if changed == "principal":
        context["principal_id"] = "another-subject"
    elif changed == "credential":
        context.update(
            credential_context_version="1", credential_id="personal-1", credential_kind="personal"
        )
        admission["destination_policy"].update(
            credential_context_version="1", credential_id="personal-2", credential_kind="personal"
        )
    elif changed == "kind":
        context["destination_kind"] = "application"
    elif changed == "coverage":
        admission["model_bindings"] = {}
    elif changed in {"collision", "canonical_collision"}:
        admission["destination_policy"] = {
            **MODEL_POLICY,
            "principal_id": context["principal_id"],
            "models": {"example-chat": True, "other": False},
        }
        admission["model_bindings"]["other"] = {
            "upstream_model": "provider-chat" if changed == "collision" else "example-chat",
            "required_permissions": ["llm:invoke", "model:other:invoke"],
            "content_tracing_enabled": False,
        }
    elif changed == "permissions":
        context["permissions"].append("llm:invoke")
    elif changed == "mode":
        context["provider_credential_source"] = "caller"
    elif changed == "provider":
        context["provider_id"] = None
    elif changed == "profile":
        context["protocol_profile"] = "caller-selected-code"
    elif changed == "unpaired":
        context["credential_id"] = "personal-1"
    with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
        replies = await exchange(
            engine_client,
            [header_request(headers(), policy=admission), body_request(body(), policy=admission)],
        )
    analyze.assert_not_awaited()
    assert replies[-1].immediate_response.status.code == 503


@pytest.mark.parametrize(
    "field,value",
    [
        ("binding_id", "other-binding"),
        ("provider_id", "other-provider"),
        ("permissions", ["llm:invoke"]),
        ("policy_revision", "other-revision"),
        ("provider_credential_source", "company"),
    ],
)
async def test_envelope_is_locked_including_binding_and_permissions(
    admission, engine_client, field, value
):
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(headers(), policy=admission))
    changed = deepcopy(admission)
    changed["request_context"][field] = value
    with pytest.raises(TrustedMetadataError):
        await handler.handle(body_request(body(), policy=changed))


@pytest.mark.parametrize(
    "model,permissions,status",
    [
        ("model-on-another-binding", ["llm:invoke", "model:model-on-another-binding:invoke"], 400),
        ("provider-chat", ["llm:invoke"], 403),
    ],
)
async def test_selection_outside_binding_or_grant_never_reaches_analysis(
    admission, engine_client, model, permissions, status
):
    admission["request_context"]["permissions"] = permissions
    with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
        replies = await exchange(
            engine_client,
            [
                header_request(headers(), policy=admission),
                body_request(body(model), policy=admission),
            ],
        )
    analyze.assert_not_awaited()
    assert replies[-1].immediate_response.status.code == status


@pytest.mark.parametrize(
    "source,grant,status", [("company", False, 403), ("company", True, None), ("none", False, None)]
)
async def test_company_use_requires_extra_permission_and_strips_credentials(
    admission, engine_client, source, grant, status
):
    context = admission["request_context"]
    context["provider_credential_source"] = source
    if source == "none":
        context["provider_id"] = None
    if grant:
        context["permissions"].append("provider:openai:company-use")
    original = headers()
    if source == "none":
        del original["authorization"]
    handler = StreamHandler(engine_client)
    reply = await handler.handle(header_request(original, policy=admission))
    assert reply is not None
    if source == "company":
        assert "authorization" in reply.request_headers.response.header_mutation.remove_headers
    assert SECRET not in repr(vars(handler))
    if status:
        with pytest.raises(GatewayPolicyError) as rejected:
            await handler.handle(body_request(body(), policy=admission))
        assert rejected.value.status_code == status
    else:
        result = await handler.handle(body_request(body(), policy=admission))
        assert result is not None
        assert result.HasField("request_body")
        context = MessageToDict(result.dynamic_metadata)["neurwerk.request_context"]
        assert context["billing_classification"] == (
            "company_api" if source == "company" else "unknown"
        )
        assert context["usage_completeness"] == "unknown"
        assert context["target_credential_mode"] == source


@pytest.mark.parametrize(
    "field,value",
    [
        (":method", "GET"),
        (":path", "/v1/chat/completions?key=credential-sentinel"),
        ("content-type", "multipart/form-data"),
        ("content-encoding", "gzip"),
        ("x-api-key", SECRET),
        ("x-goog-api-key", SECRET),
        ("cookie", SECRET),
        ("authorization", "Basic credential-sentinel"),
    ],
)
async def test_wrong_transport_or_credential_location_is_rejected_without_retention(
    admission, engine_client, field, value
):
    handler = StreamHandler(engine_client)
    with pytest.raises(GatewayPolicyError):
        await handler.handle(header_request(headers(**{field: value}), policy=admission))
    assert handler.request_headers == {}
    assert SECRET not in repr(vars(handler))


async def test_sensitive_headers_are_preserved_or_removed_without_value_copies(
    admission, engine_client
):
    original = headers(
        **{
            "x-neurwerk-authorization": f"Bearer {SECRET}",
            "proxy-authorization": SECRET,
            "x-agentgateway-auth-context": SECRET,
        }
    )
    event = header_request(original, policy=admission)
    handler = StreamHandler(engine_client)
    reply = await handler.handle(event)
    assert reply is not None
    mutation = reply.request_headers.response.header_mutation
    upstream = {
        name: value for name, value in original.items() if name not in mutation.remove_headers
    }
    assert upstream["authorization"] == original["authorization"]
    assert set(mutation.remove_headers) == {
        "x-neurwerk-authorization",
        "proxy-authorization",
        "x-agentgateway-auth-context",
    }
    assert not mutation.set_headers
    assert SECRET not in repr(vars(handler)) + str(reply)
    await handler.handle(body_request(body(stream=True), policy=admission))
    response = await handler.handle(
        response_headers(
            "text/event-stream",
            extra_headers={"set-cookie": SECRET, "authorization": SECRET},
            policy=admission,
        )
    )
    assert response is not None
    assert {"set-cookie", "authorization"} <= set(
        response.response_headers.response.header_mutation.remove_headers
    )
    assert SECRET not in repr(vars(handler)) + str(response)


@pytest.mark.parametrize("status", [401, 403, 429])
async def test_provider_rejection_keeps_status_without_consuming_or_echoing_credentials(
    admission, engine_client, caplog, status
):
    async def requests():
        yield header_request(headers(), policy=admission)
        yield body_request(body(), policy=admission)
        yield response_headers(
            "application/json",
            status=status,
            extra_headers={"set-cookie": SECRET, "www-authenticate": SECRET},
            policy=admission,
        )
        pytest.fail("provider error body must not be consumed")

    replies = [
        reply async for reply in ExtProcServicer(engine_client).Process(requests(), object())
    ]
    assert replies[-1].immediate_response.status.code == status
    assert SECRET not in str(replies) + caplog.text


@pytest.mark.parametrize(
    "reason", ["limit", "invalid_json", "permission", "changed_metadata", "overload"]
)
async def test_v5_rejections_do_not_disclose_content(admission, engine_client, caplog, reason):
    settings = Settings(max_request_bytes=1024)
    servicer = ExtProcServicer(engine_client, settings)
    payload = body()
    body_policy = deepcopy(admission)
    if reason == "limit":
        payload = SECRET.encode() * 200
    elif reason == "invalid_json":
        payload = SECRET.encode()
    elif reason == "permission":
        admission["request_context"]["permissions"] = []
        body_policy = deepcopy(admission)
    elif reason == "changed_metadata":
        body_policy = MODEL_POLICY
    elif reason == "overload":
        servicer._buffered_bytes = 4 * settings.max_request_bytes

    async def requests():
        yield header_request(headers(), policy=admission)
        yield body_request(payload, policy=body_policy)

    replies = [reply async for reply in servicer.Process(requests(), object())]
    assert replies[-1].HasField("immediate_response")
    assert SECRET not in str(replies) + caplog.text


@pytest.mark.parametrize(
    "change",
    [
        "profile",
        "operation",
        "reasoning",
        "partial_body",
        "background",
        "history",
        "raw_attachment",
    ],
)
async def test_unqualified_features_stop_before_analysis(admission, engine_client, change):
    payload = body()
    request_headers = headers()
    if change == "profile":
        admission["request_context"]["protocol_profile"] = "chatgpt-web-v1"
    elif change == "operation":
        admission["request_context"]["operation_id"] = "history"
    elif change == "reasoning":
        data = json.loads(payload)
        data["messages"].append(
            {
                "role": "assistant",
                "content": "text",
                "reasoning_details": [{"type": "unqualified", "text": SECRET}],
            }
        )
        payload = json.dumps(data).encode()
    elif change in {"background", "history"}:
        admission["request_context"]["protocol_profile"] = "openai-responses-v1"
        request_headers[":path"] = "/bindings/openai-api/v1/responses"
        data = json.loads(body(responses=True))
        data["background" if change == "background" else "previous_response_id"] = (
            True if change == "background" else "provider-history"
        )
        payload = json.dumps(data).encode()
    elif change == "raw_attachment":
        admission["destination_policy"] = {
            **MODEL_POLICY,
            "principal_id": "example-subject",
            "models": {"example-chat": False},
            "attachment_modes": {"example-chat": "passthrough"},
        }
        payload = json.dumps(
            {
                "model": "example-chat",
                "messages": [
                    {"role": "user", "content": [{"type": "input_audio", "data": SECRET}]}
                ],
            }
        ).encode()
    with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
        replies = await exchange(
            engine_client,
            [
                header_request(request_headers, policy=admission),
                body_request(payload, policy=admission, end_of_stream=change != "partial_body"),
            ],
        )
    assert replies[-1].immediate_response.status.code == 400
    analyze.assert_not_awaited()


@pytest.mark.parametrize("cached", [False, True])
async def test_reroute_requires_company_source_and_explicit_local_permission(
    engine_client, engine_reply, admission, cached
):
    engine_reply.update(
        decision="reroute",
        remote_allowed=False,
        route_class="local",
        applied_actions=["reroute"],
        reversal={},
        request=json.loads(request_json()),
        report={
            "rows": [
                {
                    "entity_type": "PERSON",
                    "action": "reroute",
                    "detected_count": 1,
                    "transformed_count": 0,
                    "unique_transformed_count": 0,
                }
            ]
        },
    )
    if cached:
        engine_reply["analysis"].update(
            source="cached_decision",
            scan_performed=False,
            duration_ms=None,
            cached_decision_applied=True,
        )
    replies = await exchange(
        engine_client,
        [header_request(headers(), policy=admission), body_request(body(), policy=admission)],
    )
    assert replies[-1].immediate_response.status.code == 403
    assert "unsupported_reroute" in replies[-1].immediate_response.body
    admission["model_bindings"]["example-chat"]["allowed_reroutes"] = ["local"]
    admission["reroute_targets"] = {
        "local": {
            "binding_id": "local-binding",
            "upstream_model": "local-provider-model",
            "required_permissions": ["llm:invoke", "model:local:invoke"],
            "content_tracing_enabled": True,
        }
    }
    admission["request_context"]["provider_credential_source"] = "company"
    admission["request_context"]["permissions"].append("provider:openai:company-use")
    for allowed in (False, True):
        if allowed:
            admission["request_context"]["permissions"].append("model:local:invoke")
        replies = await exchange(
            engine_client,
            [header_request(headers(), policy=admission), body_request(body(), policy=admission)],
        )
        if not allowed:
            assert replies[-1].immediate_response.status.code == 403
            assert "permission_denied" in replies[-1].immediate_response.body
        else:
            assert replies[-1].HasField("request_body")
            context = MessageToDict(replies[-1].dynamic_metadata)["neurwerk.request_context"]
            assert context["dispatch_binding_id"] == "local-binding"
            assert context["dispatch_destination_id"] == "local"
            assert context["resolved_model"] == "local-provider-model"
            assert context["destination_id"] == "example-chat"
            assert context["provider_credential_source"] == "company"
            assert context["target_credential_mode"] == "none"
            assert context["request_text_inspection"] == (
                "cached_decision" if cached else "scanned"
            )
            assert context["content_tracing_enabled"] is False
            assert context["billing_classification"] == "unknown"


def test_legacy_schema_does_not_accept_v5_fields(admission):
    with pytest.raises(TrustedMetadataError):
        metadata_from_request(
            header_request(policy=MODEL_POLICY | {"request_context": admission["request_context"]})
        )
    assert isinstance(metadata_from_request(header_request(policy=admission)), AdmissionEnvelope)


def test_admission_nested_policy_and_bindings_cannot_be_modified(admission):
    admission["destination_policy"]["image_reroutes"] = {"example-chat": {"local": "local-chat"}}
    envelope = AdmissionEnvelope.model_validate(admission)
    policy = envelope.destination_policy
    assert policy.destination_kind == "model"
    for mapping, key, value in [
        (policy.models, "example-chat", False),
        (policy.image_reroutes["example-chat"], "local", "other-model"),
        (policy.attachment_modes, "example-chat", "passthrough"),
        (envelope.model_bindings, "example-chat", None),
    ]:
        with pytest.raises(TypeError):
            operator.setitem(mapping, key, value)  # ty: ignore[no-matching-overload]
    assert envelope.model_dump(mode="json", exclude_unset=True) == admission


@pytest.mark.parametrize("changed", ["upstream", "pii", "trace", "explicit_default"])
async def test_complete_wire_envelope_stays_locked_after_resolution(
    admission, engine_client, changed
):
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(headers(), policy=admission))
    envelope = handler.admission
    await handler.handle(body_request(body(), policy=admission))
    assert handler.admission is envelope
    if changed == "upstream":
        admission["model_bindings"]["example-chat"]["upstream_model"] = "other-model"
    elif changed == "pii":
        admission["destination_policy"]["models"]["example-chat"] = False
    elif changed == "trace":
        admission["model_bindings"]["example-chat"]["content_tracing_enabled"] = True
    else:
        admission["destination_policy"]["account_email"] = None
    with pytest.raises(TrustedMetadataError):
        await handler.handle(response_headers("application/json", policy=admission))
    assert handler.admission is envelope


@pytest.mark.parametrize(
    "requested", ["example-chat", "provider-chat", "other-chat", "other-provider-chat"]
)
async def test_two_model_selection_never_mixes_pii_tracing_grants_or_upstream(
    admission, engine_client, requested
):
    admission["destination_policy"] = {
        **MODEL_POLICY,
        "principal_id": "example-subject",
        "models": {"example-chat": True, "other-chat": False},
    }
    admission["model_bindings"]["other-chat"] = {
        "upstream_model": "other-provider-chat",
        "required_permissions": ["llm:invoke", "model:other-chat:invoke", "extra:invoke"],
        "content_tracing_enabled": True,
    }
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(headers(), policy=admission))
    if "other" in requested:
        with pytest.raises(GatewayPolicyError) as denied:
            await handler.handle(body_request(body(requested), policy=admission))
        assert denied.value.code == "permission_denied"
    admission["request_context"]["permissions"] += ["model:other-chat:invoke", "extra:invoke"]
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(headers(), policy=admission))
    result = await handler.handle(body_request(body(requested), policy=admission))
    assert result is not None
    context = handler.request_context
    assert context is not None
    other = "other" in requested
    assert context.destination_id == ("other-chat" if other else "example-chat")
    assert context.pii_policy.enabled is not other
    assert context.content_tracing_enabled is other
    expected = "other-provider-chat" if other else "provider-chat"
    forwarded = result.request_body.response.body_mutation.body or body(requested)
    assert json.loads(forwarded)["model"] == context.destination_id
    assert context.resolved_model == expected


async def test_attachment_conversion_uses_the_same_alias_selection(admission, engine_client):
    admission["destination_policy"]["document_modes"]["example-chat"] = "extract-text"
    admission["destination_policy"]["face_protection"]["example-chat"] = True
    docling = Mock(convert=AsyncMock(return_value=["Jane Doe"]))
    handler = StreamHandler(engine_client, docling=docling)
    await handler.handle(header_request(headers(), policy=admission))
    payload = {
        "model": "provider-chat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "file", "file": {"file_data": "data:text/plain;base64,SmFuZSBEb2U="}}
                ],
            }
        ],
    }
    reply = await handler.handle(body_request(json.dumps(payload).encode(), policy=admission))
    assert reply is not None and reply.HasField("request_body")
    docling.convert.assert_awaited_once()
    forwarded = json.loads(reply.request_body.response.body_mutation.body)
    assert forwarded["model"] == "example-chat"
    assert SECRET not in str(docling.convert.call_args)
    assert "Jane Doe" not in str(forwarded)
    assert REVERSIBLE_TOKEN in str(forwarded)


@pytest.mark.parametrize("pii", [False, True])
async def test_response_trailers_strip_credentials_and_empty_eos_keeps_compatibility(
    admission, engine_client, pii
):
    admission["destination_policy"]["models"]["example-chat"] = pii
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(headers(), policy=admission))
    await handler.handle(body_request(body(stream=True), policy=admission))
    await handler.handle(response_headers("text/event-stream", policy=admission))
    frames = b"".join(event_bytes("chat", event) for event in stream_events("chat"))
    await handler.handle(response_body(frames, end_of_stream=False, policy=admission))
    reply = await handler.handle(response_body(b"", policy=None))
    assert reply is not None and reply.HasField("response_body")
    trailers = add_policy(
        ext_proc_pb2.ProcessingRequest(
            response_trailers={
                "trailers": {
                    "headers": [
                        {"key": name, "raw_value": SECRET.encode()}
                        for name in ["set-cookie", "x-goog-api-key", "x-neurwerk-authorization"]
                    ]
                }
            }
        ),
        admission,
    )
    result = await handler.handle(trailers)
    assert result is not None
    assert set(result.response_trailers.header_mutation.remove_headers) == {
        "set-cookie",
        "x-goog-api-key",
        "x-neurwerk-authorization",
    }
    assert SECRET not in repr(vars(handler)) + str(result)


@pytest.mark.parametrize("pii", [False, True])
@pytest.mark.parametrize("sse", [False, True])
async def test_in_band_provider_error_never_echoes_error_content(
    admission, engine_client, caplog, pii, sse
):
    admission["destination_policy"]["models"]["example-chat"] = pii
    payload = json.dumps({"error": {"message": SECRET}}).encode()
    context = FakeAioContext()
    replies = []

    async def requests():
        yield header_request(headers(), policy=admission)
        yield body_request(body(stream=sse), policy=admission)
        yield response_headers("text/event-stream" if sse else "application/json", policy=admission)
        yield response_body(b"data: " + payload + b"\n\n" if sse else payload, policy=admission)

    async def collect():
        async for reply in ExtProcServicer(engine_client).Process(requests(), context):
            replies.append(reply)

    if sse:
        with pytest.raises(StreamAbortedError):
            await collect()
    else:
        await collect()
        assert replies[-1].immediate_response.status.code == 503
    assert SECRET not in str(replies) + caplog.text + context.details
    # A later rejection must not overwrite the delivered request facts with success.
    assert all(
        not reply.dynamic_metadata.fields for reply in replies if not reply.HasField("request_body")
    )


@pytest.mark.parametrize(
    "invalid", ["missing", "duplicate", "conflicting_encodings", "no_credential_mode", "length"]
)
async def test_ambiguous_or_missing_transport_is_rejected(admission, engine_client, invalid):
    event = header_request(headers(), policy=admission)
    if invalid == "missing":
        event.request_headers.headers.headers.pop()
    elif invalid == "duplicate":
        event.request_headers.headers.headers.add(
            key="Authorization", raw_value=f"Bearer {SECRET}".encode()
        )
    elif invalid == "conflicting_encodings":
        event.request_headers.headers.headers[-1].value = "Bearer another-credential"
    elif invalid == "no_credential_mode":
        admission["request_context"]["provider_credential_source"] = "none"
        event = header_request(headers(), policy=admission)
    else:
        event = header_request(headers(**{"content-length": "99999"}), policy=admission)
    with patch.object(engine_client, "analyze_request", new_callable=AsyncMock) as analyze:
        replies = await exchange(engine_client, [event, body_request(body(), policy=admission)])
    analyze.assert_not_awaited()
    assert replies[-1].immediate_response.status.code in {400, 401}
    assert SECRET not in str(replies)


@pytest.mark.parametrize("key_kind", ["personal", "managed"])
async def test_v5_notice_identity_uses_neurwerk_key_and_existing_cache(
    admission, engine_client, key_kind
):
    admission["request_context"].update(
        credential_context_version="1",
        credential_id="example-key",
        credential_kind=key_kind,
    )
    calls = []

    def lookup(request):
        calls.append(dict(request.url.params))
        assert SECRET not in str(request.url) + str(request.headers)
        return httpx.Response(200, json=DEFAULT_PREFERENCES.model_dump())

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lookup), base_url="https://studio.example.test"
    ) as http:
        preferences = NoticePreferencesClient(
            NoticePreferencesSettings(
                enabled=True, ca_cert="ca", client_cert="cert", client_key="key"
            ),
            http,
        )
        for _ in range(2):
            handler = StreamHandler(engine_client, preferences_client=preferences)
            await handler.handle(header_request(headers(), policy=admission))
        expected = {"principal_id": "example-subject"}
        if key_kind == "personal":
            expected["credential_id"] = "example-key"
        assert calls == [expected]


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize(
    "mode,prefix,accepted",
    [
        ("api", "/bindings/openai-api", True),
        ("api", "", False),
        ("api", "/bindings/another-binding", False),
        ("forward_proxy", "/bindings/openai-api", False),
        ("forward_proxy", "", True),
        ("api", "/arbitrary", False),
        ("api", "/bindings/openai-api/..", False),
        ("api", "/bindings/openai-api%2f", False),
    ],
)
async def test_api_mount_uses_only_verified_binding_for_validation(
    admission, engine_client, engine_reply, responses, mode, prefix, accepted
):
    # Ordinary Engine passes classify content without requesting a reroute.
    engine_reply.update(
        decision="pass",
        entities=[],
        entity_counts={},
        applied_actions=[],
        route_class="general",
        request=json.loads(request_json()),
        report={"rows": []},
        reversal={},
    )
    admission["request_context"]["entry_mode"] = mode
    if responses:
        admission["request_context"]["protocol_profile"] = "openai-responses-v1"
    path = prefix + ("/v1/responses" if responses else "/v1/chat/completions")
    handler = StreamHandler(engine_client)
    event = header_request(
        headers(**{":path": path, "x-forwarded-prefix": prefix}), policy=admission
    )
    with patch.object(
        engine_client, "analyze_request", wraps=engine_client.analyze_request
    ) as analyze:
        if not accepted:
            with pytest.raises(GatewayPolicyError):
                await handler.handle(event)
            analyze.assert_not_awaited()
            assert handler.request_headers == {}
            return
        reply = await handler.handle(event)
        assert reply is not None and not reply.request_headers.response.header_mutation.set_headers
        assert handler.request_headers[":path"] == path
        reply = await handler.handle(
            body_request(body("provider-chat", responses=responses), policy=admission)
        )
        assert reply is not None and reply.HasField("request_body")
        assert not any(
            h.header.key == ":path" for h in reply.request_body.response.header_mutation.set_headers
        )
        forwarded = json.loads(reply.request_body.response.body_mutation.body)
        assert forwarded["model"] == "example-chat"
        assert handler.request_context is not None
        assert handler.request_context.resolved_model == "provider-chat"
        analyze.assert_awaited_once()


@pytest.mark.parametrize("binding", [".", "..", "a/..", "a\\b", "a%2fb", "a%5cb", "%2e%2e"])
async def test_binding_id_cannot_be_an_ambiguous_path_segment(admission, engine_client, binding):
    admission["request_context"]["binding_id"] = binding
    handler = StreamHandler(engine_client)
    with pytest.raises(TrustedMetadataError):
        await handler.handle(
            header_request(
                headers(**{":path": f"/bindings/{binding}/v1/chat/completions"}), policy=admission
            )
        )
    assert handler.request_headers == {}
