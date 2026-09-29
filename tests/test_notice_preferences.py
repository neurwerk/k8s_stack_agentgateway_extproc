"""Exercise trusted preference lookup and notice-only response filtering."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agentgateway_extproc.config.settings import NoticePreferencesSettings
from agentgateway_extproc.lib.notice.preferences import (
    DEFAULT_PREFERENCES,
    NoticePreferences,
    NoticePreferencesClient,
)
from agentgateway_extproc.lib.notice.report import render_report
from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
from agentgateway_extproc.models.destination import ModelDestinationPolicy
from agentgateway_extproc.models.engine import AnalysisMetadata, PIIReport
from agentgateway_extproc.models.exceptions import TrustedMetadataError
from agentgateway_extproc.models.types import RequestStats

from .conftest import MODEL_POLICY, body_request, header_request, response_body, response_headers

STUDIO_SETTINGS = NoticePreferencesSettings(
    enabled=True, ca_cert="ca", client_cert="cert", client_key="key"
)


@pytest.mark.parametrize(
    "update",
    [
        {"credential_id": "key"},
        {"credential_kind": "personal"},
        {"credential_context_version": "1"},
        {"credential_context_version": "2", "credential_id": "key", "credential_kind": "personal"},
        {"credential_id": " ", "credential_kind": "personal"},
        {"credential_id": "x" * 257, "credential_kind": "personal"},
        {"credential_id": "key", "credential_kind": "unknown"},
    ],
)
async def test_invalid_credential_binding_is_rejected(engine_client, update) -> None:
    handler = StreamHandler(engine_client)
    with pytest.raises(TrustedMetadataError):
        await handler.handle(header_request(policy={**MODEL_POLICY, **update}))


async def test_credential_binding_cannot_change_mid_stream(engine_client) -> None:
    policy = {
        **MODEL_POLICY,
        "credential_context_version": "1",
        "credential_id": "key",
        "credential_kind": "personal",
    }
    handler = StreamHandler(engine_client)
    await handler.handle(header_request(policy=policy))
    with pytest.raises(TrustedMetadataError):
        await handler.handle(body_request(b"{}", policy=MODEL_POLICY))


async def test_lookup_uses_principal_and_only_personal_id_and_caches(engine_client) -> None:
    requests: list[dict[str, str]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(dict(request.url.params))
        return httpx.Response(200, json={**DEFAULT_PREFERENCES.model_dump(), "show_pass": False})

    client = NoticePreferencesClient(
        STUDIO_SETTINGS,
        httpx.AsyncClient(transport=httpx.MockTransport(reply), base_url="https://studio.test"),
    )
    personal = ModelDestinationPolicy.model_validate(
        {
            **MODEL_POLICY,
            "credential_context_version": "1",
            "credential_id": "key",
            "credential_kind": "personal",
        }
    )
    managed = ModelDestinationPolicy.model_validate(
        {
            **MODEL_POLICY,
            "credential_context_version": "1",
            "credential_id": "key",
            "credential_kind": "managed",
        }
    )
    handler = StreamHandler(engine_client, preferences_client=client)
    await handler.handle(
        header_request(
            policy={
                **MODEL_POLICY,
                "credential_context_version": "1",
                "credential_id": "key",
                "credential_kind": "personal",
            }
        )
    )
    assert handler.notice_preferences.show_pass is False
    assert await client.get(personal) == handler.notice_preferences
    await client.get(managed)
    await client.get(ModelDestinationPolicy.model_validate(MODEL_POLICY))
    assert requests == [
        {"principal_id": personal.principal_id, "credential_id": "key"},
        {"principal_id": personal.principal_id},
    ]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503),
        httpx.Response(200, json={"show_pass": False}),
        httpx.Response(200, json={**DEFAULT_PREFERENCES.model_dump(), "show_pass": "false"}),
    ],
)
async def test_missing_or_invalid_studio_response_defaults_all_on(response) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response), base_url="https://studio.test"
    ) as http:
        client = NoticePreferencesClient(STUDIO_SETTINGS, http)
        assert (
            await client.get(ModelDestinationPolicy.model_validate(MODEL_POLICY))
            == DEFAULT_PREFERENCES
        )


async def test_slow_studio_falls_back_within_the_short_deadline() -> None:
    async def slow(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, json=DEFAULT_PREFERENCES.model_dump())

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(slow), base_url="https://studio.test"
    ) as http:
        client = NoticePreferencesClient(STUDIO_SETTINGS.model_copy(update={"timeout": 0.01}), http)
        assert (
            await client.get(ModelDestinationPolicy.model_validate(MODEL_POLICY))
            == DEFAULT_PREFERENCES
        )


def test_mixed_actions_keep_visible_rows_and_face_safety() -> None:
    report = PIIReport.model_validate(
        {
            "rows": [
                {
                    "entity_type": "FACE",
                    "action": "text-only",
                    "detected_count": 1,
                    "transformed_count": 0,
                    "unique_transformed_count": 0,
                },
                {
                    "entity_type": "NAME",
                    "action": "pass",
                    "detected_count": 1,
                    "transformed_count": 0,
                    "unique_transformed_count": 0,
                },
                {
                    "entity_type": "PHONE",
                    "action": "mask",
                    "detected_count": 1,
                    "transformed_count": 1,
                    "unique_transformed_count": 1,
                },
            ]
        }
    )
    analysis = AnalysisMetadata.model_validate(
        {
            "source": "current_request",
            "scan_performed": True,
            "duration_ms": 100,
            "overlap_count": 0,
            "overlap_resolution": "strictest_action",
            "policy_version": "test",
            "text_leaf_count": 1,
            "cached_decision_applied": False,
        }
    )
    result = render_report(
        report,
        analysis,
        {"PHONE": 1},
        decision="apply_actions",
        route_class=None,
        preferences=NoticePreferences(show_pass=False, show_changes=False, show_timing=False),
    )
    assert "| Face |" in result
    assert "| Name |" not in result and "| Phone |" not in result
    assert "PII scan completed" not in result


@pytest.mark.parametrize(
    ("action", "disabled", "hidden"),
    [
        (None, "show_no_pii", "No sensitive"),
        ("pass", "show_pass", "| Person |"),
        ("mask", "show_changes", "| Person |"),
        ("reroute", "show_reroutes", "Effective route"),
    ],
)
async def test_result_categories_hide_only_their_notices(
    engine_client, action, disabled, hidden
) -> None:
    handler = StreamHandler(engine_client)
    handler.notice_preferences = NoticePreferences.model_validate({disabled: False})
    handler.notice_messages = ["No sensitive data was detected." if action is None else "Protected"]
    handler.request_stats = RequestStats(
        report=PIIReport.model_validate(
            {
                "rows": []
                if action is None
                else [
                    {
                        "entity_type": "PERSON",
                        "action": action,
                        "detected_count": 1,
                        "transformed_count": int(action == "mask"),
                        "unique_transformed_count": int(action == "mask"),
                    }
                ]
            }
        ),
        analysis=AnalysisMetadata.model_validate(
            {
                "source": "current_request",
                "scan_performed": True,
                "duration_ms": 100,
                "overlap_count": 0,
                "overlap_resolution": "strictest_action",
                "policy_version": "test",
                "text_leaf_count": 1,
                "cached_decision_applied": False,
            }
        ),
        decision="reroute"
        if action == "reroute"
        else "apply_actions"
        if action == "mask"
        else "pass",
        route_class="local" if action == "reroute" else None,
    )
    await handler.handle(response_headers("application/json"))
    response = await handler.handle(
        response_body(b'{"choices":[{"index":0,"message":{"content":"answer"}}]}')
    )
    assert response is not None
    content = json.loads(response.response_body.response.body_mutation.streamed_response.body)[
        "choices"
    ][0]["message"]["content"]
    assert hidden not in content
    assert "PII scan completed" in content  # timing is independent of the result
    assert content.startswith("answer")


@pytest.mark.parametrize("content_type", ["text/plain", "text/event-stream", "application/json"])
async def test_no_empty_footer_when_every_notice_is_off(engine_client, content_type) -> None:
    handler = StreamHandler(engine_client)
    handler.notice_preferences = NoticePreferences(
        show_no_pii=False,
        show_pass=False,
        show_changes=False,
        show_reroutes=False,
        show_timing=False,
    )
    handler.notice_messages = ["No sensitive data was detected."]
    handler.request_stats = RequestStats(
        report=PIIReport(rows=[]),
        analysis=AnalysisMetadata.model_validate(
            {
                "source": "current_request",
                "scan_performed": True,
                "duration_ms": 100,
                "overlap_count": 0,
                "overlap_resolution": "strictest_action",
                "policy_version": "test",
                "text_leaf_count": 1,
                "cached_decision_applied": False,
            }
        ),
    )
    await handler.handle(response_headers(content_type))
    body = (
        b'{"choices":[{"index":0,"message":{"content":"answer"}}]}'
        if content_type == "application/json"
        else b'data: {"choices":[{"index":0,"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n'
        if content_type == "text/event-stream"
        else b"answer"
    )
    response = await handler.handle(response_body(body))
    assert response is not None
    output = response.response_body.response.body_mutation.streamed_response.body.decode()
    assert "PII Engine Notice" not in output and "---" not in output
    assert "answer" in output
