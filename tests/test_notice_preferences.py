"""Exercise trusted preference lookup and notice-only response filtering."""

from __future__ import annotations

import asyncio

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
from agentgateway_extproc.models.engine import AnalysisMetadata, PIIReport, VisualFindings
from agentgateway_extproc.models.exceptions import TrustedMetadataError

from .conftest import MODEL_POLICY, body_request, header_request

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


async def test_legacy_reply_inherits_new_flags_and_invalid_versions_fall_back() -> None:
    legacy = dict.fromkeys(
        ("show_no_pii", "show_pass", "show_changes", "show_reroutes", "show_timing"), True
    )
    legacy["show_pass"] = False
    policy = ModelDestinationPolicy.model_validate(MODEL_POLICY)
    for payload, expected in [
        (legacy, NoticePreferences(show_pass=False)),
        ({**legacy, "show_detected_faces": False}, DEFAULT_PREFERENCES),
        ({**DEFAULT_PREFERENCES.model_dump(), "show_pass": "false"}, DEFAULT_PREFERENCES),
        ({**DEFAULT_PREFERENCES.model_dump(), "unknown": True}, DEFAULT_PREFERENCES),
    ]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _, value=payload: httpx.Response(200, json=value)),
            base_url="https://studio.test",
        ) as http:
            assert await NoticePreferencesClient(STUDIO_SETTINGS, http).get(policy) == expected


def _analysis() -> AnalysisMetadata:
    return AnalysisMetadata.model_validate(
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


@pytest.mark.parametrize(
    ("status", "count", "flag", "text"),
    [
        ("complete", 0, "show_no_faces", "Face scan completed: 0 detected."),
        ("complete", 2, "show_detected_faces", "Face scan completed: 2 detected."),
        ("not_scanned", None, "show_unscanned_faces", "Faces were not scanned."),
    ],
)
def test_face_status_categories_are_independent(status, count, flag, text) -> None:
    findings = VisualFindings.model_validate({"faces": {"scan_status": status, "count": count}})

    def result(preferences: NoticePreferences) -> str:
        return render_report(
            PIIReport(rows=[]),
            _analysis(),
            {},
            decision="pass",
            route_class=None,
            visual_findings=findings,
            preferences=preferences,
        )

    assert text in result(NoticePreferences(show_no_pii=False, show_timing=False))
    assert text not in result(NoticePreferences.model_validate({flag: False, "show_timing": False}))


@pytest.mark.parametrize(
    "action,flag", [("text-only", "show_changes"), ("reroute", "show_reroutes")]
)
def test_face_rows_require_both_face_and_action_flags(action, flag) -> None:
    report = PIIReport.model_validate(
        {
            "rows": [
                {
                    "entity_type": "FACE",
                    "action": action,
                    "detected_count": 2,
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
            ]
        }
    )
    findings = VisualFindings.model_validate({"faces": {"scan_status": "complete", "count": 2}})

    def result(preferences: NoticePreferences = DEFAULT_PREFERENCES) -> str:
        return render_report(
            report,
            _analysis(),
            {},
            decision="pass",
            route_class=None,
            visual_findings=findings,
            preferences=preferences,
        )

    for disabled in (flag, "show_detected_faces"):
        visible = result(NoticePreferences.model_validate({disabled: False}))
        assert "| Face |" not in visible
        assert "| Name |" in visible
    assert "| Face |" in result()


def test_failed_scan_never_looks_clean() -> None:
    findings = VisualFindings.model_validate({"faces": {"scan_status": "failed", "count": None}})

    def result(preferences: NoticePreferences) -> str:
        return render_report(
            PIIReport(rows=[]),
            _analysis(),
            {},
            decision="pass",
            route_class=None,
            visual_findings=findings,
            preferences=preferences,
        )

    assert "Face scan failed." in result(
        NoticePreferences(
            show_no_faces=False, show_detected_faces=False, show_unscanned_faces=False
        )
    )
    assert result(NoticePreferences(notices_enabled=False)) == ""


def test_cached_provenance_obeys_master_and_timing() -> None:
    cached = _analysis().model_copy(
        update={
            "source": "cached_decision",
            "scan_performed": False,
            "duration_ms": None,
            "cached_decision_applied": True,
        }
    )
    report = PIIReport.model_validate(
        {
            "rows": [
                {
                    "entity_type": "PERSON",
                    "action": "pass",
                    "detected_count": 1,
                    "transformed_count": 0,
                    "unique_transformed_count": 0,
                },
            ]
        }
    )
    assert "cached policy decision" in render_report(
        report, cached, {}, decision="pass", route_class=None
    )
    assert (
        render_report(
            report,
            cached,
            {},
            decision="pass",
            route_class=None,
            preferences=NoticePreferences(notices_enabled=False),
        )
        == ""
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
        preferences=NoticePreferences(show_pass=False, show_timing=False),
    )
    assert "| Face |" in result
    assert "| Name |" not in result and "| Phone |" in result
    assert "PII scan completed" not in result
