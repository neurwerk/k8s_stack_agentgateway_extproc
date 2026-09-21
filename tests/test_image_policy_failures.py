"""Check policy-aware image admission and content-free failures through extProc."""

import json
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from agentgateway_extproc.config.settings import Settings
from agentgateway_extproc.controllers.grpc_servicer import ExtProcServicer
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.documents import DocumentError
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.models.destination import ModelDestinationPolicy

from .conftest import MODEL_POLICY, body_request, header_request
from .test_images import image_part


@pytest.mark.parametrize(
    "version,pii,faces,valid",
    [
        (2, True, True, False),
        (3, False, True, False),
        (3, True, False, False),
        (3, True, True, True),
    ],
)
def test_policy_aware_mode_requires_v3_and_both_protections(version, pii, faces, valid):
    policy = {
        **MODEL_POLICY,
        "contract_version": version,
        "models": {"test": pii},
        "attachment_modes": {"test": "process"},
        "image_forwarding": {"test": "if-policy-allows"},
        "face_protection": {"test": faces},
    }
    if valid:
        assert ModelDestinationPolicy.model_validate(policy).contract_version == 3
    else:
        with pytest.raises(ValidationError):
            ModelDestinationPolicy.model_validate(policy)


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize(
    "case,status,message",
    [
        ("missing-scan", 503, "required image safety analysis could not be completed."),
        ("cached-scan", 503, "required image safety analysis could not be completed."),
        ("engine-unavailable", 503, "required image safety analysis could not be completed."),
        ("safety-rule", 403, "request blocked by configured data policy."),
        ("extraction-unavailable", 503, "text extraction service unavailable."),
        ("extraction-timeout", 504, "text extraction timed out."),
        ("extraction-failed", 503, "text extraction could not be completed."),
        (
            "no-engine-no-text",
            403,
            "image text extraction only; no text extracted from an image. "
            "Image forwarding is disabled for this model.",
        ),
    ],
)
async def test_image_failure_reasons_never_forward_or_claim_empty_extraction(
    engine_reply, api, case, status, message
):
    bypass = case == "no-engine-no-text"
    policy = {
        **MODEL_POLICY,
        "contract_version": 3,
        "models": {"test": not bypass},
        "attachment_modes": {"test": "process"},
        "image_forwarding": {"test": "none" if bypass else "if-policy-allows"},
        "face_protection": {"test": not bypass},
    }
    field = "messages" if api == "chat" else "input"
    payload = {"model": "test", field: [{"role": "user", "content": [image_part(api)]}]}

    async def convert(parts, *, images):
        assert len(parts) == 1
        if case.startswith("extraction-"):
            raise DocumentError(status, reason=case.replace("-", "_"))
        images.images = {0: "data:image/png;base64,bm9ybWFsaXplZA=="}
        images.text_present = {0: not bypass}
        images.scan_status = "not_scanned" if bypass else "complete"
        return ["[Image: no text extracted]" if bypass else "PRIVATE-EXTRACTED-TEXT"]

    def engine(request):
        assert not bypass and not case.startswith("extraction-")
        if case == "engine-unavailable":
            raise httpx.ConnectError("PRIVATE-UPSTREAM-DETAIL")
        sent = json.loads(request.content)
        engine_reply.update(
            request=sent["request"],
            visual_findings=sent["visual_findings"],
            decision="pass",
            entities=[],
            entity_counts={},
            applied_actions=[],
            reversal={},
            report={"rows": []},
            notices={"request": [], "response": []},
            safety_rule="synthetic-rule" if case == "safety-rule" else None,
        )
        if case == "missing-scan":
            engine_reply["analysis"].update(scan_performed=False, duration_ms=None)
        elif case == "cached-scan":
            engine_reply["analysis"].update(source="cached_decision", cached_decision_applied=True)
        return httpx.Response(200, json=engine_reply)

    settings = Settings()
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as http:
        docling = DoclingClient(settings.docling)
        servicer = ExtProcServicer(EngineClient(settings.engine, http), settings, docling)

        async def requests():
            yield header_request(policy=policy)
            yield body_request(json.dumps(payload).encode(), policy=policy)
            pytest.fail("rejected requests must not dispatch upstream")

        with patch.object(docling, "convert", side_effect=convert):
            replies = [reply async for reply in servicer.Process(requests(), object())]
        await docling.close()
    rejected = replies[-1].immediate_response
    assert rejected.status.code == status
    error = json.loads(rejected.body)["error"]
    public_message = error["message"] if isinstance(error, dict) else error
    assert public_message.startswith("neurwerk: " + message)
    assert "PRIVATE-" not in rejected.body and "base64" not in rejected.body
    assert not any(reply.HasField("request_body") for reply in replies)
