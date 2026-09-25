from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from agentgateway_extproc.config.settings import ImageInspectionSettings
from agentgateway_extproc.lib.image_inspection import ImageInspectionClient, ImageInspectionResult


async def test_client_uses_fixed_request_and_accepts_only_bounded_result():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"outcome": "text_extracted", "transcription": "Jane Doe"}
                            )
                        },
                    }
                ]
            },
            request=request,
        )

    settings = ImageInspectionSettings(
        enabled=True,
        base_url="https://inspection.test",
        api_key=SecretStr("test-only"),
        model="private-reader",
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ImageInspectionClient(settings, http)
    try:
        result = await client.inspect(b"\x89PNG\r\n\x1a\ncanonical")
    finally:
        await http.aclose()

    assert result.outcome == "text_extracted" and result.transcription == "Jane Doe"
    assert seen["model"] == "private-reader"
    assert seen["temperature"] == 0 and seen["stream"] is False
    messages = seen["messages"]
    assert isinstance(messages, list)
    user = messages[1]
    assert isinstance(user, dict)
    content = user["content"]
    assert isinstance(content, list)
    image_part = content[1]
    assert isinstance(image_part, dict)
    image = image_part["image_url"]
    assert isinstance(image, dict)
    assert image["url"].startswith("data:image/png;base64,") and image["detail"] == "high"


@pytest.mark.parametrize(
    "content",
    [
        '{"outcome":"text_extracted","transcription":""}',
        '{"outcome":"no_text_detected","transcription":" "}',
        '{"outcome":"unreadable","transcription":"description"}',
        '{"outcome":"no_text_detected"',
    ],
)
async def test_client_rejects_incomplete_or_inconsistent_results(content):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": content},
                    }
                ]
            },
            request=request,
        )

    settings = ImageInspectionSettings(
        enabled=True,
        base_url="https://inspection.test",
        api_key=SecretStr("test-only"),
        model="private-reader",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await ImageInspectionClient(settings, http).inspect(b"canonical-png")

    assert result == ImageInspectionResult(outcome="failed")
