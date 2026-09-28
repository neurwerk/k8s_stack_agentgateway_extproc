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
                        "message": {"content": "Jane Doe"},
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
    assert seen["temperature"] == 0.2 and seen["top_p"] == 0.9
    assert seen["stream"] is False and "response_format" not in seen
    messages = seen["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 1
    user = messages[0]
    assert isinstance(user, dict)
    content = user["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    image_part = content[0]
    assert isinstance(image_part, dict)
    image = image_part["image_url"]
    assert isinstance(image, dict)
    assert image["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    ("finish_reason", "content", "expected"),
    [
        ("stop", "![image](image_1.png)", ImageInspectionResult(outcome="no_text_detected")),
        ("length", "repeated output", ImageInspectionResult(outcome="unreadable")),
        ("stop", "", ImageInspectionResult(outcome="failed")),
        ("content_filter", "text", ImageInspectionResult(outcome="failed")),
    ],
)
async def test_client_maps_native_ocr_outcomes(finish_reason, content, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": finish_reason,
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

    assert result == expected
