"""Translate bounded provider rejections into fixed, content-free diagnostics."""

from __future__ import annotations

import json
import re

from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.lib.pipeline.request import immediate_response

MAX_PROVIDER_ERROR_BYTES = 8192
_ERRORS = {
    400: ("provider_bad_request", "The provider rejected the request parameters or model."),
    401: ("provider_authentication_failed", "The provider rejected its credential."),
    403: ("provider_access_denied", "The provider denied access for this account."),
    404: ("provider_not_found", "The provider could not find the requested model or endpoint."),
    429: ("provider_rate_limited", "The provider account has reached a rate or usage limit."),
}
_PARAMETERS = {
    "include",
    "max_output_tokens",
    "parallel_tool_calls",
    "reasoning",
    "reasoning.effort",
    "store",
    "stream",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
}


def provider_error_response(status: int, body: bytes = b"") -> ext_proc_pb2.ProcessingResponse:
    """Preserve status; never return provider text, headers or unrecognized fields."""
    code, message = _ERRORS.get(status, ("provider_error", "Provider request failed."))
    error = {"code": code, "message": message}
    if status == 400 and body and len(body) <= MAX_PROVIDER_ERROR_BYTES:
        error.update(_bad_request_detail(body))
    return immediate_response(status, json.dumps({"error": error}, separators=(",", ":")))


def _bad_request_detail(body: bytes) -> dict[str, str]:
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        return {}
    if not isinstance(payload, dict):
        return {}
    error = payload.get("error")
    message = error.get("message") if isinstance(error, dict) else payload.get("detail")
    if not isinstance(message, str):
        return {}
    if re.fullmatch(
        r"The '[A-Za-z0-9_.:-]{1,128}' model is not supported when using Codex "
        r"with a ChatGPT account\.",
        message,
    ):
        return {
            "code": "provider_model_unavailable",
            "message": "This model is unavailable for your ChatGPT account. Choose another model.",
        }
    parameter = re.fullmatch(r"Unsupported parameter: '([a-z_.]{1,64})'\.?", message)
    if parameter is not None and parameter[1] in _PARAMETERS:
        return {
            "code": "provider_unsupported_parameter",
            "message": f"The provider does not support the '{parameter[1]}' parameter.",
            "param": parameter[1],
        }
    if message == "Stream must be set to true.":
        return {
            "code": "provider_stream_required",
            "message": "This provider account requires streaming responses.",
            "param": "stream",
        }
    return {}
