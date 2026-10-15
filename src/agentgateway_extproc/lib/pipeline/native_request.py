"""Analyze native provider requests without an intermediate OpenAI payload."""

from __future__ import annotations

import json
import secrets
from typing import TYPE_CHECKING, cast

from neurwerk_request_segments import UnsupportedFeatureError, parse_request
from neurwerk_request_segments.native import AnthropicRequest, GoogleRequest
from pydantic import ValidationError

from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.lib.masking.reversal import placeholder_entity_prefixes
from agentgateway_extproc.lib.pipeline.guard import GUARD_INSTRUCTION
from agentgateway_extproc.lib.pipeline.profiles import (
    google_model_path,
    model_profile,
    operation_path,
)
from agentgateway_extproc.lib.pipeline.reply_validation import (
    validate_request_mutation,
    validate_reversal,
)
from agentgateway_extproc.lib.request_segments import extract_request
from agentgateway_extproc.lib.session import make_session_key
from agentgateway_extproc.models.exceptions import GatewayPolicyError, TrustedMetadataError
from agentgateway_extproc.models.types import RequestStats

if TYPE_CHECKING:
    from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler


async def process_native_request(  # noqa: C901
    handler: StreamHandler, payload: object
) -> ext_proc_pb2.ProcessingResponse:
    """Resolve once, inspect only canonical segments, then hand the canonical model to routing."""
    from agentgateway_extproc.lib.pipeline.request import (
        _literalize_request_placeholders,
        _raise_output_limit,
        immediate_response,
        request_mutation,
    )

    admission = handler.admission
    if admission is None:
        raise TrustedMetadataError
    kind = model_profile(admission).request_kind
    try:
        request = parse_request(payload, kind=kind)
    except UnsupportedFeatureError:
        raise GatewayPolicyError("unsupported_feature") from None
    except (ValidationError, ValueError, TypeError):
        raise GatewayPolicyError("invalid_request") from None
    if not isinstance(request, AnthropicRequest | GoogleRequest):
        raise GatewayPolicyError("unsupported_feature")
    if isinstance(request, GoogleRequest):
        model = google_model_path(
            operation_path(handler.request_headers[":path"], admission),
            admission.request_context.operation_id,
        )
        if request.model is not None and request.model != model:
            raise GatewayPolicyError("invalid_request")
        request = request.model_copy(update={"model": model})
    else:
        model = request.model
    context = handler.request_context = admission.resolve_model(model)
    handler.response_api_kind = kind
    streaming = (
        request.stream
        if isinstance(request, AnthropicRequest)
        else admission.request_context.operation_id == "streamGenerateContent"
    )
    handler.expected_response_format = "sse" if streaming else "json"
    handler.text_pii_enabled = context.pii_policy.enabled
    handler.response_notice_allowed = False
    handler.response_structured_json = bool(
        isinstance(request, GoogleRequest)
        and request.generationConfig is not None
        and request.generationConfig.responseMimeType == "application/json"
    )
    # Extraction limits apply even to PII-disabled native requests.
    extract_request(request)
    handler.request_body_chunks.clear()
    if context.pii_policy.enabled:
        request = cast(AnthropicRequest | GoogleRequest, _literalize_request_placeholders(request))
        conversation = handler.request_headers.get("x-session-id") or handler.request_headers.get(
            "x-conversation-id"
        )
        handler.request_nonce = secrets.token_bytes(32) if conversation is None else None
        key = make_session_key(
            admission.destination_policy,
            request.model_copy(update={"model": context.destination_id}),
            conversation_id=conversation,
            request_nonce=handler.request_nonce,
        )
        reply = await handler.client.analyze_request(
            request, key, correlation_id=handler.correlation_id
        )
        validate_request_mutation(request, reply)
        validate_reversal(request, reply)
        # Nonterminal replies also carry content classification in route_class.
        if reply.decision == "reroute":
            handler.request_context = admission.resolve_reroute(context, reply.route_class)
        if reply.decision == "block" or reply.request is None:
            handler.record_dispatch("policy_block")
            return immediate_response(403, '{"error":"request blocked by policy"}')
        handler.reversal_map.update(reply.reversal)
        handler.reversal_entity_prefixes = placeholder_entity_prefixes(handler.reversal_map)
        handler.request_stats = RequestStats(reply.report, reply.analysis, reply.decision)
        data = reply.request.model_dump(mode="json", exclude_unset=True)
        _inject_guard(data, kind)
        handler.guard_injected = True
        handler.record_dispatch("model_analyzed")
    else:
        data = request.model_dump(mode="json", exclude_unset=True)
        handler.record_dispatch("model_bypass")
    data["model"] = context.destination_id
    body = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > handler.max_transformed_request_bytes:
        _raise_output_limit(len(body), handler.max_transformed_request_bytes)
    return request_mutation(body, {}, True)


def _inject_guard(data: dict[str, object], kind: str) -> None:
    if kind == "anthropic":
        system = data.get("system")
        if isinstance(system, list):
            data["system"] = [{"type": "text", "text": GUARD_INSTRUCTION}, *system]
        else:
            data["system"] = GUARD_INSTRUCTION + (
                "\n\n" + system if isinstance(system, str) else ""
            )
    else:
        instruction = data.get("systemInstruction")
        instruction = dict(instruction) if isinstance(instruction, dict) else {}
        instruction["parts"] = [{"text": GUARD_INSTRUCTION}, *instruction.get("parts", [])]
        data["systemInstruction"] = instruction
