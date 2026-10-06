"""Validate request transport and attachments before optional engine policy."""

# ruff: noqa: C901

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from neurwerk_request_segments import CompatibilitySettings, UnsupportedFeatureError, parse_request
from pydantic import ValidationError

from agentgateway_extproc.config.settings import MAX_REQUEST_BYTES
from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.lib.documents import MAX_TEXT, DocumentError, ImageBatch
from agentgateway_extproc.lib.engine.client import EngineClient
from agentgateway_extproc.lib.image_policy import image_output
from agentgateway_extproc.lib.json_limits import JsonBudgetError
from agentgateway_extproc.lib.masking.reversal import placeholder_entity_prefixes
from agentgateway_extproc.lib.pipeline.guard import inject_guard_instruction
from agentgateway_extproc.lib.pipeline.mcp import (
    McpProtocolError,
    McpUnsupportedFeatureError,
    parse_mcp_message,
    strict_json_loads,
)
from agentgateway_extproc.lib.pipeline.reply_validation import (
    validate_request_mutation,
    validate_reversal,
)
from agentgateway_extproc.lib.pipeline.request_validation import (
    log_model_validation_failure,
    model_request_family,
)
from agentgateway_extproc.lib.request_segments import extract_request
from agentgateway_extproc.lib.session import make_session_key
from agentgateway_extproc.models.destination import ModelDestinationPolicy
from agentgateway_extproc.models.engine import (
    EngineAttachmentPart,
    EngineChatRequest,
    EngineMcpRequest,
    EngineMessage,
    EngineReply,
    EngineRequest,
    EngineResponseMessage,
    EngineResponsesRequest,
    VisualFindings,
)
from agentgateway_extproc.models.exceptions import (
    EnginePolicyError,
    EngineUnavailableError,
    InvalidEngineReplyError,
    LimitDetail,
)
from agentgateway_extproc.models.types import (
    PRESIDIO_NO_PII,
    PRESIDIO_PII_DETECTED,
    PRESIDIO_PII_TRANSFORMED,
    PRESIDIO_REROUTED,
    RESERVED_PLACEHOLDER_PREFIX_RE,
    RequestStats,
)

if TYPE_CHECKING:
    from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler

type OpaqueReasoning = dict[int, dict[str, object]]

_OPAQUE_REQUEST_REASONING_FIELDS = (
    "reasoning_content",
    "reasoning",
    "reasoning_details",
    "thinking_blocks",
    "reasoning_signature",
)
_UNCHECKED_IMAGE_MARKER = "[Image forwarded without privacy inspection]"
_logger = logging.getLogger(__name__)


async def process_request(
    handler: StreamHandler, client: EngineClient
) -> ext_proc_pb2.ProcessingResponse:
    """Validate a request, call the engine, and apply its complete reply."""
    body = b"".join(handler.request_body_chunks)
    if len(body) > handler.max_request_bytes:
        raise EnginePolicyError(
            "request_too_large",
            limit=LimitDetail(
                stage="admission",
                reason="bytes",
                measured=len(body),
                maximum=handler.max_request_bytes,
                unit="bytes",
                exact=True,
            ),
            correlation_id=handler.correlation_id,
        )
    try:
        payload = strict_json_loads(body.decode("utf-8"))
    except UnicodeDecodeError:
        return immediate_response(400, '{"error":"invalid request encoding"}')
    except JsonBudgetError as exc:
        _clear_request(handler)
        raise EnginePolicyError(
            "request_too_large", limit=exc.limit, correlation_id=handler.correlation_id
        ) from exc
    except (json.JSONDecodeError, TypeError, ValueError):
        return immediate_response(400, '{"error":"invalid request JSON"}')
    policy = handler.destination_policy
    if policy is None:
        raise ValueError("trusted destination policy is unavailable")  # noqa: TRY003
    opaque_reasoning: OpaqueReasoning = {}
    converted = False
    image_locations: dict[tuple[int, int], str] = {}
    unchecked_image_locations: dict[tuple[int, int], str] = {}
    image_forwarding = "none"
    images: ImageBatch | None = None
    visual_findings: VisualFindings | None = None
    if policy.destination_kind == "mcp":
        handler.response_api_kind = "mcp"
        if handler.mcp_headers is None:
            raise ValueError("validated MCP headers are unavailable")  # noqa: TRY003
        try:
            context = parse_mcp_message(body, handler.mcp_headers)
        except McpUnsupportedFeatureError as exc:
            handler.record_dispatch("protocol_failure")
            return immediate_response(400, exc.body)
        except McpProtocolError:
            handler.record_dispatch("protocol_failure")
            return immediate_response(400, '{"error":"invalid MCP request"}')
        engine_request = context.engine_request
        has_text_arguments = context.has_text_arguments
        handler.mcp_context = replace(
            context,
            engine_request=None,
            has_text_arguments=False,
        )
        if not policy.pii_enabled:
            _clear_request(handler)
            handler.record_dispatch("mcp_protocol_only")
            return request_mutation(body, {}, False)
        if engine_request is None:
            _clear_request(handler)
            handler.record_dispatch("mcp_lifecycle_pass")
            return request_mutation(body, {}, False)
        if not has_text_arguments:
            _clear_request(handler)
            handler.record_dispatch("mcp_no_text_pass")
            return request_mutation(body, {}, False)
        request: EngineRequest = engine_request
        if handler.mcp_headers.session_id is None:
            handler.request_nonce = secrets.token_bytes(32)
        session_key = make_session_key(
            policy,
            request,
            mcp_session_id=handler.mcp_headers.session_id,
            request_nonce=handler.request_nonce,
        )
        handler.record_dispatch("mcp_analyzed")
    else:
        try:
            opaque_reasoning = _extract_opaque_chat_reasoning(payload)
        except ValueError:
            handler.record_dispatch("protocol_failure")
            return immediate_response(400, '{"error":"invalid model request"}')
        try:
            path = handler.request_headers.get(":path", "").split("?", 1)[0]
            kind = (
                "responses"
                if path.endswith("/responses")
                else "chat"
                if path.endswith("/chat/completions")
                else None
            )
            request = parse_request(
                payload, kind=kind, controls=getattr(handler, "compatibility", None)
            )
        except UnsupportedFeatureError as exc:
            _logger.warning(
                "model request validation failed family=%s reason=%s scope=%s count=%d",
                model_request_family(payload),
                "extra_forbidden",
                {
                    "stream_options": "stream_options",
                    "message": "messages",
                    "tool": "tools",
                    "function": "tools",
                }.get(exc.location, "top_level"),
                1,
            )
            handler.record_dispatch("protocol_failure")
            return immediate_response(
                400,
                '{"error":{"code":"unsupported_feature",'
                '"message":"This request contains an unsupported feature."}}',
            )
        except ValidationError as exc:
            log_model_validation_failure(payload, exc)
            handler.record_dispatch("protocol_failure")
            return immediate_response(400, '{"error":"invalid model request"}')
        except (TypeError, ValueError):
            _logger.warning(
                "model request validation failed family=%s reason=%s scope=%s count=%d",
                model_request_family(payload),
                "invalid_value",
                "top_level",
                1,
            )
            handler.record_dispatch("protocol_failure")
            return immediate_response(400, '{"error":"invalid model request"}')
        if not isinstance(request, EngineChatRequest | EngineResponsesRequest):
            handler.record_dispatch("protocol_failure")
            return immediate_response(400, '{"error":"invalid model request"}')
        if request.model not in policy.models:
            return immediate_response(400, '{"error":"unknown model"}')
        handler.text_pii_enabled = policy.models[request.model]
        attachments = _model_attachments(request)
        attachment_mode = policy.attachment_modes.get(request.model, "block")
        image_forwarding = policy.image_forwarding.get(request.model, "none")
        unchecked_without_documents = False
        if attachments and policy.contract_version in {4, "4.1"}:
            document_indexes = {
                index
                for index, part in enumerate(attachments)
                if part.type in {"file", "input_file"}
            }
            image_indexes = {
                index
                for index, part in enumerate(attachments)
                if part.type in {"image_url", "input_image"}
            }
            try:
                if len(document_indexes | image_indexes) != len(attachments):
                    raise DocumentError(400, reason="unsupported_format")  # noqa: TRY301
                expected_type = "file" if isinstance(request, EngineChatRequest) else "input_file"
                expected_image = (
                    "image_url" if isinstance(request, EngineChatRequest) else "input_image"
                )
                if any(
                    (index in document_indexes and part.type != expected_type)
                    or (index in image_indexes and part.type != expected_image)
                    for index, part in enumerate(attachments)
                ):
                    raise DocumentError(400, reason="unsupported_format")  # noqa: TRY301
                if (document_indexes and policy.document_mode(request.model) == "block") or (
                    image_indexes and policy.image_mode(request.model) == "block"
                ):
                    raise DocumentError(403, reason="attachments_disabled")  # noqa: TRY301
                unchecked = bool(image_indexes) and image_forwarding == "pii-unchecked"
                inspect_images = (
                    image_indexes
                    if policy.contract_version == "4.1"
                    and policy.image_inspection[request.model] == "document-and-vision"
                    else set()
                )
                images = (
                    ImageBatch(
                        protect_faces=not unchecked and policy.protects_faces(request.model),
                        defer_face_inspection=bool(inspect_images),
                        policy_version=4,
                    )
                    if image_indexes
                    else None
                )
                if unchecked:
                    images = cast(ImageBatch, images)
                    if handler.docling is None:
                        raise DocumentError(reason="extraction_unavailable")  # noqa: TRY301
                    texts_by_index = await handler.docling.convert_selected(
                        attachments,
                        document_indexes,
                        images=images,
                        inspect_images=inspect_images,
                    )
                    _validate_image_inspections(images, inspect_images)
                    request, body, unchecked_image_locations = _converted_unchecked_request(
                        payload,
                        texts_by_index,
                        images,
                        opaque_reasoning,
                        controls=request._controls,
                    )
                    unchecked_without_documents = not document_indexes
                else:
                    if handler.docling is None:
                        raise DocumentError(reason="extraction_unavailable")  # noqa: TRY301
                    texts = await handler.docling.convert(
                        attachments,
                        images=images,
                        inspect_images=inspect_images,
                    )
                    if images is not None:
                        _validate_image_inspections(images, inspect_images)
                        visual_findings = _visual_findings(attachments, images)
                        if image_forwarding != "none":
                            image_locations = _image_locations(payload, images)
                    request, body = _converted_request(
                        payload, texts, opaque_reasoning, controls=request._controls
                    )
            except DocumentError as exc:
                if policy.contract_version == "4.1" and exc.reason.startswith(
                    ("extraction_", "image_analysis_", "image_inspection_")
                ):
                    return _image_policy_block(handler, exc)
                _clear_request(handler)
                handler.record_dispatch(
                    "policy_block"
                    if exc.reason in {"attachments_disabled", "image_inspection_unreadable"}
                    else "transport_failure"
                )
                return immediate_response(exc.status, json.dumps({"error": exc.message}))
            converted = True
            handler.request_body_chunks.clear()
            attachments.clear()
            payload = None
        elif attachments and attachment_mode != "passthrough":
            allowed = {"file", "input_file"}
            if isinstance(policy.contract_version, int) and policy.contract_version >= 2:
                allowed |= {"image_url", "input_image"}
            if attachment_mode not in {"extract", "process"}:
                _clear_request(handler)
                handler.record_dispatch("policy_block")
                error = DocumentError(403, reason="attachments_disabled")
                return immediate_response(error.status, json.dumps({"error": error.message}))
            try:
                if any(part.type not in allowed for part in attachments):
                    raise DocumentError(400, reason="unsupported_format")  # noqa: TRY301
                if handler.docling is None:
                    raise DocumentError(reason="extraction_unavailable")  # noqa: TRY301
                expected_type = "file" if isinstance(request, EngineChatRequest) else "input_file"
                expected_image = (
                    "image_url" if isinstance(request, EngineChatRequest) else "input_image"
                )
                if any(part.type not in {expected_type, expected_image} for part in attachments):
                    raise DocumentError(400, reason="unsupported_format")  # noqa: TRY301
                if any(part.type == expected_image for part in attachments):
                    images = ImageBatch(
                        protect_faces=(policy.contract_version == 3 or image_forwarding != "none")
                        and policy.protects_faces(request.model),
                        policy_version=cast(int, policy.contract_version),
                    )
                    texts = await handler.docling.convert(attachments, images=images)
                    if isinstance(policy.contract_version, int) and policy.contract_version >= 3:
                        visual_findings = _visual_findings(attachments, images)
                    if image_forwarding != "none":
                        image_locations = _image_locations(payload, images)
                else:
                    texts = await handler.docling.convert(attachments)
                request, body = _converted_request(
                    payload, texts, opaque_reasoning, controls=request._controls
                )
            except DocumentError as exc:
                _clear_request(handler)
                handler.record_dispatch("transport_failure")
                return immediate_response(exc.status, json.dumps({"error": exc.message}))
            converted = True
            # Discard base64 and wire buffers before the potentially long PII call.
            handler.request_body_chunks.clear()
            attachments.clear()
            payload = None
        if unchecked_without_documents or (
            not handler.text_pii_enabled and not (images and images.protect_faces)
        ):
            if (
                images is not None
                and policy.contract_version in {3, 4, "4.1"}
                and not unchecked_without_documents
            ):
                try:
                    image_output(
                        policy,
                        request.model,
                        images,
                        None,
                        allow_textless=_allow_inspected_textless(policy, request.model, images),
                    )
                except DocumentError as exc:
                    return _image_policy_block(handler, exc)
            if image_locations:
                data = cast(dict[str, object], strict_json_loads(body.decode()))
                try:
                    _restore_images(data, image_locations, offset=0)
                    body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
                    if len(body) > handler.max_transformed_request_bytes:
                        _raise_output_limit(len(body), handler.max_transformed_request_bytes)
                except DocumentError as exc:
                    _clear_request(handler)
                    return immediate_response(exc.status, json.dumps({"error": exc.message}))
            if unchecked_image_locations:
                data = cast(dict[str, object], strict_json_loads(body.decode()))
                try:
                    _replace_unchecked_images(data, unchecked_image_locations, offset=0)
                    _restore_opaque_chat_reasoning(data, opaque_reasoning, offset=0)
                    body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
                    if len(body) > handler.max_transformed_request_bytes:
                        _raise_output_limit(len(body), handler.max_transformed_request_bytes)
                except DocumentError as exc:
                    _clear_request(handler)
                    return immediate_response(exc.status, json.dumps({"error": exc.message}))
            _clear_request(handler)
            handler.response_processing_enabled = False
            handler.record_dispatch("model_bypass")
            return request_mutation(body, {}, converted)
        conversation_id = handler.request_headers.get(
            "x-session-id"
        ) or handler.request_headers.get("x-conversation-id")
        if converted:
            conversation_id = None
        if conversation_id is None:
            handler.request_nonce = secrets.token_bytes(32)
        session_key = make_session_key(
            policy,
            request,
            conversation_id=conversation_id[:256] if conversation_id else None,
            request_nonce=handler.request_nonce,
        )
        handler.record_dispatch("model_analyzed")
    request = _literalize_request_placeholders(request)
    try:
        if converted:
            _clear_request(handler)
            if visual_findings is not None:
                reply = await client.analyze_request(
                    request,
                    session_key,
                    document=True,
                    text_pii_enabled=handler.text_pii_enabled,
                    visual_findings=visual_findings,
                    correlation_id=handler.correlation_id,
                )
            else:
                reply = await client.analyze_request(
                    request, session_key, document=True, correlation_id=handler.correlation_id
                )
        else:
            reply = await client.analyze_request(
                request, session_key, correlation_id=handler.correlation_id
            )
        validate_request_mutation(request, reply)
        if handler.text_pii_enabled:
            validate_reversal(request, reply)
    except (EngineUnavailableError, InvalidEngineReplyError):
        if images is None:
            raise
        _clear_request(handler)
        handler.record_dispatch("transport_failure")
        error = DocumentError(reason="image_analysis_failed")
        return immediate_response(error.status, json.dumps({"error": error.message}))
    if (
        image_locations
        and visual_findings is None
        and image_forwarding == "if-no-pii-detected"
        and (
            reply.decision != "pass"
            or reply.entities
            or reply.report.rows
            or reply.analysis.source != "current_request"
            or not reply.analysis.scan_performed
            or reply.analysis.cached_decision_applied
            or reply.analysis.text_leaf_count < 1
            or reply.safety_rule is not None
        )
    ):
        _clear_request(handler)
        handler.record_dispatch("policy_block")
        error = DocumentError(
            403, reason=("image_pii_detected" if reply.entities else "policy_blocked")
        )
        return immediate_response(error.status, json.dumps({"error": error.message}))
    if isinstance(request, EngineMcpRequest) and (
        reply.decision == "reroute" or reply.route_class is not None
    ):
        raise InvalidEngineReplyError(  # noqa: TRY003
            "MCP engine reply contains model routing data"
        )

    _clear_request(handler)
    handler.response_api_kind = (
        "mcp"
        if isinstance(request, EngineMcpRequest)
        else "responses"
        if isinstance(request, EngineResponsesRequest)
        else "chat"
    )
    structured_response = (
        isinstance(request, EngineChatRequest) and _has_structured_chat_format(request)
    ) or (isinstance(request, EngineResponsesRequest) and _has_structured_responses_format(request))
    is_mcp = isinstance(request, EngineMcpRequest)
    handler.response_notice_allowed = not structured_response and not is_mcp
    handler.response_structured_json = structured_response
    handler.presidio_code = (
        _presidio_code(reply)
        if handler.text_pii_enabled
        and isinstance(request, EngineChatRequest | EngineResponsesRequest)
        else None
    )
    handler.notice_messages = [] if is_mcp else list(reply.notices.response)
    handler.reversal_map.update(reply.reversal)
    handler.reversal_entity_prefixes = placeholder_entity_prefixes(handler.reversal_map)
    handler.request_stats = RequestStats(
        reply.report, reply.analysis, reply.decision, reply.route_class, reply.visual_findings
    )
    if reply.decision == "block" or reply.request is None:
        handler.record_dispatch("policy_block")
        if is_mcp:
            context = handler.mcp_context
            if context is None or context.request_id is None:
                raise InvalidEngineReplyError(  # noqa: TRY003
                    "blocked MCP request has no request ID"
                )
            body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": context.request_id,
                    "error": {
                        "code": -32000,
                        "message": "Request blocked by data policy",
                    },
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            return immediate_response(200, body)
        if visual_findings is not None:
            return _image_policy_block(handler)
        return immediate_response(
            403,
            json.dumps(
                {
                    "error": DocumentError(403).message
                    if image_locations
                    else "request blocked by policy"
                }
            ),
        )
    if (
        images is not None
        and isinstance(policy, ModelDestinationPolicy)
        and isinstance(request, EngineChatRequest | EngineResponsesRequest)
        and policy.contract_version in {3, 4, "4.1"}
    ):
        try:
            output = image_output(
                policy,
                request.model,
                images,
                reply,
                allow_textless=_allow_inspected_textless(policy, request.model, images),
            )
            handler.request_stats.images_forwarded = output.forward_pixels
            if not output.forward_pixels:
                image_locations.clear()
            if output.notice:
                handler.safety_notice_messages.append(output.notice)
        except DocumentError as exc:
            return _image_policy_block(handler, exc)
    transformed = reply.request
    if handler.text_pii_enabled and isinstance(request, EngineChatRequest | EngineResponsesRequest):
        transformed = inject_guard_instruction(
            cast(EngineChatRequest | EngineResponsesRequest, transformed)
        )
        handler.guard_injected = True
    serialized = cast(
        dict[str, object],
        transformed.model_dump(mode="json", by_alias=True, exclude_unset=True),
    )
    if isinstance(request, EngineChatRequest):
        for message in _dict_list(serialized.get("messages")):
            if message.get("tool_calls") == []:
                del message["tool_calls"]
        _restore_opaque_chat_reasoning(
            serialized, opaque_reasoning, offset=int(handler.guard_injected)
        )
    if image_locations:
        try:
            _restore_images(
                serialized,
                image_locations,
                offset=int(handler.guard_injected and isinstance(request, EngineChatRequest)),
            )
        except DocumentError as exc:
            return immediate_response(exc.status, json.dumps({"error": exc.message}))
    if unchecked_image_locations:
        try:
            _replace_unchecked_images(
                serialized,
                unchecked_image_locations,
                offset=int(handler.guard_injected and isinstance(request, EngineChatRequest)),
            )
        except DocumentError as exc:
            return immediate_response(exc.status, json.dumps({"error": exc.message}))
    mutated = json.dumps(
        serialized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if len(mutated) > handler.max_transformed_request_bytes:
        raise EnginePolicyError(
            "request_too_large",
            limit=LimitDetail(
                stage="output",
                reason="transformed_bytes",
                measured=len(mutated),
                maximum=handler.max_transformed_request_bytes,
                unit="bytes",
                exact=True,
            ),
            correlation_id=handler.correlation_id,
        )
    headers: dict[str, str] = {}
    if not is_mcp:
        headers = {
            "x-remote-allowed": str(reply.remote_allowed).lower(),
            "x-route-class": reply.route_class or "",
        }
        if reply.entities:
            headers["x-pii-entities"] = ",".join(reply.entities)
    return request_mutation(mutated, headers, converted or mutated != body)


def _literalize_request_placeholders(request: EngineRequest) -> EngineRequest:
    """Keep incoming aliases masked without making them eligible for reversal."""
    extracted = extract_request(request)
    if not any(RESERVED_PLACEHOLDER_PREFIX_RE.search(part.text) for part in extracted.segments):
        return request
    return extracted.rebuild(
        [
            part.model_copy(
                update={
                    "text": RESERVED_PLACEHOLDER_PREFIX_RE.sub(
                        lambda match: f"<LITERAL_{match.group()[1:]}", part.text
                    )
                }
            )
            for part in extracted.segments
        ]
    )


def _visual_findings(attachments: list[EngineAttachmentPart], images: ImageBatch) -> VisualFindings:
    """Require complete local facts before calling central visual policy."""
    expected = {
        index for index, part in enumerate(attachments) if part.type in {"image_url", "input_image"}
    }
    if (
        images.images.keys() != expected
        or images.text_present.keys() != expected
        or any(type(present) is not bool for present in images.text_present.values())
        or images.scan_status != ("complete" if images.protect_faces else "not_scanned")
        or (not images.protect_faces and images.face_count != 0)
    ):
        raise DocumentError(reason="image_analysis_failed")
    return VisualFindings.model_validate(
        {
            "faces": {
                "scan_status": images.scan_status,
                "count": images.face_count if images.scan_status == "complete" else None,
            }
        },
        strict=True,
    )


def _validate_image_inspections(images: ImageBatch, expected: set[int]) -> None:
    """Reject missing, extra or failed inspection results before central analysis."""
    if images.inspections.keys() != expected or any(
        result.outcome == "failed" for result in images.inspections.values()
    ):
        raise DocumentError(reason="image_inspection_failed")


def _allow_inspected_textless(
    policy: ModelDestinationPolicy, model: str, images: ImageBatch
) -> bool:
    """Allow textless pixels only after the explicit v4.1 inspection completed."""
    if policy.contract_version != "4.1" or policy.image_textless[model] != "allow-if-inspected":
        return False
    textless = {index for index, present in images.text_present.items() if not present}
    return bool(textless) and all(
        images.inspections.get(index) is not None
        and images.inspections[index].outcome == "no_text_detected"
        for index in textless
    )


def _image_policy_block(
    handler: StreamHandler, error: DocumentError | None = None
) -> ext_proc_pb2.ProcessingResponse:
    """Report the actual rejection reason without rewriting the engine's FACE action."""
    _clear_request(handler)
    dispatch = "policy_block" if error is None or error.status == 403 else "transport_failure"
    handler.record_dispatch(dispatch)
    if error is None:
        blocked_entities = (
            {row.entity_type for row in handler.request_stats.report.rows if row.action == "block"}
            if handler.request_stats
            else set()
        )
        error = DocumentError(
            403,
            reason=("face_policy_blocked" if blocked_entities == {"FACE"} else "policy_blocked"),
        )
    message = error.message
    stage = (
        "extraction"
        if error.reason.startswith("extraction_")
        else "image_inspection"
        if error.reason.startswith("image_inspection_")
        else "pii_analysis"
        if error.reason.startswith("image_analysis_")
        else "policy"
    )
    _logger.log(
        logging.INFO if error.status == 403 else logging.WARNING,
        "attachment processing stopped stage=%s status=%d reason=%s",
        stage,
        error.status,
        error.reason,
    )
    payload: dict[str, object] = {
        "attachment_report": {
            "stage": stage,
            "status": error.status,
            "reason": error.reason,
        }
    }
    if stats := handler.request_stats:
        stats.decision = "block"
        stats.route_class = None
        report: dict[str, object] = {
            **stats.report.model_dump(mode="json"),
            "decision": "block",
            "reason": "no_readable_text" if error.no_text else error.reason,
            "analysis": stats.analysis.model_dump(mode="json"),
            "visual_findings": (
                stats.visual_findings.model_dump(mode="json") if stats.visual_findings else None
            ),
        }
        payload["pii_report"] = report
    return immediate_response(
        error.status,
        json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "policy_error" if error.status == 403 else "processing_error",
                    "code": "image_text_unavailable" if error.no_text else error.reason,
                },
                **payload,
            }
        ),
    )


def _converted_request(
    payload: object,
    texts: list[str],
    reasoning: OpaqueReasoning,
    *,
    controls: CompatibilitySettings | None = None,
) -> tuple[EngineChatRequest | EngineResponsesRequest, bytes]:
    """Replace only typed file parts, retaining the original protocol controls."""
    data = cast(dict[str, object], payload)
    chat = "messages" in data
    remaining = iter(texts)
    count = 0
    for message in _dict_list(data.get("messages" if chat else "input")):
        if not chat and message.get("type", "message") != "message":
            continue
        for part in _dict_list(message.get("content")):
            if part.get("type") not in {"file", "input_file", "image_url", "input_image"}:
                continue
            value = next(remaining, None)
            if value is None:
                raise DocumentError
            part.clear()
            part.update(type="text" if chat else "input_text", text=value)
            count += 1
    if count != len(texts):
        raise DocumentError
    request = cast(
        EngineChatRequest | EngineResponsesRequest,
        parse_request(data, controls=controls),
    )
    serialized_size = len(request.model_dump_json(by_alias=True, exclude_unset=True).encode())
    if serialized_size > MAX_REQUEST_BYTES:
        _raise_output_limit(serialized_size, MAX_REQUEST_BYTES)
    _check_converted_text_limit(request)
    _restore_opaque_chat_reasoning(data, reasoning, offset=0)
    body = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > MAX_REQUEST_BYTES:
        _raise_output_limit(len(body), MAX_REQUEST_BYTES)
    return request, body


def _converted_unchecked_request(
    payload: object,
    texts: dict[int, str],
    images: ImageBatch,
    reasoning: OpaqueReasoning,
    *,
    controls: CompatibilitySettings | None = None,
) -> tuple[
    EngineChatRequest | EngineResponsesRequest,
    bytes,
    dict[tuple[int, int], str],
]:
    """Convert documents while keeping unchecked image bytes outside policy analysis."""
    data = cast(dict[str, object], payload)
    chat = "messages" in data
    locations: dict[tuple[int, int], str] = {}
    attachment = 0
    for mi, message in enumerate(_dict_list(data.get("messages" if chat else "input"))):
        if not chat and message.get("type", "message") != "message":
            continue
        for pi, part in enumerate(_dict_list(message.get("content"))):
            if part.get("type") not in {"file", "input_file", "image_url", "input_image"}:
                continue
            if attachment in images.images:
                locations[mi, pi] = images.images[attachment]
                value = _UNCHECKED_IMAGE_MARKER
            else:
                value = texts.get(attachment)
                if value is None:
                    raise DocumentError
            part.clear()
            part.update(type="text" if chat else "input_text", text=value)
            attachment += 1
    if attachment != len(texts) + len(images.images) or len(locations) != len(images.images):
        raise DocumentError
    request = cast(
        EngineChatRequest | EngineResponsesRequest,
        parse_request(data, controls=controls),
    )
    _check_converted_text_limit(request)
    body = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > MAX_REQUEST_BYTES:
        _raise_output_limit(len(body), MAX_REQUEST_BYTES)
    return request, body, locations


def _raise_output_limit(measured: int, maximum: int) -> None:
    """Keep actual serialized byte measurements when converted output is rejected."""
    raise EnginePolicyError(
        "request_too_large",
        limit=LimitDetail(
            stage="output",
            reason="transformed_bytes",
            measured=measured,
            maximum=maximum,
            unit="bytes",
            exact=True,
        ),
    )


def _check_converted_text_limit(request: EngineRequest) -> None:
    """Count all independent converted text leaves without approximating wire size."""
    measured = sum(len(segment.text) for segment in extract_request(request).segments)
    if measured > MAX_TEXT:
        raise EnginePolicyError(
            "request_too_large",
            limit=LimitDetail(
                stage="inspection",
                reason="text_characters",
                measured=measured,
                maximum=MAX_TEXT,
                unit="characters",
                exact=True,
            ),
        )


def _replace_unchecked_images(
    data: dict[str, object],
    locations: dict[tuple[int, int], str],
    *,
    offset: int,
) -> None:
    """Replace private analysis markers with canonical pixels at the original position."""
    chat = "messages" in data
    messages = _dict_list(data.get("messages" if chat else "input"))
    for (mi, pi), uri in locations.items():
        content = messages[mi + offset].get("content") if mi + offset < len(messages) else None
        if not isinstance(content, list) or pi >= len(content):
            raise DocumentError
        part = content[pi]
        expected_type = "text" if chat else "input_text"
        if not isinstance(part, dict) or part != {
            "type": expected_type,
            "text": _UNCHECKED_IMAGE_MARKER,
        }:
            raise DocumentError
        content[pi] = (
            {"type": "image_url", "image_url": {"url": uri}}
            if chat
            else {"type": "input_image", "image_url": uri}
        )


def _image_locations(payload: object, images: ImageBatch) -> dict[tuple[int, int], str]:
    """Bind normalized pixels to original content positions, never to reader output."""
    data = cast(dict[str, object], payload)
    locations: dict[tuple[int, int], str] = {}
    attachment = 0
    for mi, message in enumerate(
        _dict_list(data.get("messages" if "messages" in data else "input"))
    ):
        for pi, part in enumerate(_dict_list(message.get("content"))):
            if part.get("type") in {"file", "input_file", "image_url", "input_image"}:
                if attachment in images.images:
                    locations[mi, pi] = images.images[attachment]
                attachment += 1
    if len(locations) != len(images.images):
        raise DocumentError
    return locations


def _restore_images(
    data: dict[str, object],
    locations: dict[tuple[int, int], str],
    *,
    offset: int,
) -> None:
    """Insert checked pixels after their scanned text, preserving history and ordering."""
    chat = "messages" in data
    messages = _dict_list(data.get("messages" if chat else "input"))
    for (mi, pi), uri in sorted(locations.items(), reverse=True):
        content = messages[mi + offset].get("content")
        if not isinstance(content, list):
            raise DocumentError
        if len(content) >= 64:
            raise DocumentError(413)
        part = (
            {"type": "image_url", "image_url": {"url": uri}}
            if chat
            else {"type": "input_image", "image_url": uri}
        )
        content.insert(pi + 1, part)


def _model_attachments(
    request: EngineChatRequest | EngineResponsesRequest,
) -> list[EngineAttachmentPart]:
    """Check typed content parts, including history, without interpreting tool JSON."""
    messages = request.messages if isinstance(request, EngineChatRequest) else request.input
    if isinstance(messages, str):
        return []
    return [
        part
        for message in messages
        if isinstance(message, EngineMessage | EngineResponseMessage)
        and isinstance(message.content, list)
        for part in message.content
        if isinstance(part, EngineAttachmentPart)
    ]


def _clear_request(handler: StreamHandler) -> None:
    """Discard caller body and headers after request dispatch is complete."""
    handler.request_body_chunks.clear()
    handler.request_headers.clear()


def _extract_opaque_chat_reasoning(payload: object) -> OpaqueReasoning:
    """Remove trusted assistant reasoning before strict engine validation."""
    if not isinstance(payload, dict):
        return {}
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return {}
    extracted: OpaqueReasoning = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        present = {
            field: message[field] for field in _OPAQUE_REQUEST_REASONING_FIELDS if field in message
        }
        if not present:
            continue
        if message.get("role") != "assistant":
            raise ValueError(  # noqa: TRY003
                "reasoning fields require an assistant message"
            )
        extracted[index] = present
        for field in present:
            del message[field]
    return extracted


def _restore_opaque_chat_reasoning(
    payload: dict[str, object], reasoning: OpaqueReasoning, *, offset: int
) -> None:
    """Reattach reasoning with an explicit guard offset, including PII bypass."""
    if not reasoning:
        return
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise InvalidEngineReplyError(  # noqa: TRY003
            "transformed Chat request has no messages"
        )
    for original_index, fields in reasoning.items():
        index = original_index + offset
        if index >= len(messages):
            raise InvalidEngineReplyError(  # noqa: TRY003
                "transformed Chat request changed reasoning message"
            )
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise InvalidEngineReplyError(  # noqa: TRY003
                "transformed Chat request changed reasoning message"
            )
        message.update(fields)


def _has_structured_chat_format(request: EngineChatRequest) -> bool:
    response_format = request.response_format
    return isinstance(response_format, dict) and response_format.get("type") in {
        "json_object",
        "json_schema",
    }


def _has_structured_responses_format(request: EngineResponsesRequest) -> bool:
    return (
        request.text is not None
        and request.text.format is not None
        and request.text.format.type in {"json_object", "json_schema"}
    )


def _presidio_code(reply: EngineReply) -> str:
    """Classify a successful analyzed model request into one stable response code."""
    if reply.decision == "reroute":
        return PRESIDIO_REROUTED
    if reply.decision == "apply_actions" and any(
        row.transformed_count for row in reply.report.rows
    ):
        return PRESIDIO_PII_TRANSFORMED
    if reply.entities:
        return PRESIDIO_PII_DETECTED
    return PRESIDIO_NO_PII


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def immediate_response(status: int, body: str) -> ext_proc_pb2.ProcessingResponse:
    """Build a JSON immediate response for malformed or blocked requests."""
    return ext_proc_pb2.ProcessingResponse(
        immediate_response=ext_proc_pb2.ImmediateResponse(
            status={"code": status},
            body=body,
            headers={
                "set_headers": [{"header": {"key": "content-type", "value": "application/json"}}]
            },
        )
    )


def request_mutation(
    body: bytes,
    headers: dict[str, str],
    mutate_body: bool,
) -> ext_proc_pb2.ProcessingResponse:
    """Build request header and optional body mutations with overwrite semantics."""
    set_headers = [
        {"header": {"key": key, "value": value}, "append_action": 2}
        for key, value in headers.items()
        if value
    ]
    header_mutation = {"set_headers": set_headers}
    body_mutation: dict[str, bytes] | None = None
    if mutate_body:
        body_mutation = {"body": body}
        header_mutation["set_headers"].append(
            {"header": {"key": "content-length", "value": str(len(body))}, "append_action": 2}
        )
    common = ext_proc_pb2.CommonResponse(header_mutation=header_mutation)
    if body_mutation is not None:
        common.body_mutation.CopyFrom(ext_proc_pb2.BodyMutation(body=body))
    response = ext_proc_pb2.ProcessingResponse(
        request_body=ext_proc_pb2.BodyResponse(response=common)
    )
    response.request_body.SetInParent()
    return response
