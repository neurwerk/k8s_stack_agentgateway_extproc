"""Validate request transport and attachments before optional engine policy."""

# ruff: noqa: C901

from __future__ import annotations

import json
import logging
import re
import secrets
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, cast

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
    parse_mcp_message,
    strict_json_loads,
)
from agentgateway_extproc.lib.session import make_session_key
from agentgateway_extproc.models.destination import ModelDestinationPolicy
from agentgateway_extproc.models.engine import (
    ENGINE_REQUEST_ADAPTER,
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
from agentgateway_extproc.models.exceptions import EngineUnavailableError, InvalidEngineReplyError
from agentgateway_extproc.models.types import (
    PRESIDIO_NO_PII,
    PRESIDIO_PII_DETECTED,
    PRESIDIO_PII_TRANSFORMED,
    PRESIDIO_REROUTED,
    RESERVED_PLACEHOLDER_PREFIX_RE,
    REVERSIBLE_CANDIDATE_RE,
    REVERSIBLE_TOKEN_RE,
    RequestStats,
)

if TYPE_CHECKING:
    from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler

type PathPart = str | int
type TextLeaves = dict[tuple[PathPart, ...], str]
type OpaqueReasoning = dict[int, dict[str, object]]

_TEXT_LEAF = object()
_OPAQUE_REQUEST_REASONING_FIELDS = ("reasoning_content", "reasoning_signature")
_MAX_VALIDATION_ERROR_COUNT = 100
_UNCHECKED_IMAGE_MARKER = "[Image forwarded without privacy inspection]"
_logger = logging.getLogger(__name__)


async def process_request(
    handler: StreamHandler, client: EngineClient
) -> ext_proc_pb2.ProcessingResponse:
    """Validate a request, call the engine, and apply its complete reply."""
    body = b"".join(handler.request_body_chunks)
    if len(body) > handler.max_request_bytes:
        return immediate_response(413, '{"error":"request body too large"}')
    try:
        payload = strict_json_loads(body.decode("utf-8"))
    except UnicodeDecodeError:
        return immediate_response(400, '{"error":"invalid request encoding"}')
    except JsonBudgetError:
        _clear_request(handler)
        return immediate_response(413, '{"error":"request structure too large"}')
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
            request = ENGINE_REQUEST_ADAPTER.validate_python(payload, strict=True)
        except ValidationError as exc:
            _log_model_validation_failure(payload, exc)
            handler.record_dispatch("protocol_failure")
            return immediate_response(400, '{"error":"invalid model request"}')
        except (TypeError, ValueError):
            _logger.warning(
                "model request validation failed family=%s reason=%s scope=%s count=%d",
                _model_request_family(payload),
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
        if attachments and policy.contract_version == 4:
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
                images = (
                    ImageBatch(
                        protect_faces=not unchecked and policy.protects_faces(request.model),
                        policy_version=4,
                    )
                    if image_indexes
                    else None
                )
                if unchecked:
                    images = cast(ImageBatch, images)
                    if document_indexes:
                        if handler.docling is None:
                            raise DocumentError(reason="extraction_unavailable")  # noqa: TRY301
                        texts_by_index = await handler.docling.convert_selected(
                            attachments, document_indexes, images=images
                        )
                    else:
                        if handler.docling is None:
                            raise DocumentError(reason="extraction_unavailable")  # noqa: TRY301
                        texts_by_index = await handler.docling.convert_selected(
                            attachments, set(), images=images
                        )
                    request, body, unchecked_image_locations = _converted_unchecked_request(
                        payload, texts_by_index, images, opaque_reasoning
                    )
                    unchecked_without_documents = not document_indexes
                else:
                    if handler.docling is None:
                        raise DocumentError(reason="extraction_unavailable")  # noqa: TRY301
                    texts = await handler.docling.convert(attachments, images=images)
                    if images is not None:
                        visual_findings = _visual_findings(attachments, images)
                        if image_forwarding != "none":
                            image_locations = _image_locations(payload, images)
                    request, body = _converted_request(payload, texts, opaque_reasoning)
            except DocumentError as exc:
                _clear_request(handler)
                handler.record_dispatch(
                    "policy_block" if exc.reason == "attachments_disabled" else "transport_failure"
                )
                return immediate_response(exc.status, json.dumps({"error": exc.message}))
            converted = True
            handler.request_body_chunks.clear()
            attachments.clear()
            payload = None
        elif attachments and attachment_mode != "passthrough":
            allowed = {"file", "input_file"}
            if policy.contract_version >= 2:
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
                        policy_version=policy.contract_version,
                    )
                    texts = await handler.docling.convert(attachments, images=images)
                    if policy.contract_version >= 3:
                        visual_findings = _visual_findings(attachments, images)
                    if image_forwarding != "none":
                        image_locations = _image_locations(payload, images)
                else:
                    texts = await handler.docling.convert(attachments)
                request, body = _converted_request(payload, texts, opaque_reasoning)
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
                and policy.contract_version >= 3
                and not unchecked_without_documents
            ):
                try:
                    image_output(policy, request.model, images, None)
                except DocumentError as exc:
                    return _image_policy_block(handler, exc)
            if image_locations:
                data = cast(dict[str, object], strict_json_loads(body.decode()))
                try:
                    _restore_images(data, image_locations, offset=0)
                    body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
                    if len(body) > handler.max_transformed_request_bytes:
                        raise DocumentError(413)  # noqa: TRY301
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
                        raise DocumentError(413)  # noqa: TRY301
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
                )
            else:
                reply = await client.analyze_request(request, session_key, document=True)
        else:
            reply = await client.analyze_request(request, session_key)
        _validate_request_mutation(request, reply)
        if handler.text_pii_enabled:
            _validate_reversal(request, reply)
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
        and policy.contract_version >= 3
    ):
        try:
            output = image_output(policy, request.model, images, reply)
            handler.request_stats.images_forwarded = output.forward_pixels
            if not output.forward_pixels:
                image_locations.clear()
            if output.notice:
                handler.notice_messages.append(output.notice)
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
        transformed.model_dump(mode="json", by_alias=True, exclude_none=True),
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
        raise ValueError("transformed request body too large")  # noqa: TRY003
    headers: dict[str, str] = {}
    if not is_mcp:
        headers = {
            "x-remote-allowed": str(reply.remote_allowed).lower(),
            "x-route-class": reply.route_class or "",
        }
        if reply.entities:
            headers["x-pii-entities"] = ",".join(reply.entities)
    return request_mutation(mutated, headers, converted or mutated != body)


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


def _image_policy_block(
    handler: StreamHandler, error: DocumentError | None = None
) -> ext_proc_pb2.ProcessingResponse:
    """Report the actual rejection reason without rewriting the engine's FACE action."""
    _clear_request(handler)
    handler.record_dispatch("policy_block")
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
    payload: dict[str, object] = {}
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
    payload: object, texts: list[str], reasoning: OpaqueReasoning
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
        ENGINE_REQUEST_ADAPTER.validate_python(data, strict=True),
    )
    if len(request.model_dump_json(by_alias=True, exclude_none=True).encode()) > MAX_REQUEST_BYTES:
        raise DocumentError(413)
    if sum(len(text) for text in _mutable_text_leaves(request).values()) > MAX_TEXT:
        raise DocumentError(413)
    _restore_opaque_chat_reasoning(data, reasoning, offset=0)
    body = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > MAX_REQUEST_BYTES:
        raise DocumentError(413)
    return request, body


def _converted_unchecked_request(
    payload: object,
    texts: dict[int, str],
    images: ImageBatch,
    reasoning: OpaqueReasoning,
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
        ENGINE_REQUEST_ADAPTER.validate_python(data, strict=True),
    )
    if sum(len(text) for text in _mutable_text_leaves(request).values()) > MAX_TEXT:
        raise DocumentError(413)
    body = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > MAX_REQUEST_BYTES:
        raise DocumentError(413)
    return request, body, locations


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


def _log_model_validation_failure(payload: object, exc: ValidationError) -> None:
    """Log only allowlisted aggregate facts about an invalid model request."""
    family = _model_request_family(payload)
    error_count = exc.error_count()
    if error_count > _MAX_VALIDATION_ERROR_COUNT:
        _logger.warning(
            "model request validation failed family=%s reason=%s scope=%s count=%d",
            family,
            "other",
            "other",
            _MAX_VALIDATION_ERROR_COUNT,
        )
        return
    family_model = {
        "chat": "EngineChatRequest",
        "responses": "EngineResponsesRequest",
    }.get(family)
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    if family_model is not None:
        selected = [
            error for error in errors if _validation_error_matches_model(error, family_model)
        ]
        if selected:
            errors = selected
    reasons = {_validation_reason(str(error.get("type", ""))) for error in errors}
    scopes = {_validation_scope(error.get("loc")) for error in errors}
    reason = next(iter(reasons)) if len(reasons) == 1 else "other"
    scope = next(iter(scopes)) if len(scopes) == 1 else "other"
    _logger.warning(
        "model request validation failed family=%s reason=%s scope=%s count=%d",
        family,
        reason,
        scope,
        len(errors),
    )


def _validation_error_matches_model(error: Mapping[str, object], marker: str) -> bool:
    """Recognize a union branch, including Pydantic's model-validator wrapper."""
    location = error.get("loc")
    if not isinstance(location, tuple) or not location:
        return False
    return _validation_model_branch(location[0], marker)


def _validation_model_branch(branch: object, marker: str) -> bool:
    """Return whether a location part identifies the expected model branch."""
    return branch == marker or (
        isinstance(branch, str)
        and branch.startswith("function-after[")
        and branch.endswith(f", {marker}]")
    )


def _model_request_family(payload: object) -> str:
    """Classify a request using only fixed protocol field probes."""
    if not isinstance(payload, dict):
        return "unknown"
    has_messages = "messages" in payload
    has_input = "input" in payload
    if has_messages and has_input:
        return "unknown"
    if has_messages:
        return "chat"
    if has_input:
        return "responses"
    return "unknown"


def _validation_reason(error_type: str) -> str:
    """Collapse Pydantic error types into a fixed diagnostic category."""
    if error_type == "extra_forbidden":
        return "extra_forbidden"
    if error_type == "missing":
        return "missing"
    if error_type.endswith("_type") or error_type in {
        "bool_parsing",
        "dict_type",
        "float_parsing",
        "int_parsing",
        "list_type",
        "string_type",
    }:
        return "invalid_type"
    if error_type in {"assertion_error", "literal_error", "value_error"}:
        return "invalid_value"
    return "other"


def _validation_scope(location: object) -> str:
    """Classify only known schema locations without exposing rejected field names."""
    if not isinstance(location, tuple):
        return "other"
    schema_location = location
    if location and any(
        _validation_model_branch(location[0], marker)
        for marker in (
            "EngineChatRequest",
            "EngineResponsesRequest",
            "EngineMcpRequest",
        )
    ):
        schema_location = location[1:]
    if "stream_options" in schema_location:
        return "stream_options"
    if "messages" in schema_location:
        return "messages"
    if "tools" in schema_location:
        return "tools"
    return "top_level" if len(schema_location) <= 1 else "other"


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


def _validate_reversal(original: EngineRequest, reply: EngineReply) -> None:
    """Restore reversible leaves once and require exact request provenance."""
    transformed_request = reply.request
    if RESERVED_PLACEHOLDER_PREFIX_RE.search(original.model_dump_json(exclude_none=True)):
        raise InvalidEngineReplyError(  # noqa: TRY003
            "reversal placeholder already existed in request"
        )
    if transformed_request is None:
        if reply.reversal:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "transformed request and reversal entries differ"
            )
        return
    original_leaves = _mutable_text_leaves(original)
    transformed_leaves = _mutable_text_leaves(transformed_request)
    entities_by_placeholder = _reversal_entities(reply)
    seen, restored_by_entity = _restore_reversal_leaves(
        original_leaves,
        transformed_leaves,
        reply.reversal,
        entities_by_placeholder,
    )
    if seen != set(reply.reversal):
        raise InvalidEngineReplyError(  # noqa: TRY003
            "transformed request and reversal entries differ"
        )
    if restored_by_entity != _expected_reversal_counts(reply):
        raise InvalidEngineReplyError(  # noqa: TRY003
            "reversal occurrence counts disagree with the current request report"
        )


def _restore_reversal_leaves(
    original_leaves: TextLeaves,
    transformed_leaves: TextLeaves,
    reversal: dict[str, str],
    entities_by_placeholder: dict[str, str],
) -> tuple[set[str], dict[str, int]]:
    """Scan candidate tokens once and validate plaintext against its source leaf."""
    seen: set[str] = set()
    restored_by_entity: dict[str, int] = {}
    consumed: dict[tuple[tuple[PathPart, ...], str], int] = {}
    current_path: tuple[PathPart, ...] = ()

    def restore(match: re.Match[str]) -> str:
        placeholder = match.group(0)
        plaintext = reversal.get(placeholder)
        if plaintext is None:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "transformed request and reversal entries differ"
            )
        seen.add(placeholder)
        entity_type = entities_by_placeholder[placeholder]
        restored_by_entity[entity_type] = restored_by_entity.get(entity_type, 0) + 1
        key = (current_path, plaintext)
        consumed[key] = consumed.get(key, 0) + 1
        return plaintext

    for path, transformed_text in transformed_leaves.items():
        current_path = path
        REVERSIBLE_CANDIDATE_RE.sub(restore, transformed_text)
    for (path, plaintext), count in consumed.items():
        if original_leaves.get(path, "").count(plaintext) < count:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "reversal plaintext does not match its original text leaf"
            )
    return seen, restored_by_entity


def _reversal_entities(reply: EngineReply) -> dict[str, str]:
    """Validate reversal keys against the current report in one linear pass."""
    if reply.analysis.source != "current_request":
        if reply.reversal:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "reversal entries require a current request report"
            )
        return {}
    rows = {row.entity_type: row for row in reply.report.rows}
    entities: dict[str, str] = {}
    for placeholder in reply.reversal:
        token = REVERSIBLE_TOKEN_RE.fullmatch(placeholder)
        if token is None:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "reversal contains an invalid placeholder"
            )
        entity_type = cast(str, token.group(2))
        row = rows.get(entity_type)
        if row is None or row.action not in {"encrypt", "reversible_replace"}:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "reversal entries disagree with the current request report"
            )
        if not row.transformed_count:
            raise InvalidEngineReplyError(  # noqa: TRY003
                "reversal entries require a transformed report row"
            )
        entities[placeholder] = entity_type
    return entities


def _expected_reversal_counts(reply: EngineReply) -> dict[str, int]:
    """Return placeholder counts required by current-request analysis."""
    if reply.analysis.source == "cached_decision":
        return {}
    return {
        row.entity_type: row.transformed_count
        for row in reply.report.rows
        if row.action in {"encrypt", "reversible_replace"} and row.transformed_count
    }


def _validate_request_mutation(original: EngineRequest, reply: EngineReply) -> None:
    """Reject engine mutations outside schema-designated model-visible text leaves."""
    if reply.request is None:
        return
    if type(reply.request) is not type(original) or _control_shape(reply.request) != _control_shape(
        original
    ):
        raise InvalidEngineReplyError(  # noqa: TRY003
            "engine reply changed request protocol controls"
        )


def _control_shape(request: EngineRequest) -> dict[str, object]:
    """Replace only mutable text leaves while retaining every protocol control."""
    data = cast(dict[str, object], request.model_dump(mode="python", exclude_none=True))
    if isinstance(request, EngineChatRequest):
        _normalize_chat_text(data)
    elif isinstance(request, EngineResponsesRequest):
        _normalize_responses_text(data)
    elif isinstance(request, EngineMcpRequest):
        _normalize_mcp_text(data)
    return data


def _normalize_chat_text(data: dict[str, object]) -> None:
    for message in _dict_list(data.get("messages")):
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _TEXT_LEAF
        elif isinstance(content, list):
            _normalize_content_parts(content, {"text"})
        for call in _dict_list(message.get("tool_calls")):
            function = call.get("function")
            if isinstance(function, dict):
                function["arguments"] = _normalize_json_text(function.get("arguments"))
    _normalize_tools(data.get("tools"))
    response_format = data.get("response_format")
    if isinstance(response_format, dict):
        data["response_format"] = _normalize_schema_text(response_format)


def _normalize_responses_text(data: dict[str, object]) -> None:
    if isinstance(data.get("instructions"), str):
        data["instructions"] = _TEXT_LEAF
    value = data.get("input")
    if isinstance(value, str):
        data["input"] = _TEXT_LEAF
    elif isinstance(value, list):
        for item in _dict_list(value):
            content = item.get("content")
            if item.get("type") == "message" and isinstance(content, list):
                _normalize_content_parts(content, {"input_text", "output_text"})
            elif item.get("type") == "function_call":
                item["arguments"] = _normalize_json_text(item.get("arguments"))
            elif item.get("type") == "function_call_output":
                item["output"] = _normalize_json_text(item.get("output"))
    text = data.get("text")
    if isinstance(text, dict) and isinstance(text.get("format"), dict):
        text["format"] = _normalize_schema_text(text["format"])
    _normalize_tools(data.get("tools"))


def _normalize_mcp_text(data: dict[str, object]) -> None:
    params = data.get("params")
    if not isinstance(params, dict):
        return
    if "arguments" in params:
        params["arguments"] = _normalize_json_text(params["arguments"])


def _normalize_content_parts(parts: list[object], text_types: set[str]) -> None:
    for part in parts:
        if isinstance(part, dict) and part.get("type") in text_types:
            part["text"] = _TEXT_LEAF


def _normalize_tools(value: object) -> None:
    for tool in _dict_list(value):
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        if isinstance(function.get("description"), str):
            function["description"] = _TEXT_LEAF
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            function["parameters"] = _normalize_schema_text(parameters)


def _normalize_schema_text(value: object) -> object:
    if not isinstance(value, dict):
        return value
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if key in {"description", "title", "default", "examples"}:
            normalized[key] = _normalize_json_text(item)
        elif key in {"schema", "json_schema", "items"}:
            normalized[key] = _normalize_schema_text(item)
        elif key in {"properties", "$defs"}:
            if isinstance(item, dict):
                normalized[key] = {
                    child_key: _normalize_schema_text(child) for child_key, child in item.items()
                }
            else:
                normalized[key] = item
        elif key in {"allOf", "anyOf", "oneOf"}:
            if isinstance(item, list):
                normalized[key] = [_normalize_schema_text(child) for child in item]
            else:
                normalized[key] = item
        else:
            normalized[key] = item
    return normalized


def _normalize_json_text(value: object) -> object:
    if isinstance(value, str):
        return _TEXT_LEAF
    if isinstance(value, list):
        return [_normalize_json_text(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_json_text(item) for key, item in value.items()}
    return value


def _mutable_text_leaves(request: EngineRequest) -> TextLeaves:
    """Return schema-designated model-visible strings keyed by structural path."""
    data = cast(dict[str, object], request.model_dump(mode="python", exclude_none=True))
    leaves: TextLeaves = {}
    _collect_text_leaves(data, _control_shape(request), (), leaves)
    return leaves


def _collect_text_leaves(
    value: object,
    normalized: object,
    path: tuple[PathPart, ...],
    leaves: TextLeaves,
) -> None:
    """Collect values replaced by the private control-shape sentinel."""
    if normalized is _TEXT_LEAF:
        if isinstance(value, str):
            leaves[path] = value
        return
    if isinstance(value, dict) and isinstance(normalized, dict):
        for key, child in normalized.items():
            _collect_text_leaves(value.get(key), child, (*path, str(key)), leaves)
        return
    if isinstance(value, list) and isinstance(normalized, list):
        for index, child in enumerate(normalized):
            _collect_text_leaves(value[index], child, (*path, index), leaves)


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
