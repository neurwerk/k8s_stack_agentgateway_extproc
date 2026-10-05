"""Content-free client errors and correlated request-limit diagnostics."""

from __future__ import annotations

import json
import logging

from agentgateway_extproc.gen import ext_proc_pb2
from agentgateway_extproc.models.exceptions import LimitDetail

_logger = logging.getLogger(__name__)


def log_limit_rejection(correlation_id: str, limit: LimitDetail) -> None:
    """Record a limit even when streaming headers prevent a client error body."""
    _logger.warning(
        "limit rejection request_id=%s component=%s stage=%s reason=%s "
        "measured=%d maximum=%d unit=%s exact=%s",
        correlation_id,
        limit.component,
        limit.stage,
        limit.reason,
        limit.measured,
        limit.maximum,
        limit.unit,
        limit.exact,
    )


def annotate_rejection(
    response: ext_proc_pb2.ProcessingResponse,
    correlation_id: str,
    *,
    limit: LimitDetail | None = None,
) -> None:
    """Normalize a local model-endpoint rejection without exposing request content."""
    if not response.HasField("immediate_response"):
        return
    immediate = response.immediate_response
    parsed = json.loads(immediate.body)
    raw_error = parsed.get("error")
    error: dict[str, object]
    if isinstance(raw_error, str):
        error = {"message": raw_error}
    elif isinstance(raw_error, dict):
        error = dict(raw_error)
    else:
        error = {"message": "Request processing failed."}
    error.setdefault(
        "type", "invalid_request_error" if immediate.status.code < 500 else "server_error"
    )
    error.setdefault("param", None)
    error.setdefault(
        "code",
        "request_too_large"
        if immediate.status.code == 413
        else "policy_blocked"
        if immediate.status.code == 403
        else "invalid_request"
        if immediate.status.code < 500
        else "processing_failed",
    )
    error["request_id"] = correlation_id
    if limit is not None:
        error["limit"] = limit.model_dump()
        qualifier = "" if limit.exact else "at least "
        label = {
            "segments": "text segments",
            "text_leaves": "text segments",
            "text_characters": "text characters",
            "depth": "nesting levels",
            "tokens": "structural items",
            "nodes": "structural items",
        }.get(limit.reason, limit.reason.replace("_", " "))
        error["message"] = (
            f"Limit exceeded: {qualifier}{limit.measured} {label}; maximum is {limit.maximum}."
        )
        log_limit_rejection(correlation_id, limit)
    parsed["error"] = error
    immediate.body = json.dumps(parsed, ensure_ascii=True, separators=(",", ":"))
    immediate.headers.set_headers.add(
        header={"key": "x-request-id", "value": correlation_id}, append_action=2
    )
