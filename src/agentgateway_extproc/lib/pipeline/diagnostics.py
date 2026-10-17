"""Full request and failure context for the existing DEBUG failure log."""

from __future__ import annotations

import json
import logging
import re
import traceback
from contextlib import suppress
from typing import TYPE_CHECKING

from neurwerk_request_segments import UnsupportedFeatureError
from pydantic import ValidationError

from agentgateway_extproc.models.exceptions import EnginePolicyError, GatewayPolicyError

if TYPE_CHECKING:
    from agentgateway_extproc.lib.pipeline.stream_handler import StreamHandler
    from agentgateway_extproc.models.engine import EngineReply

_logger = logging.getLogger(__name__)
_CREDENTIAL = re.compile(r"authorization|api.?key|token|password|secret|cookie", re.I)


def safe_diagnostic(  # noqa: C901
    value: object,
    handler: StreamHandler,
    exc: BaseException | None = None,
    *,
    input_data: object = None,
) -> str:
    """Redact credentials and available input without retaining another raw capture."""
    private: set[str] = set()

    def collect(item: object, *, credential: bool = False) -> None:
        pending = [(item, credential)]
        visited: set[int] = set()
        while pending:
            child, sensitive = pending.pop()
            if isinstance(child, str) and child and sensitive:
                private.add(child)
            elif isinstance(child, (dict, list, tuple)) and id(child) not in visited:
                visited.add(id(child))
                if isinstance(child, dict):
                    pending.extend(
                        (value, sensitive or bool(_CREDENTIAL.search(str(key))))
                        for key, value in child.items()
                    )
                else:
                    pending.extend((value, sensitive) for value in child)

    collect(input_data)
    for name, content in handler.request_headers.items():
        if _CREDENTIAL.search(name):
            collect(content, credential=True)
            for part in re.split(r"[\s;=]+", content):
                if part.lower() not in {"bearer", "basic"}:
                    collect(part, credential=True)
    for chunk in handler.request_body_chunks:
        with suppress(ValueError, UnicodeError, RecursionError):
            collect(json.loads(chunk))
    collect(handler.failed_response)
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ValidationError):
            for error in exc.errors(include_url=False):
                collect(error.get("input"))
        exc = exc.__cause__ or exc.__context__

    replacements = private | {json.dumps(secret, ensure_ascii=True)[1:-1] for secret in private}
    replacement_pattern = (
        re.compile(
            "|".join(re.escape(secret) for secret in sorted(replacements, key=len, reverse=True))
        )
        if replacements
        else None
    )

    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {key: clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if not isinstance(item, str):
            return item
        item = re.sub(r"(?i)\b(bearer|basic)\s+[^\s\"']+", r"\1 [redacted]", item)
        item = re.sub(
            r"(?i)((?:api[_-]?key|password|secret|token|cookie)[\"']?\s*[=:]\s*)"
            r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
            r"\1[redacted]",
            item,
        )
        if replacement_pattern is not None:
            # Never feed a replacement marker back through credential substitution.
            item = "[redacted]".join(
                replacement_pattern.sub("[redacted]", part) for part in item.split("[redacted]")
            )
        return item

    return diagnostic_json(clean(value))


def exception_diagnostic(
    exc: Exception, handler: StreamHandler, *, stack: bool = False, input_data: object = None
) -> str:
    """Describe the exception chain without Pydantic's raw input rendering."""
    chain: list[dict[str, object]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        detail: dict[str, object] = {"type": type(current).__name__}
        if isinstance(current, ValidationError):
            # Validator messages and ctx may interpolate arbitrary request content.
            detail["validation"] = {
                "errors": [
                    {"path": list(error["loc"]), "code": error["type"]}
                    for error in current.errors(include_input=False, include_url=False)
                ],
                "count": current.error_count(),
            }
        elif isinstance(current, (EnginePolicyError, GatewayPolicyError)):
            detail.update(code=current.code, message=current.message)
        if stack:
            detail["traceback"] = [
                {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_tb(current.__traceback__)
            ]
        chain.append(detail)
        current = current.__cause__ or current.__context__
    return safe_diagnostic(chain, handler, exc, input_data=input_data)


def exception_summary(exc: Exception) -> str:
    """Expose exception categories and validation locations without raw messages."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        parts.append(type(current).__name__)
        if isinstance(current, ValidationError):
            parts.extend(
                f"{error['type']} at {'.'.join(map(str, error['loc']))}"
                for error in current.errors(include_input=False, include_url=False)
            )
        current = current.__cause__ or current.__context__
    return "; ".join(parts)


def validation_rejection_message(message: str, exc: Exception, handler: StreamHandler) -> str:
    """Record locally handled request exceptions with the same safe diagnostics."""
    details = exception_diagnostic(exc, handler, stack=True)
    _logger.error(
        "request validation failed request_id=%s diagnostics=%s", handler.correlation_id, details
    )
    summary = details if handler.debug_errors else exception_summary(exc)
    return str(json.loads(safe_diagnostic(f"{message}: {summary}", handler, exc)))


def policy_block_message(
    reply: EngineReply, handler: StreamHandler, *, input_data: object = None
) -> str:
    """Preserve decision reasons, never request content or reversal values."""
    blocked = [row.entity_type for row in reply.report.rows if row.action == "block"]
    reasons = []
    if reply.safety_rule:
        reasons.append(f"safety rule {reply.safety_rule}")
    if blocked:
        reasons.append("reported blocking entity categories: " + ", ".join(blocked))
    if reply.analysis.cached_decision_applied:
        reasons.append("blocked by an earlier session policy decision (cached)")
    if reply.visual_findings is not None and reply.visual_findings.faces.scan_status == "failed":
        reasons.append("face scan failed; face detection result is unavailable")
    if not reply.safety_rule and not blocked:
        if reply.notices.response:
            reasons.extend(reply.notices.response)
        else:
            reasons.append(
                "the original reason is unavailable"
                if reply.analysis.cached_decision_applied
                else "specific cause unknown (reason not supplied by Engine)"
            )
    metadata = {
        "decision": reply.decision,
        "safety_rule": reply.safety_rule,
        "entities": reply.entities,
        "applied_actions": reply.applied_actions,
        "report": reply.report.model_dump(),
        "notices": reply.notices.model_dump(),
        "analysis": reply.analysis.model_dump(),
        "visual_findings": reply.visual_findings.model_dump() if reply.visual_findings else None,
    }
    details = safe_diagnostic(metadata, handler, input_data=input_data)
    _logger.error("policy block request_id=%s details=%s", handler.correlation_id, details)
    message = "request blocked by policy: " + "; ".join(reasons)
    message = str(json.loads(safe_diagnostic(message, handler, input_data=input_data)))
    if handler.debug_errors:
        message += ". Decision details: " + details
    return message


def diagnostic_json(value: object) -> str:
    """Serialize DEBUG context without redaction or additional payload limits."""
    return json.dumps(value, ensure_ascii=True, default=str)


def validation_details(exc: Exception) -> dict[str, object]:
    """Keep exact paths and indices alongside validation codes and messages."""
    if not isinstance(exc, ValidationError):
        return {"message": str(exc)}
    details = []
    for error in exc.errors(include_input=False, include_url=False):
        path = list(error["loc"])
        cause = error.get("ctx", {}).get("error")
        if isinstance(cause, UnsupportedFeatureError):
            path.append(cause.field)
        details.append({"path": path, "code": error["type"], "message": error["msg"]})
    return {"errors": details, "count": exc.error_count()}
