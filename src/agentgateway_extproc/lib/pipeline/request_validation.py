"""Log bounded validation facts without exposing rejected request content."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from pydantic import ValidationError

_MAX_VALIDATION_ERROR_COUNT = 100
_logger = logging.getLogger(__name__)


def log_model_validation_failure(payload: object, exc: ValidationError) -> None:
    """Log only allowlisted aggregate facts about an invalid model request."""
    family = model_request_family(payload)
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


def model_request_family(payload: object) -> str:
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
    """Classify known schema locations without exposing rejected field names."""
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
