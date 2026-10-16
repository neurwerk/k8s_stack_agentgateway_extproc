"""Full request and failure context for the existing DEBUG failure log."""

from __future__ import annotations

import json

from neurwerk_request_segments import UnsupportedFeatureError
from pydantic import ValidationError


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
