"""Translate shared extraction limit facts at the adapter boundary."""

from neurwerk_request_segments import ExtractedRequest, ExtractionLimitError, SupportedRequest
from neurwerk_request_segments import extract_request as extract_segments

from agentgateway_extproc.models.exceptions import EnginePolicyError, LimitDetail


def extract_request(request: SupportedRequest) -> ExtractedRequest:
    """Extract content without losing the source or precision of a measured limit."""
    try:
        return extract_segments(request)
    except ExtractionLimitError as exc:
        raise EnginePolicyError(
            "request_too_large",
            limit=LimitDetail.model_validate(
                {
                    "component": exc.component,
                    "stage": exc.stage,
                    "reason": exc.reason,
                    "measured": exc.measured,
                    "maximum": exc.maximum,
                    "unit": exc.unit,
                    "exact": exc.exact,
                }
            ),
        ) from exc
    except (TypeError, ValueError, RecursionError) as exc:
        raise EnginePolicyError("invalid_request") from exc
