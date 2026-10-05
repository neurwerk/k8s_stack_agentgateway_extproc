"""Canonical lightweight request extraction API."""

from .compatibility import (
    CompatibilitySettings,
    ControlRule,
    ExtractionLimitError,
    UnsupportedFeatureError,
)
from .extraction import ExtractedRequest, TextSegment, extract_request, parse_request
from .models import SupportedRequest

__all__ = [
    "CompatibilitySettings",
    "ControlRule",
    "ExtractedRequest",
    "ExtractionLimitError",
    "SupportedRequest",
    "TextSegment",
    "UnsupportedFeatureError",
    "extract_request",
    "parse_request",
]
