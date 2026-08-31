"""Stream state types and placeholder patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentgateway_extproc.models.engine import AnalysisMetadata, PIIReport

REVERSIBLE_TOKEN_RE = re.compile(
    r"<(REV|ENCRYPTED)_([A-Z][A-Z0-9_]*)_([0-9a-f]{16})_([0-9a-f]{16})>"
)
REVERSIBLE_CANDIDATE_RE = re.compile(r"<(?:REV|ENCRYPTED)_[^<>\r\n]{1,256}>")
RESERVED_PLACEHOLDER_PREFIX_RE = re.compile(r"<(?:REV|ENCRYPTED)_")

# Stable response codes: no PII, detected/pass, transformed/masked, and rerouted.
PRESIDIO_RESPONSE_HEADER = "x-presidio-code"
PRESIDIO_NO_PII = "P00"
PRESIDIO_PII_DETECTED = "P01"
PRESIDIO_PII_TRANSFORMED = "P02"
PRESIDIO_REROUTED = "P03"

REQUEST_HEADERS = {
    "x-remote-allowed",
    "x-route-class",
    "x-pii-entities",
    "x-pii-session-key",
    PRESIDIO_RESPONSE_HEADER,
}


@dataclass
class RequestStats:
    """Retain the engine report and routing provenance for the response stream."""

    report: PIIReport
    analysis: AnalysisMetadata
    decision: str = "pass"
    route_class: str | None = None
