"""Pure rendering for bounded PII analysis metadata."""

from __future__ import annotations

from agentgateway_extproc.models.engine import AnalysisMetadata


def render_analysis_notice(analysis: AnalysisMetadata) -> str:
    """Render current scan timing and overlap facts without parsing policy prose."""
    if not analysis.scan_performed or analysis.duration_ms is None:
        return ""
    duration = analysis.duration_ms / 1_000
    message = f"PII scan completed in {duration:.1f} seconds."
    if analysis.overlap_count == 1:
        return (
            f"{message} 1 item was identified for multiple PII entities; "
            "the strictest action was applied."
        )
    if analysis.overlap_count:
        return (
            f"{message} {analysis.overlap_count} items were identified for multiple PII entities; "
            "the strictest action was applied."
        )
    return message
