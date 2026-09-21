"""Pure rendering for the engine's detailed PII report."""

from __future__ import annotations

from agentgateway_extproc.lib.notice.analysis import render_analysis_notice
from agentgateway_extproc.models.engine import (
    AnalysisMetadata,
    PIIReport,
    PIIReportRow,
    VisualFindings,
)


def render_report(
    report: PIIReport,
    analysis: AnalysisMetadata,
    restored_counts: dict[str, int],
    *,
    decision: str,
    route_class: str | None,
    visual_findings: VisualFindings | None = None,
    text_pii_enabled: bool = True,
    images_forwarded: bool | None = None,
) -> str:
    """Render provenance and per-entity counts without including literal PII."""
    lines: list[str] = []
    if operational := render_analysis_notice(analysis):
        lines.append(operational)
    if visual_findings is not None:
        faces = visual_findings.faces
        if not text_pii_enabled:
            lines.append("Text PII analysis was disabled.")
        lines.append(
            f"Face scan completed: {faces.count} detected."
            if faces.scan_status == "complete"
            else "Faces were not scanned."
            if faces.scan_status == "not_scanned"
            else "Face scan failed."
        )
    if analysis.source == "cached_decision":
        lines.append(
            "Entity rows describe the cached policy decision; current-request PII analysis "
            "was skipped."
        )
    elif analysis.cached_decision_applied:
        lines.append(
            "Routing includes a cached policy decision; entity rows describe the current request."
        )
    if decision == "reroute" and route_class:
        lines.append(f"Effective route: `{route_class}`.")
    if report.rows:
        lines.extend(
            [
                "| Entity | Request | Response |",
                "| --- | --- | --- |",
                *[
                    _render_row(
                        row,
                        restored_counts.get(row.entity_type, 0),
                        decision=decision,
                        images_forwarded=images_forwarded,
                    )
                    for row in report.rows
                ],
            ]
        )
    return "\n".join(lines)


def _render_row(
    row: PIIReportRow, restored_count: int, *, decision: str, images_forwarded: bool | None
) -> str:
    request = _request_cell(row, blocked=decision == "block", images_forwarded=images_forwarded)
    response = f"{restored_count} restored" if restored_count else "-"
    return f"| {_label(row.entity_type)} | {request} | {response} |"


def _request_cell(row: PIIReportRow, *, blocked: bool, images_forwarded: bool | None) -> str:
    prefix = f"`{row.action}`: {row.detected_count} detected"
    if blocked and row.action in {"reroute", "text-only"}:
        return f"{prefix}; not forwarded (request blocked)"
    if row.entity_type == "FACE":
        if row.action == "reroute":
            if images_forwarded is False:
                return (
                    f"{prefix}; images withheld; extracted text forwarded to approved local model"
                )
            return f"{prefix}; images forwarded to approved local model without masking"
        if row.action == "text-only":
            return f"{prefix}; all request images withheld; extracted text only"
        return f"{prefix}; images blocked"
    if row.action == "reroute":
        if not row.transformed_count:
            return f"{prefix}; forwarded without masking"
        return f"{prefix}; {row.transformed_count} masked ({row.unique_transformed_count} unique)"
    if not row.transformed_count:
        return prefix
    return f"{prefix}; {row.transformed_count} transformed ({row.unique_transformed_count} unique)"


def _label(entity_type: str) -> str:
    return entity_type.lower().replace("_", " ").title()
