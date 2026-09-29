"""Pure rendering for the engine's detailed PII report."""

from __future__ import annotations

from agentgateway_extproc.lib.notice.analysis import render_analysis_notice
from agentgateway_extproc.lib.notice.preferences import DEFAULT_PREFERENCES, NoticePreferences
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
    preferences: NoticePreferences = DEFAULT_PREFERENCES,
) -> str:
    """Render provenance and per-entity counts without including literal PII."""
    lines: list[str] = []
    if preferences.show_timing and (operational := render_analysis_notice(analysis)):
        lines.append(operational)
    if visual_findings is not None:
        faces = visual_findings.faces
        show_face_status = (
            (faces.count or 0) > 0 or faces.scan_status == "failed" or preferences.show_no_pii
        )
        if not text_pii_enabled and show_face_status:
            lines.append("Text PII analysis was disabled.")
        if show_face_status:
            lines.append(
                f"Face scan completed: {faces.count} detected."
                if faces.scan_status == "complete"
                else "Faces were not scanned."
                if faces.scan_status == "not_scanned"
                else "Face scan failed."
            )
    rows = [
        row
        for row in report.rows
        if row.entity_type == "FACE"
        or (row.action == "pass" and preferences.show_pass)
        or (row.action == "reroute" and preferences.show_reroutes)
        or (row.action not in {"pass", "reroute"} and preferences.show_changes)
    ]
    show_result = bool(rows) or (not report.rows and decision == "pass" and preferences.show_no_pii)
    if show_result and analysis.source == "cached_decision":
        lines.append(
            "Entity rows describe the cached policy decision; current-request PII analysis "
            "was skipped."
        )
    elif show_result and analysis.cached_decision_applied and preferences.show_reroutes:
        lines.append(
            "Routing includes a cached policy decision; entity rows describe the current request."
        )
    if decision == "reroute" and route_class and preferences.show_reroutes:
        lines.append(f"Effective route: `{route_class}`.")
    if rows:
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
                    for row in rows
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
