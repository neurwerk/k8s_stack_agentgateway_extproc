"""Test the strict detailed-report contract and its PII-free renderer."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentgateway_extproc.lib.notice.analysis import render_analysis_notice
from agentgateway_extproc.lib.notice.inject import render_notice
from agentgateway_extproc.lib.notice.report import render_report
from agentgateway_extproc.models.engine import AnalysisMetadata, PIIReport


def _analysis(
    *,
    source: str = "current_request",
    cached: bool = False,
    scan_performed: bool = True,
    duration_ms: int = 3_200,
    overlap_count: int = 0,
) -> AnalysisMetadata:
    return AnalysisMetadata.model_validate(
        {
            "source": source,
            "scan_performed": scan_performed,
            "duration_ms": duration_ms if scan_performed else None,
            "overlap_count": overlap_count,
            "overlap_resolution": "strictest_action",
            "policy_version": "test",
            "text_leaf_count": 1 if scan_performed else 0,
            "cached_decision_applied": cached,
        }
    )


@pytest.mark.parametrize(
    ("action", "transformed", "unique", "wording"),
    [
        ("pass", 0, 0, "`pass`: 3 detected"),
        ("block", 0, 0, "`block`: 3 detected"),
        ("mask", 2, 1, "`mask`: 3 detected; 2 transformed (1 unique)"),
        ("replace", 2, 1, "`replace`: 3 detected; 2 transformed (1 unique)"),
        ("redact", 2, 1, "`redact`: 3 detected; 2 transformed (1 unique)"),
        ("hash", 2, 1, "`hash`: 3 detected; 2 transformed (1 unique)"),
        ("encrypt", 2, 1, "`encrypt`: 3 detected; 2 transformed (1 unique)"),
        (
            "reversible_replace",
            2,
            1,
            "`reversible_replace`: 3 detected; 2 transformed (1 unique)",
        ),
        ("reroute", 0, 0, "`reroute`: 3 detected; forwarded without masking"),
        ("reroute", 2, 1, "`reroute`: 3 detected; 2 masked (1 unique)"),
    ],
)
def test_report_renderer_distinguishes_every_request_action(
    action: str, transformed: int, unique: int, wording: str
) -> None:
    report = _report(action, transformed, unique)

    rendered = render_report(
        report,
        _analysis(),
        {"PERSON": 2},
        decision="reroute" if action == "reroute" else "apply_actions",
        route_class="local-sensitive",
    )

    assert "| Entity | Request | Response |" in rendered
    assert f"| Person | {wording} | 2 restored |" in rendered
    assert "Jane Doe" not in rendered


@pytest.mark.parametrize(
    ("source", "cached", "provenance"),
    [
        ("current_request", False, ""),
        (
            "cached_decision",
            True,
            "Entity rows describe the cached policy decision; current-request PII analysis "
            "was skipped.",
        ),
        (
            "current_request",
            True,
            "Routing includes a cached policy decision; entity rows describe the current request.",
        ),
    ],
    ids=["fresh", "cached", "sticky"],
)
def test_report_renderer_exposes_fresh_cached_and_sticky_provenance(
    source: str, cached: bool, provenance: str
) -> None:
    report = PIIReport.model_validate({"rows": []})
    analysis = _analysis(
        source=source,
        cached=cached,
        scan_performed=source == "current_request",
    )

    rendered = render_report(report, analysis, {}, decision="pass", route_class=None)

    if provenance:
        assert provenance in rendered
    else:
        assert "cached policy decision" not in rendered


def test_clean_report_uses_existing_notice_message_without_an_empty_table() -> None:
    report = PIIReport(rows=[])

    notice = render_notice(
        ["No sensitive data was detected."],
        render_report(report, _analysis(), {}, decision="pass", route_class="general"),
    )

    assert notice.count("PII Engine Notice") == 1
    assert "No sensitive data was detected." in notice
    assert "general" not in notice
    assert "| Entity |" not in notice
    assert "response was scanned" not in notice


def test_report_table_is_separated_from_generic_notice_text() -> None:
    notice = render_notice(
        ["Protected"],
        render_report(
            _report("mask", 2, 1),
            _analysis(),
            {},
            decision="apply_actions",
            route_class=None,
        ),
    )

    assert "Protected\n\nPII scan completed in 3.2 seconds." in notice
    assert "| Entity | Request | Response |" in notice


@pytest.mark.parametrize(
    "row_update",
    [
        {"detected_count": 0},
        {"transformed_count": 4},
        {"unique_transformed_count": 3},
        {"action": "pass", "transformed_count": 1, "unique_transformed_count": 1},
        {"action": "block", "transformed_count": 1, "unique_transformed_count": 1},
        {"detected_count": 10_000_001},
        {"unexpected": True},
    ],
)
def test_report_rows_reject_invalid_counts_and_extra_fields(row_update: dict[str, object]) -> None:
    row = _row("PERSON", "mask", 2, 1)
    row.update(row_update)
    with pytest.raises(ValidationError):
        PIIReport.model_validate({"rows": [row]})


def test_report_requires_sorted_unique_bounded_rows_and_forbids_extra_fields() -> None:
    first = _row("EMAIL_ADDRESS", "pass", 0, 0)
    second = _row("PERSON", "pass", 0, 0)
    for rows in ([second, first], [first, first]):
        with pytest.raises(ValidationError):
            PIIReport.model_validate({"rows": rows})
    with pytest.raises(ValidationError):
        PIIReport.model_validate(
            {
                "rows": [_row(f"ENTITY_{index:02d}", "pass", 0, 0) for index in range(65)],
            }
        )
    with pytest.raises(ValidationError):
        PIIReport.model_validate(
            {
                "rows": [],
                "unexpected": True,
            }
        )


@pytest.mark.parametrize(
    ("analysis", "expected"),
    [
        (_analysis(duration_ms=320), "PII scan completed in 0.3 seconds."),
        (
            _analysis(duration_ms=3_200, overlap_count=1),
            "1 item was identified for multiple PII entities",
        ),
        (
            _analysis(duration_ms=3_200, overlap_count=4),
            "4 items were identified for multiple PII entities",
        ),
        (
            _analysis(source="cached_decision", cached=True, scan_performed=False),
            "",
        ),
    ],
)
def test_analysis_notice_formats_current_facts_without_fabricating_cached_duration(
    analysis: AnalysisMetadata, expected: str
) -> None:
    rendered = render_analysis_notice(analysis)
    assert expected in rendered
    if not analysis.scan_performed:
        assert rendered == ""


def _report(action: str, transformed: int, unique: int) -> PIIReport:
    return PIIReport.model_validate({"rows": [_row("PERSON", action, transformed, unique)]})


def _row(entity_type: str, action: str, transformed: int, unique: int) -> dict[str, object]:
    return {
        "entity_type": entity_type,
        "action": action,
        "detected_count": 3,
        "transformed_count": transformed,
        "unique_transformed_count": unique,
    }
