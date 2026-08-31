"""Render bounded engine notice text for structural response processors."""

from __future__ import annotations


def render_notice(messages: list[str], report: str = "") -> str:
    """Combine non-empty engine messages and a report under the existing heading."""
    sections = [message.strip() for message in messages if message.strip()]
    if rendered_report := report.strip():
        sections.append(rendered_report)
    return "\n\n---\nPII Engine Notice\n" + "\n\n".join(sections) if sections else ""
