"""Test reversal, notice, and chunk-boundary utilities."""

from __future__ import annotations

import pytest

from agentgateway_extproc.lib.masking.reversal import (
    PlaceholderStreamRewriter,
    invalid_placeholder_marker,
    reverse_placeholders,
)
from agentgateway_extproc.lib.notice.inject import render_notice
from agentgateway_extproc.lib.pipeline.response import split_safe_prefix
from agentgateway_extproc.models.types import REVERSIBLE_CANDIDATE_RE

from .conftest import REVERSIBLE_TOKEN


def test_reversal_reports_hits_and_missing_placeholders() -> None:
    """Known values reverse while unknown values remain visible and counted."""
    unknown = "<REV_PERSON_aaaaaaaaaaaaaaaa_bbbbbbbbbbbbbbbb>"
    encrypted = "<ENCRYPTED_SECRET_cccccccccccccccc_dddddddddddddddd>"
    text, hits, misses = reverse_placeholders(
        f"{REVERSIBLE_TOKEN} {unknown} {encrypted}",
        {REVERSIBLE_TOKEN: "Jane Doe", encrypted: "secret"},
    )
    assert text == f"Jane Doe {unknown} secret"
    assert hits == {"PERSON": 1, "SECRET": 1}
    assert misses == 1


def test_malformed_reversible_marker_is_never_mapped() -> None:
    """A token with an altered digest is treated as an unknown response value."""
    altered = REVERSIBLE_TOKEN.replace("fedc", "zzzz")
    text, hits, misses = reverse_placeholders(altered, {REVERSIBLE_TOKEN: "Jane Doe"})
    assert text == altered
    assert hits == {}
    assert misses == 1


def test_invalid_human_placeholder_gets_a_history_safe_marker() -> None:
    """A modified model token remains diagnosable without poisoning later chat history."""
    altered = "<REV_PERSON_0123456789abcdef_...>"
    marker = invalid_placeholder_marker(altered, {REVERSIBLE_TOKEN: "Jane Doe"})

    text, hits, misses = reverse_placeholders(
        f"before {altered} after",
        {REVERSIBLE_TOKEN: "Jane Doe"},
        mark_invalid=True,
    )

    assert marker.startswith("<PII_INVALID_PERSON_")
    assert not REVERSIBLE_CANDIDATE_RE.search(marker)
    assert text == f"before {marker} after"
    assert hits == {}
    assert misses == 1


def test_split_safe_prefix_holds_incomplete_token() -> None:
    """A trailing opening bracket is held for the next response chunk."""
    assert split_safe_prefix("before <REV_PERSON_") == ("before ", "<REV_PERSON_")
    assert split_safe_prefix(f"complete {REVERSIBLE_TOKEN}") == (
        f"complete {REVERSIBLE_TOKEN}",
        "",
    )
    assert split_safe_prefix("comparison < 10") == ("comparison < 10", "")


def test_notice_rendering_omits_empty_messages() -> None:
    """Notice content remains bounded structural metadata until a format selects its target."""
    notice = render_notice(["Protected", "  "])
    assert notice == "\n\n---\nPII Engine Notice\nProtected"


def test_stream_rewriter_keeps_only_an_incomplete_placeholder_suffix() -> None:
    """Response chunks retain no plaintext until a complete request-scoped token is available."""
    rewriter = PlaceholderStreamRewriter({REVERSIBLE_TOKEN: 'A "quoted" value'})
    first, hits, misses = rewriter.feed(f"before {REVERSIBLE_TOKEN[:15]}")
    assert first == "before "
    assert hits == {}
    assert misses == 0
    final, hits, misses = rewriter.feed(REVERSIBLE_TOKEN[15:], final=True)
    assert final == 'A "quoted" value'
    assert hits == {"PERSON": 1}
    assert misses == 0


def test_human_stream_rewriter_marks_an_incomplete_placeholder_at_eof() -> None:
    rewriter = PlaceholderStreamRewriter({REVERSIBLE_TOKEN: "Jane Doe"}, mark_invalid=True)
    output, hits, misses = rewriter.feed("before <REV_PERSON_012345", final=True)

    assert output.startswith("before <PII_INVALID_PERSON_")
    assert hits == {}
    assert misses == 1


def test_malformed_placeholder_before_markup_is_marked_or_rejected() -> None:
    malformed = "before <REV_PERSON_broken and <b>after</b>"
    human = PlaceholderStreamRewriter({REVERSIBLE_TOKEN: "Jane Doe"}, mark_invalid=True)
    strict = PlaceholderStreamRewriter({REVERSIBLE_TOKEN: "Jane Doe"})

    output, hits, misses = human.feed(malformed, final=True)

    assert output.startswith("before <PII_INVALID_PERSON_")
    assert output.endswith("><b>after</b>")
    assert hits == {}
    assert misses == 1
    with pytest.raises(ValueError, match="malformed placeholder"):
        strict.feed(malformed, final=True)


@pytest.mark.parametrize("malformed", ["<REV_>", f"<REV_{'x' * 257}>"])
def test_closed_out_of_grammar_placeholder_is_marked_or_rejected(malformed: str) -> None:
    human = PlaceholderStreamRewriter({REVERSIBLE_TOKEN: "Jane Doe"}, mark_invalid=True)
    strict = PlaceholderStreamRewriter({REVERSIBLE_TOKEN: "Jane Doe"})

    output, hits, misses = human.feed(f"before {malformed} after", final=True)

    assert output.startswith("before <PII_INVALID_TOKEN_")
    assert output.endswith("> after")
    assert hits == {}
    assert misses == 1
    with pytest.raises(ValueError, match="malformed placeholder"):
        strict.feed(malformed, final=True)
