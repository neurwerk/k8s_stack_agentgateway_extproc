"""Protect the reviewed Docling schema boundary and canonical text projection."""

import pytest

from agentgateway_extproc.lib.documents import (
    MAX_CELLS,
    MAX_TEXT,
    DocumentError,
    Upload,
    project_document,
)

from .test_documents import _document


def test_reviewed_schema_preserves_canonical_content():
    document = _document()
    text, pages = project_document(document, Upload("example.pdf", "pdf", b"", pages=1), 250)
    assert pages == 1
    assert text == (
        "Document: example.pdf\nJane Doe\nName | Value\nCaption\n[x]\n[ ]\n[ ] Optional\nFooter"
    )


def test_inline_runs_preserve_significant_whitespace():
    document = _document()
    document.update(
        tables=[],
        pictures=[],
        texts=[
            {"self_ref": f"#/texts/{index}", "text": run}
            for index, run in enumerate(["H", "2", "O", " is water"])
        ],
    )
    document["groups"][0].update(
        label="inline", children=[{"$ref": f"#/texts/{index}"} for index in range(4)]
    )
    document["furniture"]["children"] = []
    text, _ = project_document(document, Upload("example.docx", "docx", b"", pages=1), 250)
    assert text == "Document: example.docx\nH2O is water\n"


@pytest.mark.parametrize(
    ("case", "diagnostic"),
    [
        ("cycle", "reference_cycle"),
        ("dangling", "reference_invalid"),
        ("unvisited", "projection_incomplete"),
    ],
)
def test_reviewed_schema_rejects_incomplete_trees(case, diagnostic):
    document = _document()
    if case == "cycle":
        document["groups"][0]["children"].append({"$ref": "#/body"})
    elif case == "dangling":
        document["groups"][0]["children"].append({"$ref": "#/texts/999"})
    else:
        document["texts"].append({"self_ref": "#/texts/7", "text": "unvisited"})
    with pytest.raises(DocumentError) as caught:
        project_document(document, Upload("example.pdf", "pdf", b"", pages=1), 250)
    assert caught.value.diagnostic == diagnostic


@pytest.mark.parametrize("version", ["1.10.0", "1.11.1", "1.12.0"])
def test_other_schema_versions_are_rejected(version):
    document = _document()
    document["version"] = version
    with pytest.raises(DocumentError) as caught:
        project_document(document, Upload("example.pdf", "pdf", b"", pages=1), 250)
    assert caught.value.diagnostic == "schema_unsupported"


@pytest.mark.parametrize("same_origin", [False, True])
def test_overlapping_table_cells_preserve_each_text_in_reading_order(same_origin):
    document = _document()
    data = document["tables"][0]["data"]
    data.update(
        num_rows=1,
        num_cols=1 if same_origin else 4,
        table_cells=[
            {
                "start_row_offset_idx": 0,
                "end_row_offset_idx": 1,
                "start_col_offset_idx": 0 if same_origin else col,
                "end_col_offset_idx": 1 if same_origin else 4,
                "text": text,
            }
            # Reverse distinct origins to check row-major rather than source order.
            for col, text in [(3, "165"), (1, "Tensor networks and quantum magic")]
        ],
    )
    # The fixture's rich-cell child still participates in complete tree traversal.
    text, _ = project_document(document, Upload("example.pdf", "pdf", b"", pages=1), 250)
    expected = (
        "165\nTensor networks and quantum magic"
        if same_origin
        else " | Tensor networks and quantum magic |  | 165"
    )
    assert expected in text
    assert text.count("Tensor networks and quantum magic") == 1
    assert text.count("165") == 1
    assert text.count("Value") == 1


@pytest.mark.parametrize(
    ("case", "diagnostic"),
    [
        ("offset-type", "cell_offsets_invalid"),
        ("out-of-range", "cell_offsets_invalid"),
        ("empty-span", "cell_offsets_invalid"),
        ("grid-limit", "table_cell_limit"),
        ("entry-limit", "table_cell_limit"),
        ("invalid-text", "text_invalid"),
        ("invalid-ref", "reference_invalid"),
    ],
)
def test_table_projection_keeps_validation_and_resource_bounds(case, diagnostic):
    document = _document()
    data = document["tables"][0]["data"]
    cell = data["table_cells"][0]
    if case == "offset-type":
        cell["start_row_offset_idx"] = False
    elif case == "out-of-range":
        cell["end_col_offset_idx"] = data["num_cols"] + 1
    elif case == "empty-span":
        cell["end_row_offset_idx"] = cell["start_row_offset_idx"]
    elif case == "grid-limit":
        data["num_rows"] = MAX_CELLS + 1
    elif case == "entry-limit":
        data["table_cells"] = [cell] * (MAX_CELLS + 1)
    elif case == "invalid-text":
        cell["text"] = None
    else:
        cell["ref"] = {"$ref": "#/texts/999"}
    with pytest.raises(DocumentError) as caught:
        project_document(document, Upload("example.pdf", "pdf", b"", pages=1), 250)
    assert caught.value.diagnostic == diagnostic


@pytest.mark.parametrize("canonical", ["", " \n", "canonical formula"])
def test_formula_source_text_is_used_only_without_canonical_text(canonical):
    document = _document()
    document["texts"][0].update(label="formula", text=canonical, orig="Jane Doe + x")
    text, _ = project_document(document, Upload("example.pdf", "pdf", b"", pages=1), 250)
    expected = canonical if canonical.strip() else "[Source formula text] Jane Doe + x"
    assert expected in text
    assert ("Jane Doe + x" in text) == (not canonical.strip())
    assert ("[Source formula text]" in text) == (not canonical.strip())


@pytest.mark.parametrize(
    ("canonical", "original", "diagnostic"),
    [
        (None, "source", "text_item_invalid"),
        ("", None, "text_item_invalid"),
        ("", 123, "text_item_invalid"),
        ("", " \n", "text_item_invalid"),
        ("", "source\x00text", "text_invalid"),
        ("", "x" * (MAX_TEXT - len("[Source formula text] ")), "text_limit"),
    ],
)
def test_formula_fallback_preserves_text_validation_and_budget(canonical, original, diagnostic):
    document = _document()
    document["texts"][0].update(label="formula", text=canonical, orig=original)
    with pytest.raises(DocumentError) as caught:
        project_document(document, Upload("example.pdf", "pdf", b"", pages=1), 250)
    assert caught.value.diagnostic == diagnostic


def test_other_empty_text_nodes_do_not_use_original_copies():
    document = _document()
    document["texts"][0].update(label="text", text="", orig="original copy")
    text, _ = project_document(document, Upload("example.pdf", "pdf", b"", pages=1), 250)
    assert "original copy" not in text
    assert "[Source formula text]" not in text
