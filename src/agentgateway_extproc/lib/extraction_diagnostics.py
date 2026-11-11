"""Fixed extraction explanations and bounded, content-free log metadata."""

from __future__ import annotations

import hashlib
import logging
import re

_logger = logging.getLogger(__name__)

EXPLANATIONS = {
    "document_rejected": "Attachment processing rejected the request.",
    "admission_unavailable": "Extraction admission is disabled, busy, or closed.",
    "batch_timeout": "The attachment processing deadline expired.",
    "preflight_failed": "The attachment parser could not validate the upload.",
    "batch_text_limit": "Combined extracted text exceeds the processing limit.",
    "task_envelope_invalid": "Native task identity, type, or status is invalid.",
    "task_unsuccessful": "The native conversion task did not succeed.",
    "result_rejected": "Conversion result is incomplete, has errors, or lacks a document.",
    "transport_unavailable": "The extraction HTTP client is unavailable.",
    "http_status_rejected": "The extraction service returned a non-success HTTP status.",
    "http_encoding_rejected": "The extraction response uses an unsupported content encoding.",
    "http_response_limit": "The extraction response exceeds the configured byte limit.",
    "http_timeout": "The extraction HTTP request timed out.",
    "http_failed": "The extraction HTTP request failed.",
    "http_json_invalid": "The extraction response is not valid UTF-8 JSON.",
    "object_required": "The document schema requires an object at this position.",
    "list_required": "The document schema requires a list at this position.",
    "schema_unsupported": "The document schema name or version is unsupported.",
    "fields_unsupported": "The document contains populated unsupported fields.",
    "node_limit": "The document exceeds the supported node count.",
    "projection_incomplete": "The projection has unreachable nodes or no usable text.",
    "text_limit": "The document exceeds the extracted text limit.",
    "page_limit": "The document exceeds the remaining page limit.",
    "page_count_mismatch": "The extracted page count differs from the validated upload.",
    "page_number_invalid": "The document contains invalid or inconsistent page numbers.",
    "page_sequence_invalid": "The document page sequence is incomplete.",
    "text_invalid": "Canonical text is missing, invalid, or contains a null character.",
    "text_item_invalid": "A text item lacks canonical text or has an empty formula.",
    "table_dimensions_invalid": "Table dimensions must be positive integers.",
    "table_cell_limit": "The document exceeds the supported table cell count.",
    "table_cells_invalid": "Table cells are missing or exceed the table dimensions.",
    "table_overlap": "Table cells overlap.",
    "reference_invalid": "A document reference is invalid or unresolved.",
    "reference_cycle": "Document references contain a cycle.",
    "reference_limit": "Document references exceed depth or traversal limits.",
    "self_reference_invalid": "A document node has an inconsistent self reference.",
    "cell_offsets_invalid": "Table cell offsets are invalid or outside the table.",
}
_COUNTS = {
    "expected_pages",
    "actual_pages",
    "page_limit",
    "error_count",
    "http_status",
    "node_count",
    "visited_nodes",
    "response_limit",
}
_FLAGS = {
    "document_present",
    "errors_present",
    "schema_name_supported",
    "schema_version_supported",
    "has_text",
    "unsupported_fields_present",
}
_STATUSES = {"pending", "started", "success", "partial_success", "failure", "skipped"}
VLM_EXPLANATIONS = {
    "vlm_output_token_limit": (
        "Document conversion stopped because the VLM output token limit was reached."
    ),
    "vlm_content_filtered": (
        "Document conversion stopped because the VLM output was content-filtered."
    ),
}


def native_error_metadata(errors: object) -> dict[str, str | int | bool]:
    """Describe the first native error using only recognized, content-free values."""
    if not isinstance(errors, list) or not errors or not isinstance(errors[0], dict):
        return {}
    error = errors[0]
    values = {"category": error.get("category"), "page_no": error.get("page_no")}
    message = error.get("error_message")
    for stop_reason, code in (
        ("length", "vlm_output_token_limit"),
        ("content_filter", "vlm_content_filtered"),
    ):
        if message == f"VLM output incomplete (stop_reason={stop_reason}).":
            values.update(
                stop_reason="length" if stop_reason == "length" else "content_filtered",
                error_code=code,
            )
            break
    return safe_metadata(values)


def safe_metadata(values: dict[str, object]) -> dict[str, str | int | bool]:
    """Discard arbitrary values; never retain upstream payloads or exception strings."""
    result: dict[str, str | int | bool] = {}
    for key, value in values.items():
        if (key in _COUNTS and type(value) is int and 0 <= value <= 2**31 - 1) or (
            key in _FLAGS and type(value) is bool
        ):
            result[key] = value
        elif key == "native_status":
            result[key] = value if isinstance(value, str) and value in _STATUSES else "unknown"
        elif (key == "page_no" and type(value) is int and 1 <= value <= 1000) or (
            isinstance(value, str)
            and (
                (key == "category" and value == "inference_failure")
                or (key == "stop_reason" and value in {"length", "content_filtered"})
                or (key == "error_code" and value in VLM_EXPLANATIONS)
            )
        ):
            result[key] = value
        elif key == "task_id" and isinstance(value, str) and len(value) <= 128:
            # Correlate native jobs without trusting their identifier as content.
            result["task_id_hash"] = hashlib.sha256(value.encode()).hexdigest()[:16]
    return result


def log_failure(code: str, metadata: dict[str, str | int | bool], request_id: str | None) -> None:
    """Log a fixed explanation with an existing validated request correlation ID."""
    correlation = (
        request_id
        if request_id is not None and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id)
        else "unavailable"
    )
    _logger.warning(
        "extraction rejected request_id=%s diagnostic_code=%s explanation=%s metadata=%s",
        correlation,
        code,
        VLM_EXPLANATIONS.get(str(metadata.get("error_code")), EXPLANATIONS[code]),
        metadata,
        extra={"extraction_code": code, "extraction_metadata": metadata, "request_id": correlation},
    )
