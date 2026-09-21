"""Select pixels or policy-approved extracted text from complete local image facts."""

from __future__ import annotations

from dataclasses import dataclass

from agentgateway_extproc.lib.documents import DocumentError, ImageBatch
from agentgateway_extproc.models.destination import ModelDestinationPolicy
from agentgateway_extproc.models.engine import EngineReply


@dataclass(frozen=True)
class ImageOutput:
    """Retain the actual output choice and an optional successful-processing notice."""

    forward_pixels: bool
    notice: str = ""


def image_output(
    policy: ModelDestinationPolicy, model: str, images: ImageBatch, reply: EngineReply | None
) -> ImageOutput:
    """Apply terminal and face policy first, then the model's forwarding mode."""
    forwarding = policy.image_forwarding.get(model, "none")
    has_text = bool(images.text_present) and all(images.text_present.values())
    face_action = (
        next((row.action for row in reply.report.rows if row.entity_type == "FACE"), None)
        if reply is not None
        else None
    )
    if reply is not None and reply.decision == "block":
        raise DocumentError(
            403, reason=("face_policy_blocked" if face_action == "block" else "policy_blocked")
        )
    if face_action == "text-only":
        if not has_text:
            raise DocumentError(403, reason="face_text_unavailable")
        return ImageOutput(
            False, "neurwerk: face policy requires text-only processing; images not forwarded."
        )
    if face_action == "reroute":
        return _face_reroute_output(policy, model, has_text, reply)
    if forwarding == "none":
        if not has_text:
            raise DocumentError(403, reason="image_text_unavailable")
        return ImageOutput(False, "neurwerk: text-extraction-only mode; images not forwarded.")
    if forwarding == "pii-unchecked":
        if not policy.image_models.get(model, False) or (
            reply is not None and reply.decision not in {"pass", "apply_actions"}
        ):
            raise DocumentError(403)
        return ImageOutput(True)
    return _conditional_output(forwarding, has_text, reply)


def _face_reroute_output(
    policy: ModelDestinationPolicy, model: str, has_text: bool, reply: EngineReply | None
) -> ImageOutput:
    """Require the exact local vision binding, preserving any text transformations."""
    forwarding = policy.image_forwarding.get(model, "none")
    if (
        forwarding == "none"
        or reply is None
        or reply.decision != "reroute"
        or not policy.image_reroutes.get(model, {}).get(reply.route_class or "")
    ):
        raise DocumentError(403, reason="image_reroute_unavailable")
    text_transformed = any(
        row.entity_type != "FACE" and row.transformed_count for row in reply.report.rows
    )
    if forwarding == "if-policy-allows" and text_transformed:
        return _text_fallback(has_text)
    # Approved local vision routes can receive images even when OCR found no text.
    return ImageOutput(True)


def _conditional_output(forwarding: str, has_text: bool, reply: EngineReply | None) -> ImageOutput:
    """Require fresh text analysis before applying strict or policy-aware forwarding."""
    if not has_text:
        raise DocumentError(403, reason="image_analysis_text_unavailable")
    if (
        reply is None
        or not reply.analysis.scan_performed
        or reply.analysis.text_leaf_count < 1
        or reply.analysis.source != "current_request"
        or reply.analysis.cached_decision_applied
    ):
        raise DocumentError(reason="image_analysis_failed")
    if reply.safety_rule is not None:
        raise DocumentError(403)
    if forwarding == "if-policy-allows":
        if reply.decision == "pass":
            return ImageOutput(True)
        if reply.decision in {"apply_actions", "reroute"}:
            # Text-triggered reroutes retain the engine route but never infer image
            # capability from its name. Only the FACE branch has a pixel binding.
            return _text_fallback(has_text)
        raise DocumentError(403)
    if reply.entities or reply.report.rows:
        raise DocumentError(403, reason="image_pii_detected")
    if reply.decision != "pass":
        raise DocumentError(403)
    return ImageOutput(True)


def _text_fallback(has_text: bool) -> ImageOutput:
    if not has_text:
        raise DocumentError(403, reason="image_analysis_text_unavailable")
    return ImageOutput(
        False, "neurwerk: PII policy applied; extracted text forwarded without images."
    )
