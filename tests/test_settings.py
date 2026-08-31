"""Test hard transport and engine configuration boundaries."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentgateway_extproc.config.settings import (
    DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    EngineSettings,
    Settings,
)
from agentgateway_extproc.metrics import engine_request_latency_seconds


def test_transport_limit_defaults_are_exact() -> None:
    settings = Settings()
    assert settings.max_request_bytes == 5_242_880
    assert settings.max_response_bytes == 10_485_760
    assert settings.max_transformed_request_bytes == 10_485_760
    assert settings.grpc_max_receive_message_bytes == 6_356_992
    assert settings.engine.max_response_bytes == 10_485_760


@pytest.mark.parametrize(
    ("field", "maximum"),
    [
        ("max_request_bytes", MAX_REQUEST_BYTES),
        ("max_response_bytes", MAX_RESPONSE_BYTES),
        ("max_transformed_request_bytes", MAX_RESPONSE_BYTES),
    ],
)
def test_top_level_limits_accept_the_boundary_and_reject_larger(field: str, maximum: int) -> None:
    assert getattr(Settings.model_validate({field: maximum}), field) == maximum
    with pytest.raises(ValidationError):
        Settings.model_validate({field: maximum + 1})


def test_lower_request_limit_is_allowed_but_grpc_limit_must_exceed_it() -> None:
    assert Settings(max_request_bytes=1_024).max_request_bytes == 1_024
    with pytest.raises(ValidationError, match="must exceed max_request_bytes"):
        Settings(
            max_request_bytes=MAX_REQUEST_BYTES,
            grpc_max_receive_message_bytes=MAX_REQUEST_BYTES,
        )
    with pytest.raises(ValidationError):
        Settings(grpc_max_receive_message_bytes=DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_BYTES + 1)


def test_engine_limits_accept_exact_maxima_and_reject_larger() -> None:
    settings = EngineSettings(timeout=615, max_response_bytes=MAX_RESPONSE_BYTES)
    assert settings.timeout == 615
    assert settings.max_response_bytes == MAX_RESPONSE_BYTES
    with pytest.raises(ValidationError):
        EngineSettings(timeout=615.01)
    with pytest.raises(ValidationError):
        EngineSettings(max_response_bytes=MAX_RESPONSE_BYTES + 1)


def test_engine_latency_histogram_has_buckets_through_six_hundred_seconds() -> None:
    engine_request_latency_seconds.labels(outcome="bucket-test").observe(0)
    buckets = {
        sample.labels["le"]
        for metric in engine_request_latency_seconds.collect()
        for sample in metric.samples
        if sample.name.endswith("_bucket") and sample.labels["outcome"] == "bucket-test"
    }
    assert "600.0" in buckets
    assert "+Inf" in buckets
