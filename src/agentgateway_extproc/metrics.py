"""Bounded Prometheus metrics for adapter transport and reversal."""

from prometheus_client import Counter, Gauge, Histogram

engine_requests_total = Counter(
    "extproc_engine_requests_total", "Engine request outcomes", ["outcome"]
)
engine_request_latency_seconds = Histogram(
    "extproc_engine_request_latency_seconds",
    "Engine request latency",
    ["outcome"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)
reversal_count_total = Counter(
    "extproc_reversal_count_total", "Reversed stream placeholders", ["entity_type"]
)
invalid_placeholder_total = Counter(
    "extproc_invalid_placeholder_total",
    "Unauthorized model placeholders replaced in human-readable output",
)
errors_total = Counter("extproc_errors_total", "Adapter errors", ["type"])
active_streams = Gauge("extproc_active_streams", "Active ext_proc streams")
response_transport_total = Counter(
    "extproc_response_transport_total",
    "Response streams handled by format and encoding",
    ["format", "encoding"],
)
response_failures_total = Counter(
    "extproc_response_failures_total",
    "Response processing failures by bounded phase and reason",
    ["phase", "reason"],
)
dispatcher_total = Counter(
    "extproc_dispatcher_total",
    "Destination policy dispatcher outcomes",
    ["outcome"],
)
