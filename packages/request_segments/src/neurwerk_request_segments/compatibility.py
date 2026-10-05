"""Explicit, endpoint-scoped admission of reviewed non-content controls."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

type RequestKind = Literal["chat", "responses", "mcp"]
type ControlLocation = Literal[
    "request", "message", "tool", "function", "stream_options", "text", "text_format", "input_item"
]
type ValueKind = Literal["string", "boolean", "integer", "number", "null"]


class UnsupportedFeatureError(ValueError):
    """An unreviewed field is not safe to pass through."""

    def __init__(self, location: str, field: str) -> None:
        """Retain a diagnostic location without including the rejected value."""
        super().__init__("unsupported request feature")
        self.location = location
        self.field = field


class ExtractionLimitError(ValueError):
    """A measured shared-extraction limit, without rejected content."""

    def __init__(
        self,
        measured: int,
        maximum: int,
        *,
        reason: Literal["depth", "nodes"] = "depth",
        exact: bool = False,
    ) -> None:
        """Depth traversal stops at the first excess, so the measurement is a lower bound."""
        super().__init__("request extraction depth exceeds the supported limit")
        self.component = "request_segments"
        self.stage = "inspection"
        self.reason = reason
        self.measured = measured
        self.maximum = maximum
        self.unit = "levels" if reason == "depth" else "items"
        self.exact = exact


class ControlRule(BaseModel):
    """Admit one reviewed scalar control, never arbitrary content or paths."""

    model_config = ConfigDict(extra="forbid", strict=True)
    endpoint: Literal["chat", "responses"]
    location: ControlLocation = "request"
    field: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    kinds: list[ValueKind] = Field(min_length=1, max_length=5)
    enum: list[Annotated[str, Field(max_length=256)] | int | float | bool | None] | None = Field(
        default=None, min_length=1, max_length=64
    )

    @model_validator(mode="after")
    def validate_kinds(self) -> ControlRule:
        """Never create a free-form text channel through operator configuration."""
        if len(self.kinds) != len(set(self.kinds)) or (
            "string" in self.kinds and self.enum is None
        ):
            raise ValueError("string controls require an enum and kinds must be unique")  # noqa: TRY003
        if self.enum is not None and any(not self.accepts(value) for value in self.enum):
            raise ValueError("control enum must contain finite values matching its kinds")  # noqa: TRY003
        return self

    def accepts(self, value: object) -> bool:
        """Match exact JSON types; booleans are not numbers."""
        kind = {
            str: "string",
            bool: "boolean",
            int: "integer",
            float: "number",
            type(None): "null",
        }.get(type(value))
        if kind not in self.kinds and not (type(value) is int and "number" in self.kinds):
            return False
        if isinstance(value, float) and not math.isfinite(value):
            return False
        if isinstance(value, str) and len(value) > 256:
            return False
        return self.enum is None or any(
            type(value) is type(item) and value == item for item in self.enum
        )


class CompatibilitySettings(BaseModel):
    """Reviewed additional controls shared by all consumers of the package."""

    model_config = ConfigDict(extra="forbid", strict=True)
    controls: list[ControlRule] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def validate_rules(self) -> CompatibilitySettings:
        """Prevent duplicate rules and overrides of known protocol/content fields."""
        from .models import reserved_fields

        seen: set[tuple[str, str, str]] = set()
        for rule in self.controls:
            key = (rule.endpoint, rule.location, rule.field)
            if key in seen or rule.field in reserved_fields(rule.endpoint, rule.location):
                raise ValueError("compatibility rule overlaps an existing field")  # noqa: TRY003
            seen.add(key)
        return self
