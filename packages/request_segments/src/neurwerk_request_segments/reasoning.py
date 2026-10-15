"""Shared reasoning carriers; readable history and immutable continuation are distinct."""

from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

type ReasoningIndex = Annotated[int, Field(ge=0, le=255)]
type ReasoningId = Annotated[
    str, Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
]

CHAT_REASONING_FIELDS = frozenset(
    {
        "reasoning",
        "reasoning_content",
        "reasoning_signature",
        "reasoning_details",
        "thinking_blocks",
    }
)


class ReasoningModel(BaseModel):
    """Close content-bearing reasoning shapes, including nested extension fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ReasoningDetail(ReasoningModel):
    """Identity shared by complete reasoning details and their indexed deltas."""

    id: ReasoningId | None = None
    format: Annotated[str, Field(pattern=r"^[a-z0-9-]+$", max_length=64)] | None = None
    index: ReasoningIndex | None = None


class ReasoningText(ReasoningDetail):
    """Text is inspectable unless paired with a provider signature."""

    type: Literal["reasoning.text"]
    text: str | None = None
    signature: str | None = None


class ReasoningSummary(ReasoningDetail):
    """An ordinary readable summary."""

    type: Literal["reasoning.summary"]
    summary: str


class ReasoningEncrypted(ReasoningDetail):
    """Encrypted or redacted provider continuation."""

    type: Literal["reasoning.encrypted"]
    data: str


class ThinkingBlock(ReasoningModel):
    """Chat's signed thinking block representation."""

    type: Literal["thinking"]
    thinking: str | None = None
    signature: str | None = None
    index: ReasoningIndex | None = None


class RedactedThinking(ReasoningModel):
    """Chat's opaque redacted thinking representation."""

    type: Literal["redacted_thinking"]
    data: str
    index: ReasoningIndex | None = None


class ChatReasoning(BaseModel):
    """Fields shared by request messages, response messages and response deltas."""

    reasoning: str | None = None
    reasoning_content: str | None = None
    reasoning_signature: str | None = None
    reasoning_details: (
        Annotated[
            list[
                Annotated[
                    ReasoningText | ReasoningSummary | ReasoningEncrypted,
                    Field(discriminator="type"),
                ]
            ],
            Field(max_length=256),
        ]
        | None
    ) = None
    thinking_blocks: (
        Annotated[
            list[Annotated[ThinkingBlock | RedactedThinking, Field(discriminator="type")]],
            Field(max_length=256),
        ]
        | None
    ) = None


class SummaryText(ReasoningModel):
    """Responses reasoning summary text."""

    type: Literal["summary_text"]
    text: str


class ReasoningTextPart(ReasoningModel):
    """Responses readable reasoning content."""

    type: Literal["reasoning_text"]
    text: str


class ResponseReasoning(ReasoningModel):
    """A replayable Responses reasoning item; only encrypted_content is opaque."""

    type: Literal["reasoning"]
    id: ReasoningId | None = None
    summary: list[SummaryText] = Field(default_factory=list, max_length=64)
    content: list[ReasoningTextPart] | None = Field(default=None, max_length=64)
    encrypted_content: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


def chat_reasoning_text(message: dict[str, Any]) -> Iterator[tuple[tuple[str | int, ...], str]]:
    """Enumerate editable history text; a signature protects its accompanying text too."""
    if not message.get("reasoning_signature"):
        for key in ("reasoning", "reasoning_content"):
            if isinstance(value := message.get(key), str):
                yield (key,), value
    for field, text_fields in (
        ("reasoning_details", ("text", "summary")),
        ("thinking_blocks", ("thinking",)),
    ):
        for index, block in enumerate(message.get(field) or []):
            if not block.get("signature"):
                for key in text_fields:
                    if isinstance(value := block.get(key), str):
                        yield (field, index, key), value
