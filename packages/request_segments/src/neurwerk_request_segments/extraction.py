"""Extract opaque text segments and rebuild only their original text leaves."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel, ConfigDict, ValidationError

from .compatibility import (
    CompatibilitySettings,
    ExtractionLimitError,
    RequestKind,
    UnsupportedFeatureError,
)
from .models import EngineChatRequest, EngineMcpRequest, EngineResponsesRequest, SupportedRequest

type Path = tuple[str | int, ...]


class TextSegment(BaseModel):
    """An opaque identity and one independently analyzed text leaf."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    id: str
    text: str


@dataclass(frozen=True)
class _Leaf:
    path: Path
    text: str
    json_root: Path | None = None


def parse_request(
    payload: object,
    *,
    kind: RequestKind | None = None,
    controls: CompatibilitySettings | None = None,
) -> SupportedRequest:
    """Validate one endpoint shape without rewriting omitted or explicit controls."""
    if not isinstance(payload, dict):
        raise TypeError("request must be an object")  # noqa: TRY003
    if kind is None:
        candidates = [
            name
            for name, key in (("chat", "messages"), ("responses", "input"), ("mcp", "jsonrpc"))
            if key in payload
        ]
        if len(candidates) != 1:
            raise ValueError("request endpoint is ambiguous")  # noqa: TRY003
        kind = cast(RequestKind, candidates[0])
    model = {
        "chat": EngineChatRequest,
        "responses": EngineResponsesRequest,
        "mcp": EngineMcpRequest,
    }[kind]
    try:
        request = model.model_validate(payload, strict=True, context={"controls": controls})
    except ValidationError as exc:
        for error in exc.errors(include_url=False, include_input=False):
            cause = error.get("ctx", {}).get("error")
            if isinstance(cause, UnsupportedFeatureError):
                raise cause from exc
        raise
    request._controls = controls
    return request


@dataclass(frozen=True)
class ExtractedRequest:
    """Request-local extraction; protocol data never crosses the segment wire."""

    segments: list[TextSegment]
    request_kind: RequestKind
    attachments_present: bool
    _request: SupportedRequest
    _leaves: tuple[_Leaf, ...]
    _max_depth: int = 32

    def diagnostic_path(self, id: str) -> Path:
        """Resolve an opaque ID locally, never in the analysis service."""
        for segment, leaf in zip(self.segments, self._leaves, strict=True):
            if segment.id == id:
                return leaf.path
        raise ValueError("unknown segment identity")  # noqa: TRY003

    def rebuild(self, segments: Sequence[TextSegment]) -> SupportedRequest:
        """Require the exact IDs/order and replace only their authorized text."""
        if [segment.id for segment in segments] != [segment.id for segment in self.segments]:
            raise ValueError("segment identities or order changed")  # noqa: TRY003
        data = self._request.model_dump(mode="python", by_alias=True, exclude_unset=True)
        encoded: dict[Path, object] = {}
        for old, new, leaf in zip(self.segments, segments, self._leaves, strict=True):
            if not isinstance(new.text, str):
                raise TypeError("segment replacement must be text")  # noqa: TRY003
            if old.text == new.text:
                continue
            if leaf.json_root is None:
                _set(data, leaf.path, new.text)
            else:
                root = leaf.json_root
                if root not in encoded:
                    encoded[root] = _loads(cast(str, _get(data, root)), self._max_depth)
                nested = leaf.path[len(root) :]
                if nested:
                    _set(encoded[root], nested, new.text)
                else:
                    encoded[root] = new.text
        for root, value in encoded.items():
            _set(
                data,
                root,
                json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            )
        return parse_request(data, kind=self.request_kind, controls=self._request._controls)

    def control_shape(self) -> dict[str, object]:
        """Normalize text, including encoded argument leaves, for mutation checks."""
        data = self._request.model_dump(mode="python", by_alias=True, exclude_unset=True)
        encoded: dict[Path, object] = {}
        for leaf in self._leaves:
            if leaf.json_root is None:
                _set(data, leaf.path, None)
            else:
                root = leaf.json_root
                if root not in encoded:
                    encoded[root] = _loads(cast(str, _get(data, root)), self._max_depth)
                nested = leaf.path[len(root) :]
                if nested:
                    _set(encoded[root], nested, None)
                else:
                    encoded[root] = None
        for root, value in encoded.items():
            # Retain the encoded-vs-object distinction too.
            _set(data, root, ("json-string", value))
        return cast(dict[str, object], data)


def extract_request(request: SupportedRequest, *, max_depth: int = 32) -> ExtractedRequest:  # noqa: C901
    """Traverse supported content, excluding IDs, schema enums and protocol controls."""
    data = request.model_dump(mode="python", by_alias=True, exclude_unset=True)
    leaves: list[_Leaf] = []
    attachments = False

    def text(value: object, path: Path) -> None:
        if isinstance(value, str):
            leaves.append(_Leaf(path, value))

    def content(value: object, path: Path) -> None:
        nonlocal attachments
        if isinstance(value, str):
            text(value, path)
        elif isinstance(value, list):
            for index, part in enumerate(value):
                if part.get("type") in {"text", "input_text", "output_text"}:
                    text(part.get("text"), (*path, index, "text"))
                else:
                    attachments = True

    if isinstance(request, EngineMcpRequest):
        kind: RequestKind = "mcp"
        _validate_mcp_metadata(request.params.meta, max_depth)
        leaves.extend(
            _json_leaves(request.params.arguments, ("params", "arguments"), max_depth=max_depth)
        )
    elif isinstance(request, EngineChatRequest):
        kind = "chat"
        for index, message in enumerate(data["messages"]):
            path: Path = ("messages", index)
            content(message.get("content"), (*path, "content"))
            text(message.get("refusal"), (*path, "refusal"))
            for call_index, call in enumerate(message.get("tool_calls") or []):
                root = (*path, "tool_calls", call_index, "function", "arguments")
                if "arguments" in call["function"]:
                    leaves.extend(_arguments(call["function"]["arguments"], root, max_depth))
            if (
                isinstance(message.get("function_call"), dict)
                and "arguments" in message["function_call"]
            ):
                leaves.extend(
                    _arguments(
                        message["function_call"]["arguments"],
                        (*path, "function_call", "arguments"),
                        max_depth,
                    )
                )
        for index, tool in enumerate(data.get("tools") or []):
            leaves.extend(_definition(tool["function"], ("tools", index, "function"), max_depth))
        for index, function in enumerate(data.get("functions") or []):
            leaves.extend(_definition(function, ("functions", index), max_depth))
        leaves.extend(
            _schema(data.get("response_format"), ("response_format",), max_depth=max_depth)
        )
    else:
        kind = "responses"
        text(data.get("instructions"), ("instructions",))
        value = data["input"]
        if isinstance(value, str):
            text(value, ("input",))
        else:
            for index, item in enumerate(value):
                path = ("input", index)
                if item.get("type", "message") == "message":
                    content(item["content"], (*path, "content"))
                elif item["type"] == "function_call":
                    leaves.extend(_arguments(item["arguments"], (*path, "arguments"), max_depth))
                elif item["type"] == "function_call_output":
                    leaves.extend(
                        _json_leaves(item["output"], (*path, "output"), max_depth=max_depth)
                    )
        for index, tool in enumerate(data.get("tools") or []):
            leaves.extend(_definition(tool, ("tools", index), max_depth))
        config = data.get("text")
        if isinstance(config, dict):
            leaves.extend(_schema(config.get("format"), ("text", "format"), max_depth=max_depth))
    return ExtractedRequest(
        [TextSegment(id=f"s{index}", text=leaf.text) for index, leaf in enumerate(leaves)],
        kind,
        attachments,
        deepcopy(request),
        tuple(leaves),
        max_depth,
    )


def _validate_mcp_metadata(value: object, max_depth: int) -> None:
    """Bound immutable metadata without treating its strings as inspection text."""
    pending = [(value, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if depth > max_depth:
            raise ExtractionLimitError(depth, max_depth)
        if nodes > 4_096:
            raise ExtractionLimitError(nodes, 4_096, reason="nodes")
        if isinstance(current, dict):
            pending.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)


def _json_leaves(
    value: object, path: Path, depth: int = 0, root: Path | None = None, max_depth: int = 32
) -> list[_Leaf]:
    if depth > max_depth:
        raise ExtractionLimitError(depth, max_depth)
    if isinstance(value, str):
        return [_Leaf(path, value, root)]
    if isinstance(value, list):
        return [
            leaf
            for index, item in enumerate(value)
            for leaf in _json_leaves(item, (*path, index), depth + 1, root, max_depth)
        ]
    if isinstance(value, dict):
        return [
            leaf
            for key, item in value.items()
            for leaf in _json_leaves(item, (*path, key), depth + 1, root, max_depth)
        ]
    return []


def _arguments(value: object, path: Path, max_depth: int) -> list[_Leaf]:
    if isinstance(value, str):
        return _json_leaves(_loads(value, max_depth), path, root=path, max_depth=max_depth)
    return _json_leaves(value, path, max_depth=max_depth)


def _definition(value: dict[str, object], path: Path, max_depth: int) -> list[_Leaf]:
    description = value.get("description")
    leaves: list[_Leaf] = (
        [_Leaf((*path, "description"), description)] if isinstance(description, str) else []
    )
    return leaves + _schema(value.get("parameters"), (*path, "parameters"), max_depth=max_depth)


def _schema(value: object, path: Path, depth: int = 0, max_depth: int = 32) -> list[_Leaf]:
    if depth > max_depth:
        raise ExtractionLimitError(depth, max_depth)
    if not isinstance(value, dict):
        return []
    leaves: list[_Leaf] = []
    for key, item in value.items():
        child = (*path, key)
        if key in {"description", "title", "default", "examples"}:
            leaves.extend(_json_leaves(item, child, depth + 1, max_depth=max_depth))
        elif key in {
            "schema",
            "json_schema",
            "items",
            "additionalProperties",
            "contains",
            "not",
            "if",
            "then",
            "else",
            "propertyNames",
            "unevaluatedItems",
            "unevaluatedProperties",
            "contentSchema",
        }:
            leaves.extend(_schema(item, child, depth + 1, max_depth))
        elif key in {
            "properties",
            "$defs",
            "definitions",
            "patternProperties",
            "dependentSchemas",
        } and isinstance(item, dict):
            for name, schema in item.items():
                leaves.extend(_schema(schema, (*child, name), depth + 1, max_depth))
        elif key in {"allOf", "anyOf", "oneOf", "prefixItems"} and isinstance(item, list):
            for index, schema in enumerate(item):
                leaves.extend(_schema(schema, (*child, index), depth + 1, max_depth))
    return leaves


def _get(root: object, path: Path) -> object:
    current = root
    for part in path:
        if isinstance(current, dict) and isinstance(part, str):  # noqa: SIM114
            current = current[part]
        elif isinstance(current, list) and isinstance(part, int):
            current = current[part]
        else:
            raise TypeError("invalid text path")  # noqa: TRY003
    return current


def _set(root: object, path: Path, value: object) -> None:
    parent = _get(root, path[:-1])
    key = path[-1]
    if isinstance(parent, dict) and isinstance(key, str):  # noqa: SIM114
        parent[key] = value
    elif isinstance(parent, list) and isinstance(key, int):
        parent[key] = value
    else:
        raise TypeError("invalid text path")  # noqa: TRY003


def _loads(value: str, max_depth: int) -> object:
    _check_encoded_depth(value, max_depth)

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate argument key")  # noqa: TRY003
            result[key] = item
        return result

    def number(raw: str) -> float:
        result = float(raw)
        if not math.isfinite(result):
            raise ValueError("non-finite argument value")  # noqa: TRY003
        return result

    return json.loads(value, object_pairs_hook=pairs, parse_float=number, parse_constant=number)


def _check_encoded_depth(value: str, max_depth: int) -> None:
    """Bound nesting before the JSON decoder, ignoring escaped string contents."""
    depth = 0
    quoted = False
    escaped = False
    for char in value:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > max_depth:
                raise ExtractionLimitError(depth, max_depth)
        elif char in "]}":
            depth = max(0, depth - 1)
