"""Incremental, source-independent Server-Sent Event framing."""

# ruff: noqa: TRY003

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SseEvent:
    """One complete SSE event with its semantic fields and original field order."""

    lines: list[tuple[str, str]] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)

    @property
    def data(self) -> str:
        """Return SSE data lines joined according to the EventSource algorithm."""
        return "\n".join(value for name, value in self.lines if name == "data")

    @property
    def event(self) -> str:
        """Return the explicitly supplied event type or the EventSource default."""
        values = [value for name, value in self.lines if name == "event"]
        return values[-1] if values else "message"

    def replace_data(self, data: str) -> None:
        """Replace all data fields while retaining event/id/retry fields."""
        data_lines = [index for index, (name, _value) in enumerate(self.lines) if name == "data"]
        raw_data_lines = [
            index for index, line in enumerate(self.raw_lines) if _sse_field_name(line) == "data"
        ]
        if (
            len(data_lines) == 1
            and len(raw_data_lines) == 1
            and "\n" not in data
            and "\r" not in data
        ):
            index = raw_data_lines[0]
            self.raw_lines[index] = _replace_raw_data(self.raw_lines[index], data)
        else:
            self.raw_lines.clear()
        self.lines = [(name, value) for name, value in self.lines if name != "data"]
        self.lines.append(("data", data))

    def render(self) -> str:
        """Render a valid normalized SSE frame."""
        if self.raw_lines:
            return "".join(self.raw_lines)
        output: list[str] = []
        for name, value in self.lines:
            if name == "":
                output.append(f":{value}\n")
            elif name == "data" and "\n" in value:
                output.extend(f"data: {part}\n" for part in value.split("\n"))
            elif value:
                output.append(f"{name}: {value}\n")
            else:
                output.append(f"{name}\n")
        return "".join(output) + "\n"


class SseDecoder:
    """Frame SSE text across arbitrary decoded body chunks."""

    def __init__(self, *, emit_empty_frames: bool = False) -> None:
        """Initialize retained line and event state."""
        self._buffer = ""
        self._event = SseEvent()
        self._emit_empty_frames = emit_empty_frames

    def feed(self, text: str, *, final: bool = False) -> list[SseEvent]:
        """Consume text and return fully framed events only."""
        self._buffer += text
        events: list[SseEvent] = []
        position = 0
        while position < len(self._buffer):
            ending = self._line_ending(position, final)
            if ending is None:
                break
            end, next_position = ending
            event = self._feed_line(
                self._buffer[position:end],
                self._buffer[position:next_position],
            )
            if event is not None:
                events.append(event)
            position = next_position
        self._buffer = self._buffer[position:]
        if final:
            if self._buffer:
                raise ValueError("SSE stream ended with an unterminated line")
            if self._event.lines:
                raise ValueError("SSE stream ended with an unterminated event")
        return events

    def _line_ending(self, position: int, final: bool) -> tuple[int, int] | None:
        for index in range(position, len(self._buffer)):
            character = self._buffer[index]
            if character == "\n":
                return index, index + 1
            if character == "\r":
                if index + 1 == len(self._buffer) and not final:
                    return None
                is_crlf = self._buffer[index + 1 : index + 2] == "\n"
                next_position = index + 2 if is_crlf else index + 1
                return index, next_position
        return None

    def _feed_line(self, line: str, raw_line: str) -> SseEvent | None:
        if not line:
            if not self._event.lines:
                return SseEvent(raw_lines=[raw_line]) if self._emit_empty_frames else None
            self._event.raw_lines.append(raw_line)
            event = self._event
            self._event = SseEvent()
            return event
        self._event.raw_lines.append(raw_line)
        if line.startswith(":"):
            self._event.lines.append(("", line[1:]))
            return None
        name, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        self._event.lines.append((name, value if separator else ""))
        return None


def _sse_field_name(raw_line: str) -> str:
    line = raw_line.rstrip("\r\n")
    if line.startswith(":"):
        return ""
    return line.partition(":")[0]


def _replace_raw_data(raw_line: str, data: str) -> str:
    content = raw_line.rstrip("\r\n")
    ending = raw_line[len(content) :]
    _name, separator, value = content.partition(":")
    if not separator:
        return f"data:{data}{ending}"
    spacing = " " if value.startswith(" ") else ""
    return f"data:{spacing}{data}{ending}"
