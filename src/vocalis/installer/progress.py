"""Reads the progress lines printed by the Vocalis installer script."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_STEP_RE = re.compile(r"^STEP (\d+)/(\d+) (.+)$")
_NOTE_PREFIX = "NOTE "


@dataclass(frozen=True)
class ProgressEvent:
    """One progress message: a numbered step, or a free-form note."""

    kind: Literal["step", "note"]
    index: int
    total: int
    text: str


def parse_line(line: str) -> ProgressEvent | None:
    """Turn one installer output line into an event, or None if it isn't one."""
    stripped = line.strip()
    match = _STEP_RE.match(stripped)
    if match:
        index, total, text = int(match[1]), int(match[2]), match[3].strip()
        if index < 1 or index > total or not text:
            return None
        return ProgressEvent("step", index, total, text)
    if stripped.startswith(_NOTE_PREFIX):
        text = stripped[len(_NOTE_PREFIX) :].strip()
        if text:
            return ProgressEvent("note", 0, 0, text)
    return None
