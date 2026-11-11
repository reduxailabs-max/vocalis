"""Reads the progress lines printed by the Vocalis installer script."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_STEP_RE = re.compile(r"^STEP (\d{1,6})/(\d{1,6}) (.+)$")
_PROGRESS_RE = re.compile(r"^PROGRESS (\d{1,15}) (\d{1,15}) (\d{1,15}) (-1|\d{1,9})$")
_NOTE_PREFIX = "NOTE "


@dataclass(frozen=True)
class ProgressEvent:
    """One progress message: a numbered step, a free-form note, or download progress.

    For "progress" events, `done` and `total` are bytes, `speed` is bytes per
    second and `eta` is seconds left (-1 when unknown).
    """

    kind: Literal["step", "note", "progress"]
    index: int
    total: int
    text: str
    done: int = 0
    speed: int = 0
    eta: int = -1


def parse_line(line: str) -> ProgressEvent | None:
    """Turn one installer output line into an event, or None if it isn't one."""
    stripped = line.strip()
    match = _STEP_RE.match(stripped)
    if match:
        index, total, text = int(match[1]), int(match[2]), match[3].strip()
        if index < 1 or index > total or not text:
            return None
        return ProgressEvent("step", index, total, text)
    match = _PROGRESS_RE.match(stripped)
    if match:
        done, total, speed, eta = (int(match[i]) for i in range(1, 5))
        if total < 1 or done > total:
            return None
        return ProgressEvent("progress", 0, total, "", done=done, speed=speed, eta=eta)
    if stripped.startswith(_NOTE_PREFIX):
        text = stripped[len(_NOTE_PREFIX) :].strip()
        if text:
            return ProgressEvent("note", 0, 0, text)
    return None
