"""Resumable, parallel file downloader for the installer (standard library only).

Each file is saved to ``<target>.part`` in fixed-size chunks fetched with HTTP
Range requests by a small pool of threads. A ``<target>.part.json`` record lists
the chunks already finished, so an interrupted run (or "Try again") continues
where it stopped. A finished file is checked (size, and sha256 when known)
before it is moved into place.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import queue
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vocalis.installer.download_progress import Reporter, SpeedMeter

CHUNK_BYTES = 16 * 1024 * 1024
WORKERS = 4
MAX_ATTEMPTS = 8
SOCKET_TIMEOUT_SECONDS = 30
BACKOFF_CAP_SECONDS = 30
_READ_BLOCK = 256 * 1024
_USER_AGENT = "vocalis-installer"
_NO_RETRY_CODES = frozenset({400, 401, 403, 404, 410})

Opener = Callable[[str, dict[str, str]], Any]


class DownloadError(RuntimeError):
    """A download failed; the message is plain English for the user."""


class DownloadStopped(DownloadError):
    """The download was stopped on purpose; partial files are kept for resume."""


@dataclass(frozen=True)
class DownloadItem:
    """One file to download."""

    url: str
    target: Path
    size: int
    sha256: str | None = None


def open_url(url: str, headers: dict[str, str]) -> Any:
    """Open a URL with a timeout so a stalled connection is retried, not hung."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **headers})
    return urllib.request.urlopen(request, timeout=SOCKET_TIMEOUT_SECONDS)


def _part(item: DownloadItem) -> Path:
    return item.target.with_name(item.target.name + ".part")


def _record(item: DownloadItem) -> Path:
    return item.target.with_name(item.target.name + ".part.json")


class _NoRange(Exception):
    """The server ignored the Range request and sent the whole file."""


class _ItemState:
    """Progress of one file inside a running download."""

    def __init__(self, item: DownloadItem, chunk_bytes: int, done: set[int]) -> None:
        self.item = item
        self.chunk_bytes = chunk_bytes
        self.count = max(1, -(-item.size // chunk_bytes))
        self.done = done
        self.lock = threading.Lock()
        self.whole_lock = threading.Lock()
        self.fd = -1

    def length(self, index: int) -> int:
        """Bytes in chunk number index."""
        return min(self.chunk_bytes, self.item.size - index * self.chunk_bytes)

    @property
    def complete(self) -> bool:
        """Whether every chunk is on disk."""
        return len(self.done) >= self.count


class Downloader:
    """Downloads a list of files with resume, retries and live progress."""

    def __init__(
        self,
        items: list[DownloadItem],
        *,
        opener: Opener = open_url,
        sleep: Callable[[float], object] | None = None,
        stop: threading.Event | None = None,
        workers: int = WORKERS,
        chunk_bytes: int = CHUNK_BYTES,
        max_attempts: int = MAX_ATTEMPTS,
        reporter: Reporter | None = None,
        meter: SpeedMeter | None = None,
    ) -> None:
        self._items = items
        self._opener = opener
        self._stop = stop if stop is not None else threading.Event()
        self._sleep = sleep if sleep is not None else self._stop.wait
        self._workers = workers
        self._chunk_bytes = chunk_bytes
        self._max_attempts = max_attempts
        self._reporter = reporter
        self._meter = meter
        self._failed = threading.Event()
        self._error: DownloadError | None = None
        self._error_lock = threading.Lock()

    def resumed_bytes(self) -> int:
        """Bytes already on disk from an earlier run (valid chunks only)."""
        total = 0
        for item in self._items:
            state = self._load_state(item, create=False)
            total += sum(state.length(i) for i in state.done) if state else 0
        return total

    def run(self) -> None:
        """Download everything; raise DownloadError (or DownloadStopped) on failure."""
        states = [self._prepare(item) for item in self._items]
        tasks: queue.Queue[tuple[_ItemState, int]] = queue.Queue()
        for state in states:
            for index in range(state.count):
                if index not in state.done:
                    tasks.put((state, index))
        try:
            threads = [threading.Thread(target=self._worker, args=(tasks,), daemon=True) for _ in range(self._workers)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            for state in states:
                if state.fd >= 0:
                    os.close(state.fd)
                    state.fd = -1
        if self._error is not None:
            raise self._error
        if self._stop.is_set():
            raise DownloadStopped("The download was stopped. It will continue where it left off next time.")
        for state in states:
            self._finish(state)
        if self._reporter is not None:
            self._reporter.finish()

    # -- preparation ---------------------------------------------------

    def _identity(self, item: DownloadItem) -> dict[str, Any]:
        return {"url": item.url, "size": item.size, "sha256": item.sha256, "chunk_bytes": self._chunk_bytes}

    def _load_state(self, item: DownloadItem, create: bool) -> _ItemState | None:
        """Read the resume record; start fresh if it does not match this download."""
        done: set[int] = set()
        part, record = _part(item), _record(item)
        try:
            data = json.loads(record.read_text())
            matches = (
                part.exists()
                and part.stat().st_size == item.size
                and all(data.get(k) == v for k, v in self._identity(item).items())
            )
            if matches:
                done = {int(i) for i in data.get("done", [])}
        except (OSError, ValueError, TypeError, AttributeError):
            done = set()
        state = _ItemState(item, self._chunk_bytes, done)
        state.done = {i for i in state.done if 0 <= i < state.count}
        if not state.done and not create:
            return None
        return state

    def _prepare(self, item: DownloadItem) -> _ItemState:
        """Open (or create) the part file and get the list of finished chunks."""
        item.target.parent.mkdir(parents=True, exist_ok=True)
        state = self._load_state(item, create=True)
        assert state is not None
        part = _part(item)
        fresh = not state.done
        if fresh:
            self._save_record(state)
        try:
            state.fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o644)
            if fresh or os.fstat(state.fd).st_size != item.size:
                self._reserve(state.fd, item)
        except OSError as error:
            if state.fd >= 0:
                os.close(state.fd)
                state.fd = -1
            raise DownloadError(f"Could not prepare {item.target.name}: {error.strerror or error}") from error
        return state

    @staticmethod
    def _reserve(fd: int, item: DownloadItem) -> None:
        """Reserve the whole file's disk space now, so a full disk fails early."""
        try:
            os.posix_fallocate(fd, 0, item.size)
        except OSError as error:
            if error.errno in (errno.ENOSPC, errno.EDQUOT):
                raise DownloadError(
                    f"There is not enough free disk space for {item.target.name}. Free some space and try again."
                ) from error
            os.ftruncate(fd, item.size)  # file system without fallocate support

    def _save_record(self, state: _ItemState) -> None:
        """Write the resume record safely (temp file, then rename)."""
        record = _record(state.item)
        data = {**self._identity(state.item), "done": sorted(state.done)}
        temp = record.with_name(record.name + ".tmp")
        temp.write_text(json.dumps(data))
        os.replace(temp, record)

    # -- workers -------------------------------------------------------

    def _worker(self, tasks: queue.Queue[tuple[_ItemState, int]]) -> None:
        while not self._stop.is_set() and not self._failed.is_set():
            try:
                state, index = tasks.get_nowait()
            except queue.Empty:
                return
            try:
                if not state.complete and index not in state.done:
                    self._do_chunk(state, index)
            except DownloadStopped:
                return
            except DownloadError as error:
                self._fail(error)
                return
            except Exception as error:  # noqa: BLE001 - never let a thread die silently
                self._fail(DownloadError(f"Unexpected download problem: {error}"))
                return

    def _fail(self, error: DownloadError) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error
        self._failed.set()

    def _do_chunk(self, state: _ItemState, index: int) -> None:
        """Fetch one chunk, retrying with exponential backoff."""
        name = state.item.target.name
        last: Exception | None = None
        for attempt in range(self._max_attempts):
            if self._stop.is_set():
                raise DownloadStopped("stopped")
            added = 0
            try:
                try:
                    added = self._fetch_chunk(state, index)
                except _NoRange:
                    self._fetch_whole(state)
                    return
                with state.lock:
                    state.done.add(index)
                    self._save_record(state)
                return
            except urllib.error.HTTPError as error:
                self._undo(added_bytes=getattr(error, "_vocalis_added", 0))
                if error.code in _NO_RETRY_CODES:
                    raise DownloadError(f"The server refused {name} (error {error.code}).") from error
                last = error
            except (OSError, TimeoutError, ValueError) as error:
                self._undo(added_bytes=getattr(error, "_vocalis_added", 0))
                last = error
            if attempt + 1 < self._max_attempts:
                self._sleep(min(2**attempt, BACKOFF_CAP_SECONDS))
        raise DownloadError(
            f"Could not download {name} after {self._max_attempts} tries ({last}). "
            "Check your internet connection and click Try again; it will continue where it stopped."
        )

    def _undo(self, added_bytes: int) -> None:
        if self._meter is not None and added_bytes:
            self._meter.add(-added_bytes)

    def _read_into(self, response: Any, state: _ItemState, offset: int, length: int) -> int:
        """Read exactly length bytes from response into the part file at offset."""
        got = 0
        try:
            while got < length:
                if self._stop.is_set():
                    raise DownloadStopped("stopped")
                block = response.read(min(_READ_BLOCK, length - got))
                if not block:
                    raise OSError(f"connection closed early ({got} of {length} bytes)")
                os.pwrite(state.fd, block, offset + got)
                got += len(block)
                if self._meter is not None:
                    self._meter.add(len(block))
                if self._reporter is not None:
                    self._reporter.update()
        except Exception as error:
            error._vocalis_added = got  # type: ignore[attr-defined]
            raise
        return got

    def _fetch_chunk(self, state: _ItemState, index: int) -> int:
        """One Range request for one chunk. Returns the bytes written."""
        offset = index * state.chunk_bytes
        length = state.length(index)
        headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
        with self._opener(state.item.url, headers) as response:
            status = getattr(response, "status", 206)
            if status != 206:
                if status == 200 and state.count == 1:
                    return self._read_into(response, state, 0, length)
                raise _NoRange
            return self._read_into(response, state, offset, length)

    def _fetch_whole(self, state: _ItemState) -> None:
        """The server cannot do ranges: stream the whole file once, from the start."""
        with state.whole_lock:
            if state.complete:
                return
            with self._opener(state.item.url, {}) as response:
                self._read_into(response, state, 0, state.item.size)
            with state.lock:
                state.done = set(range(state.count))
                self._save_record(state)

    # -- finishing -----------------------------------------------------

    def _finish(self, state: _ItemState) -> None:
        """Check the finished part file and move it into place."""
        item, part = state.item, _part(state.item)
        problem = None
        if part.stat().st_size != item.size:
            problem = "has the wrong size"
        elif item.sha256 is not None and _sha256(part) != item.sha256.lower():
            problem = "failed its checksum"
        if problem is not None:
            part.unlink(missing_ok=True)
            _record(item).unlink(missing_ok=True)
            raise DownloadError(
                f"{item.target.name} was damaged while downloading ({problem}). Click Try again to fetch it afresh."
            )
        os.replace(part, item.target)
        _record(item).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
