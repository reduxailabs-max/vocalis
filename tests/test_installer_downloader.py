"""Tests for the resumable installer downloader (fake server, no network)."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from vocalis.installer import downloader as dl
from vocalis.installer.download_progress import Reporter, SpeedMeter

CHUNK = 100
DATA = bytes(range(256)) * 3  # 768 bytes -> 8 chunks of 100


class FakeResponse:
    def __init__(self, body: bytes, status: int) -> None:
        self._io = io.BytesIO(body)
        self.status = status

    def read(self, n: int = -1) -> bytes:
        return self._io.read(n)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeServer:
    def __init__(self, data: bytes = DATA, *, ranges: bool = True) -> None:
        self.data, self.ranges = data, ranges
        self.requests: list[str | None] = []
        self.fail_first = 0
        self.fail_with: Exception | None = None
        self.short = False
        self.lock = threading.Lock()

    def __call__(self, url: str, headers: dict[str, str]) -> FakeResponse:
        with self.lock:
            self.requests.append(headers.get("Range"))
            if self.fail_first > 0:
                self.fail_first -= 1
                raise self.fail_with or OSError("boom")
            if self.fail_with is not None and self.fail_first < 0:
                raise self.fail_with
        rng = headers.get("Range")
        if rng and self.ranges:
            start, end = (int(x) for x in rng.split("=")[1].split("-"))
            body = self.data[start : end + 1]
            if self.short:
                body = body[: len(body) // 2]
            return FakeResponse(body, 206)
        return FakeResponse(self.data, 200)


def make(tmp_path: Path, **kw: Any) -> tuple[dl.Downloader, dl.DownloadItem, FakeServer]:
    server = kw.pop("server", None) or FakeServer()
    item = kw.pop("item", None) or dl.DownloadItem("http://x/f", tmp_path / "f.bin", len(DATA), hashlib.sha256(DATA).hexdigest())
    sleeps: list[float] = []
    kw.setdefault("sleep", sleeps.append)
    d = dl.Downloader([item], opener=server, chunk_bytes=CHUNK, workers=3, **kw)
    d.sleeps = sleeps  # type: ignore[attr-defined]
    return d, item, server


def test_downloads_whole_file_in_chunks(tmp_path: Path) -> None:
    d, item, server = make(tmp_path)
    d.run()
    assert item.target.read_bytes() == DATA
    assert sorted(r for r in server.requests if r) == sorted(f"bytes={i}-{min(i + 99, 767)}" for i in range(0, 768, 100))
    assert not Path(str(item.target) + ".part").exists()
    assert not Path(str(item.target) + ".part.json").exists()


def test_resume_only_requests_missing_chunks(tmp_path: Path) -> None:
    d, item, server = make(tmp_path)
    part = Path(str(item.target) + ".part")
    part.write_bytes(DATA[:300] + b"\0" * (len(DATA) - 300))
    Path(str(item.target) + ".part.json").write_text(
        json.dumps({"url": item.url, "size": item.size, "sha256": item.sha256, "chunk_bytes": CHUNK, "done": [0, 1, 2]})
    )
    assert d.resumed_bytes() == 300
    d.run()
    assert item.target.read_bytes() == DATA
    assert "bytes=0-99" not in server.requests and "bytes=200-299" not in server.requests
    assert len(server.requests) == 5


def test_changed_identity_restarts(tmp_path: Path) -> None:
    d, item, server = make(tmp_path)
    Path(str(item.target) + ".part").write_bytes(b"\1" * len(DATA))
    Path(str(item.target) + ".part.json").write_text(
        json.dumps({"url": "http://other", "size": item.size, "sha256": item.sha256, "chunk_bytes": CHUNK, "done": [0, 1, 2, 3, 4, 5, 6, 7]})
    )
    assert d.resumed_bytes() == 0
    d.run()
    assert item.target.read_bytes() == DATA
    assert len(server.requests) == 8


def test_corrupt_record_is_ignored(tmp_path: Path) -> None:
    d, item, _ = make(tmp_path)
    Path(str(item.target) + ".part.json").write_text("not json")
    d.run()
    assert item.target.read_bytes() == DATA


def test_retries_with_exponential_backoff(tmp_path: Path) -> None:
    server = FakeServer()
    server.fail_first = 3
    d, item, _ = make(tmp_path, server=server)
    d.run()
    assert item.target.read_bytes() == DATA
    assert len(d.sleeps) == 3  # type: ignore[attr-defined]


def test_gives_up_after_max_attempts_and_keeps_partial(tmp_path: Path) -> None:
    server = FakeServer()
    server.fail_first = -1
    server.fail_with = OSError("down")
    d, item, _ = make(tmp_path, server=server, max_attempts=3)
    with pytest.raises(dl.DownloadError, match="Try again"):
        d.run()
    assert Path(str(item.target) + ".part").exists()
    assert Path(str(item.target) + ".part.json").exists()
    assert not item.target.exists()
    assert 1 in d.sleeps and 2 in d.sleeps  # type: ignore[attr-defined]


def test_backoff_is_capped(tmp_path: Path) -> None:
    server = FakeServer()
    server.fail_first = -1
    server.fail_with = OSError("down")
    d, _, _ = make(tmp_path, server=server, max_attempts=9)
    with pytest.raises(dl.DownloadError):
        d.run()
    assert max(d.sleeps) == dl.BACKOFF_CAP_SECONDS  # type: ignore[attr-defined]


def test_http_404_is_not_retried(tmp_path: Path) -> None:
    server = FakeServer()
    server.fail_first = -1
    server.fail_with = urllib.error.HTTPError("u", 404, "nf", {}, None)  # type: ignore[arg-type]
    d, _, _ = make(tmp_path, server=server)
    with pytest.raises(dl.DownloadError, match="refused"):
        d.run()
    assert d.sleeps == []  # type: ignore[attr-defined]


def test_http_503_is_retried(tmp_path: Path) -> None:
    server = FakeServer()
    server.fail_first = 2
    server.fail_with = urllib.error.HTTPError("u", 503, "busy", {}, None)  # type: ignore[arg-type]
    d, item, _ = make(tmp_path, server=server)
    d.run()
    assert item.target.read_bytes() == DATA


def test_short_read_is_retried_then_fails(tmp_path: Path) -> None:
    server = FakeServer()
    server.short = True
    d, _, _ = make(tmp_path, server=server, max_attempts=2)
    with pytest.raises(dl.DownloadError):
        d.run()


def test_checksum_mismatch_is_rejected_and_cleaned(tmp_path: Path) -> None:
    item = dl.DownloadItem("http://x/f", tmp_path / "f.bin", len(DATA), "0" * 64)
    d, _, _ = make(tmp_path, item=item)
    with pytest.raises(dl.DownloadError, match="damaged"):
        d.run()
    assert not item.target.exists()
    assert not Path(str(item.target) + ".part").exists()
    assert not Path(str(item.target) + ".part.json").exists()


def test_no_sha_skips_checksum(tmp_path: Path) -> None:
    item = dl.DownloadItem("http://x/f", tmp_path / "f.bin", len(DATA), None)
    d, _, _ = make(tmp_path, item=item)
    d.run()
    assert item.target.read_bytes() == DATA


def test_server_without_range_falls_back_to_whole_stream(tmp_path: Path) -> None:
    d, item, _ = make(tmp_path, server=FakeServer(ranges=False))
    d.run()
    assert item.target.read_bytes() == DATA


def test_small_file_single_chunk_with_200(tmp_path: Path) -> None:
    data = b"hello"
    item = dl.DownloadItem("http://x/s", tmp_path / "s.txt", len(data), None)
    d = dl.Downloader([item], opener=FakeServer(data, ranges=False), chunk_bytes=CHUNK, sleep=lambda s: None)
    d.run()
    assert item.target.read_bytes() == data


def test_stop_keeps_partial_files(tmp_path: Path) -> None:
    stop = threading.Event()
    stop.set()
    d, item, _ = make(tmp_path, stop=stop)
    with pytest.raises(dl.DownloadStopped):
        d.run()
    assert Path(str(item.target) + ".part").exists()
    assert not item.target.exists()


def test_stop_during_download_then_resume_finishes(tmp_path: Path) -> None:
    stop = threading.Event()
    server = FakeServer()
    original = server.__call__

    seen = {"n": 0}

    def opener(url: str, headers: dict[str, str]) -> FakeResponse:
        seen["n"] += 1
        if seen["n"] == 4:
            stop.set()
        return original(url, headers)

    item = dl.DownloadItem("http://x/f", tmp_path / "f.bin", len(DATA), hashlib.sha256(DATA).hexdigest())
    first = dl.Downloader([item], opener=opener, chunk_bytes=CHUNK, workers=1, stop=stop, sleep=lambda s: None)
    with pytest.raises(dl.DownloadStopped):
        first.run()
    second_server = FakeServer()
    second = dl.Downloader([item], opener=second_server, chunk_bytes=CHUNK, workers=1, sleep=lambda s: None)
    assert second.resumed_bytes() > 0
    second.run()
    assert item.target.read_bytes() == DATA
    assert len(second_server.requests) < 8


def test_disk_full_gives_plain_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import errno

    def full(fd: int, off: int, length: int) -> None:
        raise OSError(errno.ENOSPC, "No space")

    monkeypatch.setattr("os.posix_fallocate", full)
    d, _, _ = make(tmp_path)
    with pytest.raises(dl.DownloadError, match="free disk space"):
        d.run()


def test_filesystem_without_fallocate_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import errno

    def unsupported(fd: int, off: int, length: int) -> None:
        raise OSError(errno.EOPNOTSUPP, "nope")

    monkeypatch.setattr("os.posix_fallocate", unsupported)
    d, item, _ = make(tmp_path)
    d.run()
    assert item.target.read_bytes() == DATA


def test_progress_reaches_total_and_reports(tmp_path: Path) -> None:
    meter = SpeedMeter(len(DATA))
    out = io.StringIO()
    reporter = Reporter(meter, stream=out)
    d, _, _ = make(tmp_path, meter=meter, reporter=reporter)
    d.run()
    assert meter.done == len(DATA)
    last = [line for line in out.getvalue().splitlines() if line.startswith("PROGRESS")][-1]
    assert last.startswith(f"PROGRESS {len(DATA)} {len(DATA)} ")


def test_failed_attempt_bytes_are_taken_back_from_progress(tmp_path: Path) -> None:
    server = FakeServer()
    server.short = True
    meter = SpeedMeter(len(DATA))
    d, _, _ = make(tmp_path, server=server, meter=meter, max_attempts=2)
    with pytest.raises(dl.DownloadError):
        d.run()
    assert meter.done == 0


def test_open_url_sets_timeout_and_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(request: Any, timeout: float) -> str:
        seen["timeout"], seen["ua"], seen["range"] = timeout, request.get_header("User-agent"), request.get_header("Range")
        return "resp"

    monkeypatch.setattr("urllib.request.urlopen", fake)
    assert dl.open_url("http://x", {"Range": "bytes=0-1"}) == "resp"
    assert seen == {"timeout": dl.SOCKET_TIMEOUT_SECONDS, "ua": "vocalis-installer", "range": "bytes=0-1"}


def test_several_files_share_the_pool(tmp_path: Path) -> None:
    a = dl.DownloadItem("http://x/a", tmp_path / "a.bin", len(DATA), None)
    b = dl.DownloadItem("http://x/b", tmp_path / "sub" / "b.bin", len(DATA), None)
    d = dl.Downloader([a, b], opener=FakeServer(), chunk_bytes=CHUNK, workers=4, sleep=lambda s: None)
    d.run()
    assert a.target.read_bytes() == DATA and b.target.read_bytes() == DATA


def test_real_http_range_requests_over_loopback(tmp_path: Path) -> None:
    """The real urllib opener against a tiny local server that honours Range (and redirects)."""
    import http.server

    payload = bytes(range(256)) * 40  # 10240 bytes

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/start":
                self.send_response(302)
                self.send_header("Location", "/file")
                self.end_headers()
                return
            rng = self.headers.get("Range")
            if rng:
                start, end = (int(x) for x in rng.split("=")[1].split("-"))
                body = payload[start : end + 1]
                self.send_response(206)
            else:
                body = payload
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        item = dl.DownloadItem(
            f"http://127.0.0.1:{server.server_port}/start", tmp_path / "real.bin", len(payload), hashlib.sha256(payload).hexdigest()
        )
        dl.Downloader([item], chunk_bytes=1000, workers=4).run()
    finally:
        server.shutdown()
        server.server_close()
    assert item.target.read_bytes() == payload
