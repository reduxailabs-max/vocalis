"""Tests for VocalisDaemon's Unix socket lifecycle (start/stop/echo)."""
import os
import socket
import stat
import time
from pathlib import Path

import pytest

from vocalis.daemon import DaemonAlreadyRunningError, VocalisDaemon
from vocalis.logging_setup import configure_logging
from vocalis.model_manager import ModelManager


def make_daemon(tmp_path: Path) -> VocalisDaemon:
    logger = configure_logging()
    return VocalisDaemon(
        socket_path=tmp_path / "vocalis.sock",
        logger=logger,
        model_manager=ModelManager(),
    )


def test_start_creates_socket_file(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    try:
        assert (tmp_path / "vocalis.sock").exists()
        assert daemon.is_running is True
    finally:
        daemon.stop()


def test_stop_removes_socket_file(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    daemon.stop()
    assert not (tmp_path / "vocalis.sock").exists()
    assert daemon.is_running is False


def test_echo_roundtrip(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(tmp_path / "vocalis.sock"))
        client.sendall(b"ping\n")
        response = client.recv(1024)
        client.close()
        assert response == b"ack:ping\n"
    finally:
        daemon.stop()


def test_start_removes_stale_socket_file(tmp_path: Path) -> None:
    socket_path = tmp_path / "vocalis.sock"
    socket_path.write_text("stale file from a previous crashed run")
    daemon = VocalisDaemon(
        socket_path=socket_path,
        logger=configure_logging(),
        model_manager=ModelManager(),
    )
    daemon.start()
    try:
        assert daemon.is_running is True
    finally:
        daemon.stop()


def test_double_start_raises_daemon_already_running_error(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    try:
        with pytest.raises(DaemonAlreadyRunningError):
            daemon.start()
    finally:
        daemon.stop()


def test_stop_calls_model_manager_shutdown(tmp_path: Path) -> None:
    manager = ModelManager()
    shutdown_calls = []
    manager.shutdown = lambda: shutdown_calls.append(True)  # type: ignore[method-assign]
    daemon = VocalisDaemon(
        socket_path=tmp_path / "vocalis.sock", logger=configure_logging(), model_manager=manager
    )
    daemon.start()
    daemon.stop()
    assert shutdown_calls == [True]


def test_stop_returns_promptly_even_with_a_silent_connected_client(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(tmp_path / "vocalis.sock"))
    # Deliberately send nothing and don't close - simulates a hung/slow client.
    thread = daemon._accept_thread  # capture before stop() nulls it, to check it actually died
    start_time = time.monotonic()
    daemon.stop()
    elapsed = time.monotonic() - start_time
    client.close()
    assert thread is not None
    assert not thread.is_alive()
    assert elapsed < 2.5


def test_malformed_payload_does_not_kill_accept_loop(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    try:
        daemon.start()
        bad_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bad_client.connect(str(tmp_path / "vocalis.sock"))
        bad_client.sendall(b"\xff\xfe\x00invalid-utf8")
        bad_client.close()
        time.sleep(0.3)  # let the accept loop process and recover

        good_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        good_client.connect(str(tmp_path / "vocalis.sock"))
        good_client.sendall(b"ping\n")
        response = good_client.recv(1024)
        good_client.close()
        assert response == b"ack:ping\n"
    finally:
        daemon.stop()


def test_stop_calls_model_manager_shutdown_even_if_socket_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ModelManager()
    shutdown_calls = []
    manager.shutdown = lambda: shutdown_calls.append(True)  # type: ignore[method-assign]
    daemon = VocalisDaemon(
        socket_path=tmp_path / "vocalis.sock", logger=configure_logging(), model_manager=manager
    )
    daemon.start()

    def failing_unlink(self: Path) -> None:
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    daemon.stop()

    assert shutdown_calls == [True]


def test_start_sets_restrictive_socket_permissions(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    try:
        mode = stat.S_IMODE(os.stat(tmp_path / "vocalis.sock").st_mode)
        assert mode == 0o600
    finally:
        daemon.stop()


class _RefusingSocket:
    """A probe socket whose connect() always refuses, like a dead socket file."""

    def connect(self, _path: str) -> None:
        """Simulate connecting to a socket file with nothing listening."""
        raise ConnectionRefusedError("no one listening")

    def settimeout(self, _seconds: float) -> None:
        """No-op: a fake probe socket does not need a real timeout."""

    def close(self) -> None:
        """No-op: a fake probe socket has nothing to close."""


class _AcceptingSocket:
    """A probe socket whose connect() succeeds, like a real live instance."""

    def connect(self, _path: str) -> None:
        """Simulate connecting to a socket file with a live listener."""
        return None

    def settimeout(self, _seconds: float) -> None:
        """No-op: a fake probe socket does not need a real timeout."""

    def close(self) -> None:
        """No-op: a fake probe socket has nothing to close."""


class _OddlyFailingSocket:
    """A probe socket whose connect() raises something unexpected."""

    def connect(self, _path: str) -> None:
        """Simulate an ambiguous probe failure unrelated to liveness."""
        raise PermissionError("unexpected")

    def settimeout(self, _seconds: float) -> None:
        """No-op: a fake probe socket does not need a real timeout."""

    def close(self) -> None:
        """No-op: a fake probe socket has nothing to close."""


def test_start_with_no_existing_socket_file_works_as_before(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    assert daemon.is_running
    daemon.stop()


def test_a_genuinely_dead_socket_file_is_removed_and_start_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "vocalis.sock"
    path.write_bytes(b"")  # a leftover file, nothing listening
    daemon = VocalisDaemon(
        socket_path=path,
        logger=configure_logging(),
        model_manager=ModelManager(),
        socket_factory=lambda: _RefusingSocket(),  # type: ignore[arg-type, return-value]
    )
    daemon.start()
    assert daemon.is_running
    daemon.stop()


def test_a_genuinely_alive_socket_refuses_to_start(tmp_path: Path) -> None:
    path = tmp_path / "vocalis.sock"
    path.write_bytes(b"")
    daemon = VocalisDaemon(
        socket_path=path,
        logger=configure_logging(),
        model_manager=ModelManager(),
        socket_factory=lambda: _AcceptingSocket(),  # type: ignore[arg-type, return-value]
    )
    with pytest.raises(DaemonAlreadyRunningError, match=str(path)):
        daemon.start()
    assert path.exists()  # the live instance's file must survive
    assert not daemon.is_running


def test_a_refused_instances_stop_does_not_delete_the_live_instances_socket_file(
    tmp_path: Path,
) -> None:
    """A second, refused launch must never destroy the first instance's socket.

    Instance A genuinely starts and owns the socket file. Instance B,
    launched against the same path while A is alive, is refused by the
    real connect-before-delete probe. Calling stop() on the REFUSED
    instance B (which never successfully bound anything) must not delete
    the socket file -- and A must still be genuinely reachable afterward.
    """
    socket_path = tmp_path / "vocalis.sock"
    instance_a = VocalisDaemon(
        socket_path=socket_path, logger=configure_logging(), model_manager=ModelManager()
    )
    instance_a.start()
    try:
        instance_b = VocalisDaemon(
            socket_path=socket_path, logger=configure_logging(), model_manager=ModelManager()
        )
        with pytest.raises(DaemonAlreadyRunningError):
            instance_b.start()
        assert instance_b._owns_socket_file is False  # noqa: SLF001

        instance_b.stop()

        assert socket_path.exists()
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(socket_path))
        client.sendall(b"ping\n")
        response = client.recv(1024)
        client.close()
        assert response == b"ack:ping\n"
    finally:
        instance_a.stop()


def test_stop_resets_owns_socket_file_after_a_normal_start_and_stop(tmp_path: Path) -> None:
    """The normal case (own file, delete it on stop) still works unchanged."""
    daemon = make_daemon(tmp_path)
    daemon.start()
    assert daemon._owns_socket_file is True  # noqa: SLF001
    daemon.stop()
    assert not (tmp_path / "vocalis.sock").exists()
    assert daemon._owns_socket_file is False  # noqa: SLF001


def test_an_ambiguous_probe_failure_is_treated_as_stale_not_blocking(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "vocalis.sock"
    path.write_bytes(b"")
    daemon = VocalisDaemon(
        socket_path=path,
        logger=configure_logging(),
        model_manager=ModelManager(),
        socket_factory=lambda: _OddlyFailingSocket(),  # type: ignore[arg-type, return-value]
    )
    daemon.start()
    assert daemon.is_running
    assert "Could not verify" in caplog.text
    daemon.stop()
