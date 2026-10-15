"""Tests for vocalis.remote — the `vocalis --settings` client. Real sockets only in tmp_path."""
import socket
import threading
from pathlib import Path

from vocalis.remote import SHOW_SETTINGS, open_settings, send_command


def _serve_once(path: Path, replies: list[bytes]) -> threading.Thread:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)

    def run() -> None:
        conn, _ = server.accept()
        replies.append(conn.recv(1024))
        conn.sendall(b"ack:show-settings\n")
        conn.close()
        server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_send_command_delivers_and_reads_the_ack(tmp_path: Path) -> None:
    path: Path = tmp_path / "v.sock"
    got: list[bytes] = []
    thread = _serve_once(path, got)
    assert send_command(path, SHOW_SETTINGS) is True
    thread.join(2)
    assert got == [b"show-settings\n"]


def test_send_command_to_nothing_returns_false(tmp_path: Path) -> None:
    assert send_command(tmp_path / "missing.sock", SHOW_SETTINGS) is False


def test_open_settings_when_running_does_not_start_the_service(tmp_path: Path) -> None:
    started: list[int] = []

    def start_service() -> bool:
        started.append(1)
        return True

    code = open_settings(tmp_path / "v.sock", start_service=start_service,
                         send=lambda _p, _c: True, sleep=lambda _s: None)
    assert code == 0 and started == []


def test_open_settings_starts_the_service_then_retries_until_it_answers(tmp_path: Path) -> None:
    answers = iter([False, False, False, True])
    started: list[int] = []

    def start_service() -> bool:
        started.append(1)
        return True

    code = open_settings(tmp_path / "v.sock", start_service=start_service,
                         send=lambda _p, _c: next(answers), sleep=lambda _s: None)
    assert code == 0 and started == [1]


def test_open_settings_gives_up_after_the_wait_limit_with_a_plain_message(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    slept: list[float] = []
    code = open_settings(tmp_path / "v.sock", start_service=lambda: True, send=lambda _p, _c: False,
                         sleep=slept.append, wait_seconds=2.0, poll_seconds=0.5)
    assert code == 1
    assert sum(slept) <= 2.0
    assert "Vocalis did not start" in capsys.readouterr().err


def test_open_settings_reports_a_failed_service_start(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    code = open_settings(tmp_path / "v.sock", start_service=lambda: False,
                         send=lambda _p, _c: False, sleep=lambda _s: None)
    assert code == 1
    assert "could not start" in capsys.readouterr().err.lower()
