"""The client side of `vocalis --settings`: ask the running Vocalis to show its
Settings window, starting the background service first if it is not running."""
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

SHOW_SETTINGS = "show-settings"
_START_TIMEOUT_SECONDS = 10.0


def _unix_socket() -> socket.socket:
    """A fresh AF_UNIX stream socket."""
    return socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)


def send_command(socket_path: Path, command: str, timeout: float = 2.0,
                  socket_factory: Callable[[], socket.socket] = _unix_socket) -> bool:
    """Send one command line; True only if the daemon answered with its ack."""
    client = socket_factory()
    try:
        client.settimeout(timeout)
        client.connect(str(socket_path))
        client.sendall(f"{command}\n".encode("utf-8"))
        return client.recv(1024).decode("utf-8", errors="replace").strip() == f"ack:{command}"
    except OSError:
        return False
    finally:
        client.close()


def start_user_service() -> bool:
    """Start the Vocalis user service; True if systemctl reported success."""
    try:
        result = subprocess.run(["systemctl", "--user", "start", "vocalis"],
                                 capture_output=True, timeout=_START_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def open_settings(socket_path: Path, start_service: Callable[[], bool] = start_user_service,
                   sleep: Callable[[float], None] = time.sleep, wait_seconds: float = 20.0,
                   poll_seconds: float = 0.5,
                   send: Callable[[Path, str], bool] = send_command) -> int:
    """Show Settings in the running Vocalis; start it if needed. Returns an exit code."""
    if send(socket_path, SHOW_SETTINGS):
        return 0
    if not start_service():
        print("Vocalis is not running and could not start it. "
              "Try: systemctl --user start vocalis", file=sys.stderr)
        return 1
    waited = 0.0
    while waited < wait_seconds:
        sleep(poll_seconds)
        waited += poll_seconds
        if send(socket_path, SHOW_SETTINGS):
            return 0
    print(f"Vocalis did not start within {int(wait_seconds)} seconds. "
          "Check: systemctl --user status vocalis", file=sys.stderr)
    return 1
