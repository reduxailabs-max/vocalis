"""The client side of `vocalis --settings`: ask the running Vocalis to show its
Settings window, starting the background service first if it is not running."""
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from vocalis.config import default_config_path, load_config
from vocalis.notify import Notifier

SHOW_SETTINGS = "show-settings"
_START_TIMEOUT_SECONDS = 10.0
_NOTIFY_TITLE = "Vocalis"


def default_socket_path() -> Path:
    """Standard runtime location for the daemon's Unix socket."""
    return Path.home() / ".local" / "state" / "vocalis" / "vocalis.sock"


def _default_notifier() -> Notifier:
    """A Notifier honoring the user's saved notification preference, or
    enabled by default if the config can't be read."""
    try:
        config = load_config(default_config_path())
        return Notifier(enabled=config.notifications_enabled)
    except Exception:
        return Notifier(enabled=True)


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
                   send: Callable[[Path, str], bool] = send_command,
                   notifier: Notifier | None = None) -> int:
    """Show Settings in the running Vocalis; start it if needed. Returns an exit code."""
    notify_target = notifier if notifier is not None else _default_notifier()
    if send(socket_path, SHOW_SETTINGS):
        return 0
    if not start_service():
        print("Vocalis is not running and could not start it. "
              "Try: systemctl --user start vocalis", file=sys.stderr)
        notify_target.notify(_NOTIFY_TITLE, "Vocalis could not start. Try: systemctl --user status vocalis")
        return 1
    waited = 0.0
    while waited < wait_seconds:
        sleep(poll_seconds)
        waited += poll_seconds
        if send(socket_path, SHOW_SETTINGS):
            return 0
    print(f"Vocalis did not start within {int(wait_seconds)} seconds. "
          "Check: systemctl --user status vocalis", file=sys.stderr)
    notify_target.notify(_NOTIFY_TITLE, "Vocalis is taking a long time to start. Check: systemctl --user status vocalis")
    return 1
