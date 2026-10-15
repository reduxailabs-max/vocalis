"""The vocalis background daemon: a long-lived process listening on a Unix
domain socket for commands from the `vocalis` CLI and hotkey layer.

This module only implements the socket lifecycle (start/stop/accept-loop).
Command handling for real features (dictation, quick reader, document
reader, OCR) is added in their own phases on top of this skeleton.
"""
import contextlib
import logging
import os
import socket
import threading
from pathlib import Path
from typing import Callable

from vocalis.model_manager import ModelManager

_CONNECTION_TIMEOUT_SECONDS = 1.0
_ACCEPT_THREAD_JOIN_TIMEOUT_SECONDS = 3.0


class DaemonAlreadyRunningError(Exception):
    """Raised when start() cannot safely bind the socket because an
    instance is already running.

    Two distinct cases raise this: start() called twice on the same
    VocalisDaemon object in this process, and -- detected by the
    connect-before-delete probe -- a different, real process already
    listening on the socket file. Either way, this instance never binds
    or owns the socket file, so its own stop() must not delete it.
    """


class VocalisDaemon:
    """Owns the Unix socket server and the ModelManager for one vocalis run."""

    def __init__(
        self,
        socket_path: Path,
        logger: logging.Logger,
        model_manager: ModelManager,
        socket_factory: "Callable[[], socket.socket]" = lambda: socket.socket(
            socket.AF_UNIX, socket.SOCK_STREAM
        ),
    ) -> None:
        """Create a VocalisDaemon.

        Args:
            socket_path: Filesystem path for the Unix domain socket.
            logger: Logger to use for lifecycle and per-connection events.
            model_manager: The ModelManager to shut down when the daemon stops.
            socket_factory: Constructs a fresh socket, used only to probe
                whether an existing socket file is still alive before
                deleting it as stale. Injectable for tests.
        """
        self._socket_path = socket_path
        self._logger = logger
        self._model_manager = model_manager
        self._socket_factory = socket_factory
        self._server_socket: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._owns_socket_file = False

    @property
    def is_running(self) -> bool:
        """Whether the daemon's accept loop is currently active."""
        return (
            self._server_socket is not None
            and self._accept_thread is not None
            and self._accept_thread.is_alive()
        )

    def start(self) -> None:
        """Start listening on the Unix socket and accepting connections.

        Removes a stale socket file left over from a previous crashed run
        before binding.

        Raises:
            DaemonAlreadyRunningError: If start() is called while already running.
        """
        if self.is_running:
            raise DaemonAlreadyRunningError(
                f"Daemon already running on {self._socket_path}"
            )

        if self._socket_path.exists():
            if self._is_another_instance_alive():
                raise DaemonAlreadyRunningError(
                    f"Another vocalis instance is already running on {self._socket_path}"
                )
            self._logger.warning(
                "Removing stale socket file at %s from a previous run", self._socket_path
            )
            self._socket_path.unlink()

        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_socket.bind(str(self._socket_path))
        self._owns_socket_file = True
        os.chmod(self._socket_path, 0o600)
        server_socket.listen(5)
        server_socket.settimeout(0.2)
        self._server_socket = server_socket
        self._stop_requested.clear()

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        self._logger.info("Vocalis daemon listening on %s", self._socket_path)

    def _is_another_instance_alive(self) -> bool:
        """Whether a real process is still listening on the existing socket file.

        Connects a short-lived probe socket to the existing file. A
        successful connect means another instance is genuinely alive.
        ConnectionRefusedError or FileNotFoundError means the file is a
        stale leftover from a crashed run. Any other error is ambiguous
        and is treated as "not alive" so startup is never blocked by an
        inconclusive probe.
        """
        probe = self._socket_factory()
        try:
            probe.settimeout(0.5)
            probe.connect(str(self._socket_path))
            return True
        except (ConnectionRefusedError, FileNotFoundError):
            return False
        except Exception as exc:
            self._logger.warning(
                "Could not verify whether an existing vocalis socket is alive "
                "(%s); treating it as stale",
                exc,
            )
            return False
        finally:
            with contextlib.suppress(Exception):
                probe.close()

    def stop(self) -> None:
        """Stop accepting connections, unload models, and clean up the socket file.

        Only deletes the socket file when THIS instance actually bound it
        (`_owns_socket_file`, set in start() right after a successful
        bind()). A call that never started (for example, a start() that
        was refused with DaemonAlreadyRunningError because another real
        instance already owns the file) must never delete that other
        instance's live socket file; closing the server socket, joining
        the accept thread, and shutting down the ModelManager still
        always run, since those are this instance's own resources
        regardless of whether it ever bound a socket.

        Guarantees ModelManager.shutdown() runs even if socket/file cleanup
        raises, since releasing VRAM is more important than clean cleanup.
        """
        self._stop_requested.set()
        try:
            if self._accept_thread is not None:
                self._accept_thread.join(timeout=_ACCEPT_THREAD_JOIN_TIMEOUT_SECONDS)
                if self._accept_thread.is_alive():
                    self._logger.warning(
                        "Accept-loop thread did not stop within %.1fs; "
                        "abandoning it as a daemon thread",
                        _ACCEPT_THREAD_JOIN_TIMEOUT_SECONDS,
                    )
                self._accept_thread = None

            if self._server_socket is not None:
                try:
                    self._server_socket.close()
                except OSError as exc:
                    self._logger.warning("Error closing server socket: %s", exc)
                self._server_socket = None

            if self._owns_socket_file:
                try:
                    if self._socket_path.exists():
                        self._socket_path.unlink()
                    self._owns_socket_file = False
                except OSError as exc:
                    self._logger.warning(
                        "Error removing socket file %s: %s", self._socket_path, exc
                    )
        finally:
            self._model_manager.shutdown()
            self._logger.info("Vocalis daemon stopped")

    def _accept_loop(self) -> None:
        """Accept connections until stop() is called, echoing each command."""
        assert self._server_socket is not None
        while not self._stop_requested.is_set():
            try:
                connection, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            connection.settimeout(_CONNECTION_TIMEOUT_SECONDS)
            try:
                self._handle_connection(connection)
            except (OSError, UnicodeDecodeError) as exc:
                self._logger.warning("Error handling connection: %s", exc)
            finally:
                connection.close()

    def _handle_connection(self, connection: socket.socket) -> None:
        """Read one line and echo it back as an acknowledgement.

        Real command dispatch (dictation/quick-reader/document-reader/ocr)
        is added by later phases; this skeleton only proves the socket
        round-trip works.
        """
        data = connection.recv(1024)
        if not data:
            return
        command = data.decode("utf-8").strip()
        self._logger.info("Received command: %s", command)
        connection.sendall(f"ack:{command}\n".encode("utf-8"))
