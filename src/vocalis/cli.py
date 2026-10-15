"""The `vocalis` command-line entrypoint.

Wires together config loading, logging, a single shared ModelManager, the
VocalisDaemon, the dictation feature, the quick reader feature, and the
document reader feature, then runs a PySide6 Qt event loop (app.exec()) on
the main thread until Ctrl+C. A SIGINT handler turns Ctrl+C into app.quit(),
so the loop returns normally and the shutdown code in main() runs.

Both SIGINT (Ctrl+C) and SIGTERM (systemd `stop`/`restart`) call app.quit()
through the identical handler pattern, so the shutdown code in main()'s
finally block runs the same way regardless of which signal arrived.
"""
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Callable, cast

import numpy as np
import numpy.typing as npt
import sounddevice as sd
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from vocalis.config import VocalisConfig, default_config_path, load_config
from vocalis.daemon import DaemonAlreadyRunningError, VocalisDaemon
from vocalis.engines.parakeet import ParakeetSTT
from vocalis.features.dictation import DictationFeature
from vocalis.hardware import log_hardware_summary
from vocalis.hotkeys.numlock import NumlockListener, list_numlock_capable_devices
from vocalis.logging_setup import configure_logging
from vocalis.model_manager import ModelManager
from vocalis.notify import Notifier
from vocalis.output.typer import TextOutput
from vocalis.remote import SHOW_SETTINGS, default_socket_path, open_settings
from vocalis.startup_checks import run_startup_checks
from vocalis.audio.capture import MicRecorder, pick_input_device
from vocalis.audio.output_device import AUTO, OutputDeviceChooser
from vocalis.audio.pace import apply_speed
from vocalis.audio.playback import AudioPlayer
from vocalis.engines.piper_tts import PiperTTS
from vocalis.engines.qwen3_tts import Qwen3TTS
from vocalis.features.quick_reader import QuickReaderFeature
from vocalis.hotkeys.ctrl_alt_r import CtrlAltRListener, list_hotkey_capable_devices
from vocalis.output.clipboard import ClipboardReader
from vocalis.features.doc_reader import DocumentReaderFeature
from vocalis.features.reading_coordinator import ReadingCoordinator
from vocalis.hotkeys.ctrl_alt_key import CtrlAltKeyListener, list_devices_with_key
from vocalis.ui.doc_popup import DocumentReaderLike, DocumentReaderPopup
from vocalis.engines.tesseract_ocr import TesseractOCR
from vocalis.features.ocr_reader import OcrReaderFeature
from vocalis.output.clipboard import ClipboardWriter
from vocalis.output.screen_capture import ScreenRegionCapture
from vocalis.hotkeys.shortcut_parser import DEFAULT_SHORTCUTS, ShortcutError, ShortcutSet, parse_shortcuts
from vocalis.settings import LiveSettings
from vocalis.ui.floating_widget import FloatingWidget
from vocalis.ui.settings_window import SettingsWindow
from vocalis.ui.status_hub import ReadingControls, StatusHub
from vocalis.ui.tray import TrayIcon

logger = logging.getLogger("vocalis")

_KEY_R = 19
_KEY_D = 32
_KEY_O = 24

PREVIEW_SENTENCE = "This is how Vocalis sounds with your current voice and speed."

_SOUND_TITLE = "Vocalis — Sound"
_DEVICE_CHANGED_MESSAGE = "Sound output changed because it stopped working"
_BUSY_MESSAGE = "A reading is playing right now — try again when it has finished."

_TEST_TONE_SECONDS = 0.4
_TEST_TONE_HZ = 440.0
_TEST_TONE_AMPLITUDE = 0.08
_TEST_TONE_SAMPLE_RATE = 24000
_TEST_SOUND_TIMEOUT_MARGIN_SECONDS = 2.0
_NO_OUTPUT_FOUND_MESSAGE = "Sound test failed: no working output found"

_WARM_UP_SAMPLE_RATE = 24000
_WARM_UP_SAMPLE_COUNT = 24000
_WARM_UP_SPEED = 1.1


def _warm_up_pace_changer() -> None:
    """Pay librosa/numba's first-use JIT cost now, not on the first reading.

    `apply_speed()` (the speaking-pace time-stretch) imports librosa lazily
    and librosa's phase vocoder is numba-JIT-compiled on its first call,
    which costs several seconds. Because the default speaking speed is
    1.15 (not the neutral 1.0), that cost would otherwise be paid by the
    very first reading of every run. Running it once here, on a second of
    silence, warms the JIT ahead of time so the user's first real reading
    is not delayed.

    Any failure (missing librosa, unexpected error, ...) is logged as a
    warning and swallowed: this is a best-effort optimization, never a
    requirement for Vocalis to work.
    """
    try:
        silence = np.zeros(_WARM_UP_SAMPLE_COUNT, dtype=np.float32)
        apply_speed(silence, _WARM_UP_SAMPLE_RATE, _WARM_UP_SPEED)
    except Exception:
        logger.warning("Failed to warm up the speaking-pace library", exc_info=True)


def _socket_path() -> Path:
    """Return the standard runtime location for the daemon's Unix socket."""
    return default_socket_path()


def build_daemon(
    model_manager: ModelManager,
    socket_path: Path,
    command_handler: Callable[[str], None] | None = None,
) -> VocalisDaemon:
    """Construct a VocalisDaemon using the given shared ModelManager.

    Args:
        model_manager: The ModelManager this daemon (and any features
            sharing it) will use. Constructed once by the caller so it can
            also be passed to build_dictation() — only one ModelManager
            may exist per running vocalis process, since it enforces
            single-model GPU residency across every feature.
        socket_path: Where the daemon's Unix socket should live.
        command_handler: Optional callable invoked with each command the
            daemon receives over its socket (e.g. to bridge "show-settings"
            onto the Qt main thread).

    Returns:
        A VocalisDaemon ready to have .start() called on it.
    """
    logger = configure_logging(level="INFO")
    return VocalisDaemon(
        socket_path=socket_path,
        logger=logger,
        model_manager=model_manager,
        command_handler=command_handler,
    )


def resolve_shortcuts(config: VocalisConfig) -> ShortcutSet:
    """Return the key codes for the configured shortcuts.

    A shortcut typed wrongly (or duplicated) in the config file must not
    stop Vocalis from starting, so the error is logged and the default
    shortcuts are used instead.
    """
    try:
        return parse_shortcuts(config.shortcuts)
    except ShortcutError as exc:
        logger.error("Invalid shortcut in the config (%s); using the default shortcuts", exc)
        return DEFAULT_SHORTCUTS


def _make_notifier(config: VocalisConfig, settings: LiveSettings | None) -> Notifier:
    """A Notifier that follows the live setting when there is one, else the config value."""
    if settings is not None:
        return Notifier(enabled=settings.notifications_enabled)
    return Notifier(enabled=config.notifications_enabled)


def _make_qwen_factory(
    config: VocalisConfig, settings: LiveSettings | None
) -> Callable[[], Qwen3TTS]:
    """A factory building Qwen3TTS; the voice is read live when settings are given."""
    qwen_model_dir = Path(config.engines.qwen3_tts_model_dir)
    speaker: "str | Callable[[], str]" = (
        settings.voice if settings is not None else config.engines.qwen3_tts_speaker
    )

    def primary_engine_factory() -> Qwen3TTS:
        """Build the primary text-to-speech engine."""
        return Qwen3TTS(model_dir=qwen_model_dir, speaker=speaker)

    return primary_engine_factory


_NEVER_NOTIFIED = object()


def _make_device_failure_hook(
    chooser: OutputDeviceChooser, notifier: Notifier
) -> Callable[["int | None"], None]:
    """Build an AudioPlayer `on_device_failed` hook tied to one chooser.

    The returned hook marks the failed device bad on `chooser` (so the
    next pick avoids it) every time it is called. It sends the desktop
    notification "Sound output changed because it stopped working"
    whenever the failing device is DIFFERENT from the last device it
    already notified about, but not twice in a row for the SAME repeated
    failing device: hook(2) notifies, hook(2) again does not, hook(5)
    notifies again (a different device), and so on. `None` (the system
    default) counts as a distinct "device" for this purpose, so a switch
    to or from it also triggers a fresh notification.

    Args:
        chooser: The shared OutputDeviceChooser to mark failures on.
        notifier: Sends the notification on every device switch.

    Returns:
        A callable suitable for AudioPlayer's `on_device_failed`.
    """
    last_notified: "list[object]" = [_NEVER_NOTIFIED]

    def on_device_failed(device: "int | None") -> None:
        """Mark `device` bad on the chooser; notify only on a device switch."""
        chooser.mark_failed(device)
        if device == last_notified[0]:
            return
        last_notified[0] = device
        notifier.notify(_SOUND_TITLE, _DEVICE_CHANGED_MESSAGE)

    return on_device_failed


def _test_tone() -> "npt.NDArray[np.float32]":
    """A short, quiet 440 Hz sine tone used to test a sound output."""
    sample_count = int(_TEST_TONE_SAMPLE_RATE * _TEST_TONE_SECONDS)
    times = np.linspace(0.0, _TEST_TONE_SECONDS, sample_count, endpoint=False)
    wave = _TEST_TONE_AMPLITUDE * np.sin(2.0 * np.pi * _TEST_TONE_HZ * times)
    return wave.astype(np.float32)


def _device_name(index: int) -> str:
    """Return the real name of output device `index` (sounddevice's query)."""
    return str(sd.query_devices(index)["name"])


def _named_device_index(chooser: OutputDeviceChooser, name: str) -> "int | None":
    """Return the index of `chooser`'s output device named `name`, else None.

    Matching is case-insensitive. This is used to find exactly the one
    device the user asked for, so "Test sound" can probe only that
    device -- never a fallback to pulse/pipewire/default/other.
    """
    for index, device_name in chooser._output_devices():  # noqa: SLF001
        if device_name.lower() == name.lower():
            return index
    return None


def _make_on_test_sound(
    document_reader: DocumentReaderLike, output_chooser: OutputDeviceChooser
) -> Callable[[str], str]:
    """Build the Settings window's "Test sound" callback.

    Design choice: resolving a *named* output (anything other than
    "auto") uses a fresh, one-off `OutputDeviceChooser(preference=lambda:
    selected)` rather than reaching into `output_chooser`'s private
    collaborators directly -- but the one-off chooser's own `.probe()`
    is called on exactly the matching device, never its `.current()`
    (which would silently fall through to pulse/pipewire/default/other
    if the requested device failed). Tests patch the `OutputDeviceChooser`
    class (and `AudioPlayer`) so no real sound device is ever touched.

    Args:
        document_reader: Checked for `is_reading` as a busy guard, so a
            test tone never interrupts an in-progress reading.
        output_chooser: The app's shared chooser, used for "auto".

    Returns:
        A callable(selected) -> message, matching SettingsWindow's
        `on_test_sound`.
    """

    def on_test_sound(selected: str) -> str:
        """Play a short test tone on `selected` ("auto" or a device name).

        Returns one of three kinds of message:
        - the busy message, with no stream opened at all, if a reading
          is already active;
        - "Sound test failed: no working output found" if no device
          could be resolved at all (checked BEFORE any AudioPlayer is
          constructed, so a silently stalling system default never opens
          a stream and freezes the window);
        - for a named selection, "Sound test failed: <name> did not
          respond" if that one device (and only that one -- no
          fallback) failed its probe;
        - "Sound test OK (<real device name>)" once the tone actually
          played on the resolved device;
        - "Sound test failed: ..." if playing the tone itself raised
          (including a PlaybackTimeoutError).
        """
        if document_reader.is_reading:
            return _BUSY_MESSAGE
        device: int
        if selected == AUTO:
            resolved = output_chooser.current()
            if resolved is None:
                return _NO_OUTPUT_FOUND_MESSAGE
            device = resolved
        else:
            probe_chooser = OutputDeviceChooser(preference=lambda: selected)
            index = _named_device_index(probe_chooser, selected)
            if index is None:
                return _NO_OUTPUT_FOUND_MESSAGE
            if not probe_chooser.probe(index):
                return f"Sound test failed: {selected} did not respond"
            device = index
        try:
            player = AudioPlayer(
                device_provider=lambda: device,
                timeout_margin_seconds=_TEST_SOUND_TIMEOUT_MARGIN_SECONDS,
            )
            player.play(_test_tone(), _TEST_TONE_SAMPLE_RATE)
            player.wait_until_done()
        except Exception as exc:  # noqa: BLE001 - any failure is reported, not raised
            return f"Sound test failed: {exc}"
        return f"Sound test OK ({_device_name(device)})"

    return on_test_sound


def _make_on_preview_voice(document_reader: DocumentReaderLike) -> Callable[[], None]:
    """Build the Settings window's "Preview voice" callback.

    Args:
        document_reader: Reads PREVIEW_SENTENCE aloud; this already
            preempts any current reading via the ReadingCoordinator and
            shows the Document Reader window as usual.

    Returns:
        A no-argument callable matching SettingsWindow's `on_preview_voice`.
    """

    def on_preview_voice() -> None:
        """Read PREVIEW_SENTENCE aloud using the saved voice and speed."""
        document_reader.read_text(PREVIEW_SENTENCE)

    return on_preview_voice


def build_dictation(
    config: VocalisConfig,
    model_manager: ModelManager,
    settings: LiveSettings | None = None,
    hub: StatusHub | None = None,
    key_code: int = 69,
) -> tuple[DictationFeature, NumlockListener]:
    """Construct the dictation feature and its NUMLOCK listener from config.

    Args:
        config: Loaded, validated vocalis configuration.
        model_manager: The shared ModelManager (same instance passed to
            build_daemon()).
        settings: Optional live settings; notifications then follow them.
        hub: Optional status board told when listening starts/stops.
        key_code: The evdev push-to-talk key (default NUMLOCK, 69).

    Returns:
        A (feature, listener) pair. The caller is responsible for calling
        listener.start()/listener.stop() around the daemon's own lifecycle.
    """
    model_dir = Path(config.engines.parakeet_model_dir)

    def engine_factory() -> ParakeetSTT:
        """Build the speech-to-text engine."""
        return ParakeetSTT(model_dir=model_dir)

    recorder = MicRecorder(max_duration_seconds=60.0, device=pick_input_device())
    text_output = TextOutput()
    notifier = _make_notifier(config, settings)
    feature = DictationFeature(
        model_manager=model_manager,
        engine_factory=engine_factory,
        recorder=recorder,
        text_output=text_output,
        notifier=notifier,
        on_listening_changed=hub.set_listening if hub is not None else None,
    )
    listener = NumlockListener(
        on_press=feature.on_press,
        on_release=feature.on_release,
        device_enumerator=lambda: list_numlock_capable_devices(key_code),
        key_code=key_code,
    )
    return feature, listener


def build_quick_reader(
    config: VocalisConfig,
    model_manager: ModelManager,
    reading_coordinator: ReadingCoordinator,
    settings: LiveSettings | None = None,
    hub: StatusHub | None = None,
    key_code: int = 19,
    output_chooser: OutputDeviceChooser | None = None,
) -> tuple[QuickReaderFeature, CtrlAltKeyListener]:
    """Construct the Quick Reader feature and its Ctrl+Alt+R listener.

    Args:
        config: Loaded, validated vocalis configuration.
        model_manager: The shared ModelManager (same instance passed to
            build_daemon() and build_dictation()) — only the primary
            (Qwen3-TTS) engine is routed through it; the Piper fallback
            never touches the GPU and is constructed/loaded directly.
        reading_coordinator: Shared with build_doc_reader() so Quick
            Reader and Document Reader never play audio at the same time.
        settings: Optional live settings (voice, speed and notifications).
        hub: Optional status board told when a reading begins/ends.
        key_code: The evdev key used with Ctrl+Alt (default R, 19).
        output_chooser: Optional shared OutputDeviceChooser. When given,
            the player uses its `current()` as the device provider and
            marks a failed device on it (notifying once); when None, the
            player keeps today's plain AudioPlayer() with the system
            default device.

    Returns:
        A (feature, listener) pair. The caller is responsible for calling
        listener.start()/listener.stop() around the daemon's own lifecycle.
    """
    primary_engine_factory = _make_qwen_factory(config, settings)

    fallback_engine = PiperTTS(
        binary_path=Path(config.engines.piper_binary_path),
        model_path=Path(config.engines.piper_voice_model_path),
    )
    fallback_engine.load()
    notifier = _make_notifier(config, settings)
    if output_chooser is not None:
        player = AudioPlayer(
            device_provider=output_chooser.current,
            on_device_failed=_make_device_failure_hook(output_chooser, notifier),
        )
    else:
        player = AudioPlayer()

    def on_reading_changed(active: bool) -> None:
        """Show/hide the reading on the status board."""
        if hub is None:
            return
        if active:
            hub.begin_reading("Quick read", ReadingControls(stop=feature.stop))
        else:
            hub.end_reading()

    feature = QuickReaderFeature(
        model_manager=model_manager,
        primary_engine_factory=primary_engine_factory,
        fallback_engine=fallback_engine,
        clipboard_reader=ClipboardReader(),
        player=player,
        notifier=notifier,
        reading_coordinator=reading_coordinator,
        on_reading_changed=on_reading_changed if hub is not None else None,
        speed_provider=settings.speaking_speed if settings is not None else None,
    )
    listener: CtrlAltKeyListener
    if key_code == _KEY_R:
        listener = CtrlAltRListener(
            on_trigger=feature.on_trigger,
            device_enumerator=list_hotkey_capable_devices,
        )
    else:
        listener = CtrlAltKeyListener(
            key_code=key_code,
            on_trigger=feature.on_trigger,
            device_enumerator=lambda: list_devices_with_key(key_code),
        )
    return feature, listener


def build_doc_reader(
    config: VocalisConfig,
    model_manager: ModelManager,
    reading_coordinator: ReadingCoordinator,
    settings: LiveSettings | None = None,
    hub: StatusHub | None = None,
    key_code: int = _KEY_D,
    output_chooser: OutputDeviceChooser | None = None,
) -> tuple[DocumentReaderFeature, DocumentReaderPopup, CtrlAltKeyListener]:
    """Construct the Document Reader feature, its popup, and its
    Ctrl+Alt+D listener.

    The listener runs on its own evdev thread, so it is wired to the
    popup's `hotkey_pressed` signal (never to a widget method): Qt then
    delivers the call on the GUI thread.

    The feature is constructed before the popup that will display its
    status, but the feature's on_status/on_progress/on_finished
    callbacks need to call methods on that not-yet-created popup. This
    is resolved with a closure over a `popup` variable assigned after
    the feature is built: by the time any callback is actually
    *invoked* (only once a reading starts, which cannot happen before
    this function returns and the caller starts the listener), `popup`
    is already bound. Python closures capture variables by reference,
    not by value at definition time, so this ordering is safe.

    Args:
        config: Loaded, validated vocalis configuration.
        model_manager: The shared ModelManager (same instance passed to
            build_daemon(), build_dictation(), and build_quick_reader()).
        reading_coordinator: Shared with build_quick_reader() so Quick
            Reader and Document Reader never play audio at the same time.
        settings: Optional live settings (voice and notifications).
        hub: Optional status board that also receives status, progress,
            pause and finish reports (in addition to the popup).
        key_code: The evdev key used with Ctrl+Alt (default D, 32).
        output_chooser: Optional shared OutputDeviceChooser. When given,
            the player uses its `current()` as the device provider and
            marks a failed device on it (notifying once); when None, the
            player keeps today's plain AudioPlayer() with the system
            default device.

    Returns:
        A (feature, popup, listener) triple. The caller is responsible
        for calling listener.start()/listener.stop() around the daemon's
        own lifecycle; the popup itself needs no start()/stop() — it
        stays hidden until the user first presses Ctrl+Alt+D.
    """
    primary_engine_factory = _make_qwen_factory(config, settings)

    fallback_engine = PiperTTS(
        binary_path=Path(config.engines.piper_binary_path),
        model_path=Path(config.engines.piper_voice_model_path),
    )
    fallback_engine.load()
    notifier = _make_notifier(config, settings)
    if output_chooser is not None:
        player = AudioPlayer(
            device_provider=output_chooser.current,
            on_device_failed=_make_device_failure_hook(output_chooser, notifier),
        )
    else:
        player = AudioPlayer()

    popup: DocumentReaderPopup

    def on_status(message: str) -> None:
        """Forward a status message to the popup (and the hub)."""
        popup.status_changed.emit(message)
        if hub is None:
            return
        hub.report_status(message)
        if message == "Paused":
            hub.set_paused(True)
        elif message == "Resuming...":
            hub.set_paused(False)

    def on_progress(current: int, total: int) -> None:
        """Forward chunk progress to the popup (and the hub)."""
        popup.progress_changed.emit(current, total)
        if hub is not None:
            hub.report_progress(current, total)

    def on_finished() -> None:
        """Tell the popup (and the hub) the reading ended."""
        popup.reading_finished.emit()
        if hub is not None:
            hub.end_reading()

    def on_started() -> None:
        """Tell the hub a document reading began."""
        if hub is None:
            return
        hub.begin_reading(
            "Document",
            ReadingControls(stop=feature.stop, pause=feature.pause, resume=feature.resume),
        )

    feature = DocumentReaderFeature(
        model_manager=model_manager,
        primary_engine_factory=primary_engine_factory,
        fallback_engine=fallback_engine,
        player=player,
        notifier=notifier,
        reading_coordinator=reading_coordinator,
        on_status=on_status,
        on_progress=on_progress,
        on_finished=on_finished,
        on_started=on_started if hub is not None else None,
        speed_provider=settings.speaking_speed if settings is not None else None,
    )
    popup = DocumentReaderPopup(feature)
    listener = CtrlAltKeyListener(
        key_code=key_code,
        on_trigger=popup.hotkey_pressed.emit,
        device_enumerator=lambda: list_devices_with_key(key_code),
    )
    return feature, popup, listener


def build_ocr_reader(
    config: VocalisConfig,
    document_reader: DocumentReaderFeature,
    popup: DocumentReaderPopup,
    settings: LiveSettings | None = None,
    hub: StatusHub | None = None,
    key_code: int = _KEY_O,
) -> tuple[OcrReaderFeature, CtrlAltKeyListener]:
    """Construct the OCR Reader feature and its Ctrl+Alt+O listener.

    The OCR Reader speaks through the already-built Document Reader, so it
    shares that reader's voices, pause/stop and ReadingCoordinator. Its
    "reading is starting" hook is wired to the popup's
    `external_reading_started` signal so the Document Reader window shows
    the recognised text with Pause/Stop enabled. The hook is a signal
    emit, so it is safe to call from the OCR background thread.

    Args:
        config: Loaded, validated vocalis configuration.
        document_reader: The Document Reader built by build_doc_reader().
        popup: The Document Reader popup built by build_doc_reader().
        settings: Optional live settings; when the floating widget is
            switched on the popup is not opened for OCR readings.
        hub: Optional status board; the reading is labelled "Screen text".
        key_code: The evdev key used with Ctrl+Alt (default O, 24).

    Returns:
        A (feature, listener) pair. The caller starts/stops the listener.
    """
    def on_reading_starting(text: str) -> None:
        """Label the reading and open the popup unless the widget shows it."""
        if hub is not None:
            hub.set_next_source("Screen text")
        if settings is not None and settings.floating_widget_enabled():
            return
        popup.external_reading_started.emit(text)

    feature = OcrReaderFeature(
        capture=ScreenRegionCapture(),
        ocr=TesseractOCR(),
        clipboard=ClipboardWriter(),
        reader=document_reader,
        notifier=_make_notifier(config, settings),
        on_reading_starting=on_reading_starting,
    )
    listener = CtrlAltKeyListener(
        key_code=key_code,
        on_trigger=feature.on_trigger,
        device_enumerator=lambda: list_devices_with_key(key_code),
    )
    return feature, listener


class _SettingsBridge(QObject):
    """Carries 'show Settings' requests from the socket thread to the Qt thread."""

    show_settings = Signal()

    def handle_command(self, command: str) -> None:
        """Called on the daemon's socket thread for every received command."""
        if command == SHOW_SETTINGS:
            self.show_settings.emit()


def build_ui(
    settings: LiveSettings,
    hub: StatusHub,
    doc_popup: DocumentReaderPopup,
    on_quit: Callable[[], None],
    output_chooser: OutputDeviceChooser | None = None,
) -> tuple[FloatingWidget, SettingsWindow, TrayIcon]:
    """Build the floating widget, the Settings window and the tray icon.

    Saving in the Settings window re-checks the widget's on/off switch at
    once. Note: with the widget enabled, an OCR reading still sends status,
    progress and finished reports to the Document Reader popup although the
    popup never receives the OCR text; this is by design.

    Args:
        settings: The shared live settings.
        hub: The status board all three pieces listen to.
        doc_popup: The Document Reader window the tray can open. Its
            Document Reader feature is also reused for the Settings
            window's "Test sound" busy guard and "Preview voice" button.
        on_quit: Called by the tray's Quit entry.
        output_chooser: Optional shared OutputDeviceChooser used to fill
            the Settings window's sound output drop-down and wire its
            "Test sound" button. None (the default, used by callers and
            tests that predate sound output settings) keeps the Settings
            window's own built-in defaults (no outputs listed, no-op
            test sound).

    Returns:
        The (widget, settings window, tray icon) triple; the caller shows
        the tray and tears all three down.
    """
    widget = FloatingWidget(hub, settings.floating_widget_enabled)
    document_reader = doc_popup._feature  # noqa: SLF001 - shared with Test sound / Preview voice
    if output_chooser is not None:
        settings_window = SettingsWindow(
            settings,
            on_applied=widget.apply_enabled,
            output_names=output_chooser.output_names,
            on_test_sound=_make_on_test_sound(document_reader, output_chooser),
            on_preview_voice=_make_on_preview_voice(document_reader),
        )
    else:
        settings_window = SettingsWindow(settings, on_applied=widget.apply_enabled)

    def open_document_reader() -> None:
        """Show the Document Reader window."""
        doc_popup.show()
        doc_popup.raise_()

    def open_settings() -> None:
        """Show the Settings window, reusing saved values only if it was hidden."""
        settings_window.present()

    tray = TrayIcon(
        hub,
        on_open_document_reader=open_document_reader,
        on_open_settings=open_settings,
        on_quit=on_quit,
    )
    return widget, settings_window, tray


def _stop_all_independently(stops: "list[tuple[str, Callable[[], object]]]") -> None:
    """Call every stop callable in order; log and continue if one raises.

    Ensures a failing listener stop can never prevent later stops (the
    daemon's, which shuts down ModelManager, is passed last).

    Args:
        stops: (human-readable name, callable) pairs, in the order to run.
    """
    for name, stop in stops:
        try:
            stop()
        except Exception:
            logger.exception("Error while stopping %s", name)


def main() -> None:
    """Entrypoint for the `vocalis` console script.

    Loads config, starts the daemon and all four hotkey listeners
    (dictation's NumlockListener, quick reader's CtrlAltRListener,
    document reader's Ctrl+Alt+D CtrlAltKeyListener, OCR Reader's
    Ctrl+Alt+O CtrlAltKeyListener), and runs the PySide6 Qt event
    loop until interrupted, at which point it stops everything cleanly.
    Every start()/stop() call is inside the try/finally, so daemon.stop()
    (and therefore ModelManager.shutdown()) always runs on the way out.

    If daemon.start() raises DaemonAlreadyRunningError (another real
    vocalis instance already owns the socket), that is caught, logged at
    INFO level, and main() returns normally -- the same try/finally still
    runs its cleanup on the way out, but nothing was actually started yet
    (no listener, no tray) so there is nothing real to undo. This keeps a
    second launch quiet (no traceback, exit code 0) instead of spamming
    the user or counting as a crash against systemd's StartLimitBurst.

    Right after the config is loaded, run_startup_checks() runs its
    non-fatal checks (missing models, ydotoold, OCR tools) and any
    problem it reports is sent as a desktop notification through a
    fresh Notifier built from the config's notifications setting --
    this happens before the QApplication, hub or any listener exists,
    so it must not depend on anything built later in this function.

    Right after the OutputDeviceChooser is created, a daemon background
    thread runs _warm_up_pace_changer() so librosa/numba's first-use JIT
    cost is paid before the user's first reading, not during it.

    The Qt event loop (app.exec()) replaces the previous bare sleep loop
    -- Document Reader's popup needs one running Qt application in this
    process, and there is exactly one process, so it runs on the main
    thread here. Ctrl+C is handled by a SIGINT handler that calls
    app.quit(), which makes app.exec() return normally so the finally
    block runs. SIGTERM (sent by systemd on `stop`/`restart`) is handled
    identically, by the same app.quit() lambda. The keepalive_timer
    exists only to give Python a chance to run those handlers: Qt's C++
    event loop does not return control to the Python interpreter unless
    something Python-level runs periodically. (A KeyboardInterrupt raised
    inside a Qt slot would be printed and swallowed by PySide6, so it
    cannot unwind exec().)
    The keepalive_timer is stopped and deleted on exit so it cannot fire
    later against a stale Qt state.
    The previous SIGINT and SIGTERM handlers are both restored on the way out.

    The user interface: a StatusHub (created after the QApplication, which
    must exist first), the floating widget, the Settings window and the
    system tray icon are built by build_ui(). quitOnLastWindowClosed is
    set to False so closing the Settings or Document Reader window never
    quits Vocalis; only the tray's Quit entry or Ctrl+C does. Teardown
    order on the way out, all while the QApplication is still alive:
    the keepalive timer first, then the tray icon (closed first, also
    when Quit is chosen), the floating widget, the listeners and the
    daemon, and last the Settings window and the widget window. Each
    step is independent, so one failing cannot stop the rest.
    """
    config = load_config(default_config_path())
    startup_notifier = Notifier(enabled=config.notifications_enabled)
    for message in run_startup_checks(config):
        startup_notifier.notify("Vocalis", message)
    settings = LiveSettings(config, default_config_path())
    output_chooser = OutputDeviceChooser(preference=settings.audio_output_device)
    threading.Thread(target=_warm_up_pace_changer, daemon=True).start()
    codes = resolve_shortcuts(config)
    app = cast(QApplication, QApplication.instance() or QApplication(sys.argv))
    app.setQuitOnLastWindowClosed(False)
    hub = StatusHub()
    bridge = _SettingsBridge()
    model_manager = ModelManager(idle_timeout_seconds=config.model_idle_timeout_seconds)
    reading_coordinator = ReadingCoordinator()
    daemon = build_daemon(
        model_manager=model_manager, socket_path=_socket_path(), command_handler=bridge.handle_command
    )
    threading.Thread(target=log_hardware_summary, daemon=True).start()
    _, dictation_listener = build_dictation(
        config=config, model_manager=model_manager, settings=settings, hub=hub,
        key_code=codes.push_to_talk,
    )
    _, quick_reader_listener = build_quick_reader(
        config=config, model_manager=model_manager, reading_coordinator=reading_coordinator,
        settings=settings, hub=hub, key_code=codes.quick_reader, output_chooser=output_chooser,
    )
    doc_reader_feature, doc_popup, doc_reader_listener = build_doc_reader(
        config=config, model_manager=model_manager, reading_coordinator=reading_coordinator,
        settings=settings, hub=hub, key_code=codes.document_reader, output_chooser=output_chooser,
    )
    _, ocr_listener = build_ocr_reader(
        config=config, document_reader=doc_reader_feature, popup=doc_popup,
        settings=settings, hub=hub, key_code=codes.ocr_reader,
    )

    def quit_app() -> None:
        """Close the tray first (while Qt is alive), then quit."""
        tray.close()
        app.quit()

    widget, settings_window, tray = build_ui(settings, hub, doc_popup, quit_app, output_chooser)
    bridge.show_settings.connect(settings_window.present)
    previous_sigint_handler: "signal.Handlers | int | Callable[..., object] | None" = None
    previous_sigterm_handler: "signal.Handlers | int | Callable[..., object] | None" = None
    keepalive_timer: QTimer | None = None
    try:
        previous_sigint_handler = signal.signal(signal.SIGINT, lambda *_: app.quit())
        previous_sigterm_handler = signal.signal(signal.SIGTERM, lambda *_: app.quit())
        keepalive_timer = QTimer()
        keepalive_timer.timeout.connect(lambda: None)
        keepalive_timer.start(200)
        daemon.start()
        dictation_listener.start()
        quick_reader_listener.start()
        doc_reader_listener.start()
        ocr_listener.start()
        tray.show()
        app.exec()
    except KeyboardInterrupt:
        pass
    except DaemonAlreadyRunningError as exc:
        logger.info("Vocalis is already running; exiting. (%s)", exc)
    finally:
        if keepalive_timer is not None:
            keepalive_timer.stop()
            keepalive_timer.deleteLater()
        if previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, previous_sigint_handler)
        if previous_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
        _stop_all_independently(
            [
                ("tray icon", tray.close),
                ("floating widget", widget.hide),
                ("ocr reader listener", ocr_listener.stop),
                ("document reader listener", doc_reader_listener.stop),
                ("quick reader listener", quick_reader_listener.stop),
                ("dictation listener", dictation_listener.stop),
                ("daemon", daemon.stop),
                ("settings window", settings_window.close),
                ("floating widget window", widget.close),
            ]
        )
