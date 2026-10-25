"""The one shared, live copy of the user's settings.

Created once at startup from the config file. The Settings window is the
only writer (via update()); features read it at the moment they need a
value (notifications on/off, widget on/off, voice) instead of copying it
at construction, so those settings apply immediately. Shortcuts are read
once at startup; changing them applies after a restart.
"""
import threading
from pathlib import Path
from typing import Callable

from vocalis.config import VocalisConfig, save_config
from vocalis.hotkeys.shortcut_parser import DEFAULT_SHORTCUTS, ShortcutError, ShortcutSet, parse_shortcuts

VOICES: tuple[str, ...] = (
    "Serena", "Vivian", "Uncle_Fu", "Ryan", "Aiden", "Ono_Anna", "Sohee", "Eric", "Dylan",
)

KOKORO_VOICES: tuple[str, ...] = (
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
)


class LiveSettings:
    """Thread-safe holder of the current VocalisConfig."""

    def __init__(
        self,
        config: VocalisConfig,
        path: Path,
        saver: Callable[[VocalisConfig, Path], None] = save_config,
    ) -> None:
        """Create the holder.

        Args:
            config: The config loaded at startup.
            path: Where update() saves the config file.
            saver: Writes a config to a path (default: the atomic
                save_config). Injectable for tests.
        """
        self._lock = threading.Lock()
        self._config = config
        self._started_with = config.shortcuts
        self._path = path
        self._saver = saver

    @property
    def config(self) -> VocalisConfig:
        """The current config (an immutable-by-convention snapshot)."""
        with self._lock:
            return self._config

    def notifications_enabled(self) -> bool:
        """Whether desktop notifications are currently switched on."""
        return self.config.notifications_enabled

    def floating_widget_enabled(self) -> bool:
        """Whether the floating widget is currently switched on."""
        return self.config.floating_widget_enabled

    def voice(self) -> str:
        """The TTS voice to use for the next reading."""
        return self.config.engines.qwen3_tts_speaker

    def quick_reader_engine(self) -> str:
        """Which voice engine Quick Reader should use ("auto" lets Vocalis pick)."""
        return self.config.engines.quick_reader_engine

    def document_reader_engine(self) -> str:
        """Which voice engine Document Reader should use ("auto" lets Vocalis pick)."""
        return self.config.engines.document_reader_engine

    def kokoro_voice(self) -> str:
        """The Kokoro voice to use for the next reading."""
        return self.config.engines.kokoro_voice

    def audio_output_device(self) -> str:
        """The chosen sound output ("auto" lets the program pick one)."""
        return self.config.audio_output_device

    def speaking_speed(self) -> float:
        """The speaking pace to use for the next reading."""
        return self.config.speaking_speed

    def max_cpu_threads(self) -> int | None:
        """The CPU thread cap to use for the next load, or None for no cap."""
        value = self.config.max_cpu_threads
        return value if value > 0 else None

    def max_ram_mb(self) -> int | None:
        """The advisory RAM ceiling used by Auto's stress check, or None for no cap."""
        value = self.config.max_ram_mb
        return value if value > 0 else None

    def gpu_vram_ceiling_mb(self) -> int:
        """The GPU VRAM ceiling to use for the next load."""
        return self.config.gpu_vram_ceiling_mb

    def shortcuts_need_restart(self) -> bool:
        """Whether the saved shortcuts mean something different from those in use.

        Compares the parsed keys, not the raw text, so other spellings of
        the same key do not count. A starting value that could not be
        parsed means the program is running on the defaults.
        """
        try:
            running: ShortcutSet = parse_shortcuts(self._started_with)
        except ShortcutError:
            running = DEFAULT_SHORTCUTS
        try:
            saved = parse_shortcuts(self.config.shortcuts)
        except ShortcutError:
            return True
        return saved != running

    def update(self, new_config: VocalisConfig) -> None:
        """Validate, save and then adopt a new config.

        The new config replaces the current one only after it has been
        validated and saved, so any failure leaves both the file and the
        in-memory settings as they were.

        Raises:
            ShortcutError: A shortcut is invalid or two features share one.
            ConfigError: The file could not be written.
        """
        parse_shortcuts(new_config.shortcuts)
        self._saver(new_config, self._path)
        with self._lock:
            self._config = new_config
