"""The Settings window: every setting, grouped into tabs.

Reads from and writes to the shared LiveSettings. Shortcuts and the voice
are chosen from drop-down lists so typos cannot happen. Closing the window
only hides it; the program keeps running in the tray.
"""
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vocalis.config import ConfigError, ShortcutsConfig, VocalisConfig
from vocalis.hotkeys.shortcut_parser import (
    CTRL_ALT_CHOICES,
    PUSH_TO_TALK_CHOICES,
    ShortcutError,
    ctrl_alt_choice,
    format_ctrl_alt_shortcut,
)
from vocalis.settings import KOKORO_VOICES, MODEL_PATH_FIELDS, VOICES, LiveSettings
from vocalis.ui.qwen3_panel import Qwen3Panel
from vocalis.ui.theme import mark_primary

_RESTART_NOTE = " Restart Vocalis to use the new shortcuts."
_AUTOMATIC_LABEL = "Automatic (recommended)"
_AUTOMATIC_VALUE = "auto"
_QWEN3_ENGINE = "qwen3_tts"
_QWEN3_MISSING_LABEL = "Qwen3-TTS (not installed – add it below)"
_ENGINE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Automatic (recommended)", "auto"),
    ("Qwen3-TTS (best quality, needs a strong PC)", "qwen3_tts"),
    ("Kokoro (light and natural)", "kokoro"),
    ("Piper (lightest)", "piper"),
)
_SPEED_SLIDER_MINIMUM = 70
_SPEED_SLIDER_MAXIMUM = 160
_SPEED_SCALE = 100.0
_CPU_THREADS_MAXIMUM = 128
_MAX_RAM_MAXIMUM_MB = 262144
_VRAM_CEILING_MINIMUM_MB = 256
_VRAM_CEILING_MAXIMUM_MB = 65536
_ENGINE_RESTART_NOTE = " Restart Vocalis to use the new model settings."
_IDLE_TIMEOUT_MINIMUM_S = 1
_IDLE_TIMEOUT_MAXIMUM_S = 3600
_EMPTY_PATH_TEXT = "(empty)"
_TAB_NAMES = ("General", "Voice & Engine", "Performance", "Shortcuts", "Model files")
# field -> (row label, noun used in the "not found" message, is a folder)
_MODEL_PATHS: dict[str, tuple[str, str, bool]] = {
    "parakeet_model_dir": ("Speech recognition model folder", "speech recognition model folder", True),
    "qwen3_tts_model_dir": ("Qwen3-TTS model folder", "Qwen3-TTS model folder", True),
    "piper_binary_path": ("Piper program", "Piper program", False),
    "piper_voice_model_path": ("Piper voice file", "Piper voice file", False),
    "kokoro_model_path": ("Kokoro model file", "Kokoro model file", False),
    "kokoro_voices_path": ("Kokoro voices file", "Kokoro voices file", False),
}
assert tuple(_MODEL_PATHS) == MODEL_PATH_FIELDS  # keeps the two lists from drifting apart


def _expanded(text: str) -> str:
    """A typed model location trimmed, with a leading "~" turned into the home folder."""
    text = text.strip()
    return str(Path(text).expanduser()) if text else ""


def _combo(items: tuple[str, ...]) -> QComboBox:
    """A drop-down list holding the given items."""
    box = QComboBox()
    box.addItems(list(items))
    return box


class SettingsWindow(QWidget):
    """Top-level window for changing Vocalis settings."""

    def __init__(
        self,
        settings: LiveSettings,
        on_applied: Callable[[], None] = lambda: None,
        output_names: Callable[[], list[str]] = lambda: [],
        on_test_sound: Callable[[str], str] = lambda _output: "",
        on_preview_voice: Callable[[], None] = lambda: None,
        qwen3_panel: Qwen3Panel | None = None,
    ) -> None:
        """Build the window and fill it from the current settings.

        Args:
            settings: The shared live settings this window edits.
            on_applied: Called after a successful save.
            output_names: Returns the names of the sound outputs currently
                available, to fill the sound output drop-down.
            on_test_sound: Plays a short test sound on the given output
                ("auto" or a device name) and returns a message describing
                what happened.
            on_preview_voice: Reads a short sample using the saved voice
                settings.
            qwen3_panel: The add/remove panel for the optional Qwen3-TTS
                voice (a real one is built when none is given).
        """
        super().__init__()
        self._settings = settings
        self._on_applied = on_applied
        self._output_names = output_names
        self._on_test_sound = on_test_sound
        self._on_preview_voice = on_preview_voice
        self.setWindowTitle("Vocalis — Settings")
        self._notifications_box = QCheckBox("Show desktop notifications")
        self._widget_box = QCheckBox("Show the floating widget while reading")
        self._overlay_box = QCheckBox("Show an animation at the bottom of the screen while dictating")
        self._dictation_output_combo = QComboBox()
        self._dictation_output_combo.addItem("Type it where my cursor is (recommended)", "type")
        self._dictation_output_combo.addItem("Type it and also copy it to the clipboard", "both")
        self._dictation_output_combo.addItem("Only copy it to the clipboard", "clipboard")
        self._dictation_output_combo.setToolTip(
            "If typing ever fails, the text is always saved to the clipboard."
        )
        self._ptt_combo = _combo(PUSH_TO_TALK_CHOICES)
        self._quick_combo = _combo(CTRL_ALT_CHOICES)
        self._doc_combo = _combo(CTRL_ALT_CHOICES)
        self._ocr_combo = _combo(CTRL_ALT_CHOICES)
        self._voice_combo = _combo(VOICES)
        self._quick_engine_combo = QComboBox()
        self._doc_engine_combo = QComboBox()
        for engine_combo in (self._quick_engine_combo, self._doc_engine_combo):
            for label, value in _ENGINE_CHOICES:
                engine_combo.addItem(label, value)
        self._qwen3_panel = qwen3_panel or Qwen3Panel(lambda: self._settings.config)
        self._qwen3_panel.changed.connect(self._refresh_qwen3_state)
        self._kokoro_voice_combo = _combo(tuple(KOKORO_VOICES))
        self._output_combo = QComboBox()
        self._test_sound_button = QPushButton("Test sound")
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setMinimum(_SPEED_SLIDER_MINIMUM)
        self._speed_slider.setMaximum(_SPEED_SLIDER_MAXIMUM)
        self._cpu_threads_spin = QSpinBox()
        self._cpu_threads_spin.setRange(0, _CPU_THREADS_MAXIMUM)
        self._max_ram_spin = QSpinBox()
        self._max_ram_spin.setRange(0, _MAX_RAM_MAXIMUM_MB)
        self._vram_ceiling_spin = QSpinBox()
        self._vram_ceiling_spin.setRange(_VRAM_CEILING_MINIMUM_MB, _VRAM_CEILING_MAXIMUM_MB)
        self._idle_timeout_spin = QSpinBox()
        self._idle_timeout_spin.setRange(_IDLE_TIMEOUT_MINIMUM_S, _IDLE_TIMEOUT_MAXIMUM_S)
        self._path_edits: dict[str, QLineEdit] = {field: QLineEdit() for field in MODEL_PATH_FIELDS}
        self._browse_buttons: dict[str, QPushButton] = {
            field: QPushButton("Browse…") for field in MODEL_PATH_FIELDS
        }
        self._tabs = QTabWidget()
        self._speed_label = QLabel("")
        self._preview_button = QPushButton("Preview voice (uses saved settings)")
        self._save_button = QPushButton("Save")
        self._close_button = QPushButton("Close")
        self._message_label = QLabel("")
        self._message_label.setWordWrap(True)
        mark_primary(self._save_button)
        self._build_layout()
        self._speed_slider.valueChanged.connect(self._on_speed_slider_changed)
        self._test_sound_button.clicked.connect(self._on_test_sound_clicked)
        self._preview_button.clicked.connect(self._on_preview_clicked)
        self._save_button.clicked.connect(self._on_save_clicked)
        self._close_button.clicked.connect(self.hide)
        for field, button in self._browse_buttons.items():
            button.clicked.connect(lambda _checked=False, name=field: self._on_browse_clicked(name))
        self.load_from_settings()

    def _build_layout(self) -> None:
        """Place the widgets on five tabs, with the message and Save/Close below."""
        for name, page in zip(_TAB_NAMES, (
            self._general_page(), self._voice_page(), self._performance_page(),
            self._shortcuts_page(), self._model_files_page(),
        )):
            self._tabs.addTab(page, name.replace("&", "&&"))
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._save_button)
        buttons.addWidget(self._close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs)
        layout.addWidget(self._message_label)
        layout.addLayout(buttons)

    @staticmethod
    def _page(form: QFormLayout, *extra: QWidget) -> QWidget:
        """A tab page holding a form and, under it, any extra widgets."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addLayout(form)
        for widget in extra:
            layout.addWidget(widget)
        layout.addStretch(1)
        return page

    def _general_page(self) -> QWidget:
        """Switches and the sound output."""
        form = QFormLayout()
        form.addRow(self._notifications_box)
        form.addRow(self._widget_box)
        form.addRow(self._overlay_box)
        form.addRow("Dictated text", self._dictation_output_combo)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_combo)
        output_row.addWidget(self._test_sound_button)
        form.addRow("Sound output", output_row)
        return self._page(form)

    def _voice_page(self) -> QWidget:
        """Voices, engines and speaking speed."""
        form = QFormLayout()
        form.addRow("Voice (Qwen3-TTS)", self._voice_combo)
        form.addRow("Quick Reader voice engine", self._quick_engine_combo)
        form.addRow("Document Reader voice engine", self._doc_engine_combo)
        form.addRow("Kokoro voice", self._kokoro_voice_combo)
        speed_row = QHBoxLayout()
        speed_row.addWidget(self._speed_slider)
        speed_row.addWidget(self._speed_label)
        form.addRow("Speaking speed", speed_row)
        return self._page(form, self._qwen3_panel, self._preview_button)

    def _performance_page(self) -> QWidget:
        """Resource limits and the idle-unload time."""
        form = QFormLayout()
        form.addRow("Max CPU threads (0 = no limit)", self._cpu_threads_spin)
        form.addRow("Max RAM, MB (0 = no limit)", self._max_ram_spin)
        form.addRow("GPU VRAM ceiling, MB", self._vram_ceiling_spin)
        form.addRow("Unload the voice after (seconds idle)", self._idle_timeout_spin)
        return self._page(form)

    def _shortcuts_page(self) -> QWidget:
        """The four keyboard shortcuts."""
        form = QFormLayout()
        form.addRow("Dictation (hold)", self._ptt_combo)
        form.addRow("Quick Reader  Ctrl+Alt+", self._quick_combo)
        form.addRow("Document Reader  Ctrl+Alt+", self._doc_combo)
        form.addRow("OCR Reader  Ctrl+Alt+", self._ocr_combo)
        return self._page(form)

    def _model_files_page(self) -> QWidget:
        """Where each model lives on disk."""
        form = QFormLayout()
        for field in MODEL_PATH_FIELDS:
            row = QHBoxLayout()
            row.addWidget(self._path_edits[field])
            row.addWidget(self._browse_buttons[field])
            form.addRow(_MODEL_PATHS[field][0], row)
        return self._page(form)

    @staticmethod
    def _choice_or_default(text: str, default_text: str) -> str:
        """The letter/digit in a stored shortcut, or the default's if unreadable."""
        try:
            return ctrl_alt_choice(text)
        except ShortcutError:
            return ctrl_alt_choice(default_text)

    def load_from_settings(self) -> None:
        """Fill every widget from the current settings and clear the old
        "Saved."/"Not saved" message (never raises)."""
        self._message_label.setText("")
        config = self._settings.config
        defaults = ShortcutsConfig()
        self._notifications_box.setChecked(config.notifications_enabled)
        self._widget_box.setChecked(config.floating_widget_enabled)
        self._overlay_box.setChecked(config.dictation_overlay_enabled)
        output_index = self._dictation_output_combo.findData(config.dictation_output)
        self._dictation_output_combo.setCurrentIndex(max(output_index, 0))
        ptt = config.shortcuts.numlock_key.strip().upper()
        self._ptt_combo.setCurrentText(ptt if ptt in PUSH_TO_TALK_CHOICES else defaults.numlock_key)
        for combo, stored, default in (
            (self._quick_combo, config.shortcuts.quick_reader, defaults.quick_reader),
            (self._doc_combo, config.shortcuts.document_reader, defaults.document_reader),
            (self._ocr_combo, config.shortcuts.ocr_reader, defaults.ocr_reader),
        ):
            combo.setCurrentText(self._choice_or_default(stored, default))
        voice = config.engines.qwen3_tts_speaker
        if self._voice_combo.findText(voice) < 0:
            self._voice_combo.addItem(voice)
        self._voice_combo.setCurrentText(voice)
        for engine_combo, stored in (
            (self._quick_engine_combo, config.engines.quick_reader_engine),
            (self._doc_engine_combo, config.engines.document_reader_engine),
        ):
            index = engine_combo.findData(stored)
            engine_combo.setCurrentIndex(index if index >= 0 else 0)
        kokoro_voice = config.engines.kokoro_voice
        if self._kokoro_voice_combo.findText(kokoro_voice) < 0:
            self._kokoro_voice_combo.addItem(kokoro_voice)
        self._kokoro_voice_combo.setCurrentText(kokoro_voice)
        self._fill_output_combo(config.audio_output_device)
        self._speed_slider.setValue(round(config.speaking_speed * _SPEED_SCALE))
        self._update_speed_label(self._speed_slider.value())
        self._cpu_threads_spin.setValue(config.max_cpu_threads)
        self._max_ram_spin.setValue(config.max_ram_mb)
        self._vram_ceiling_spin.setValue(config.gpu_vram_ceiling_mb)
        self._idle_timeout_spin.setValue(round(config.model_idle_timeout_seconds))
        for field, edit in self._path_edits.items():
            edit.setText(getattr(config.engines, field))
        self._qwen3_panel.refresh()
        self._refresh_qwen3_state()

    def _refresh_qwen3_state(self) -> None:
        """Grey out the Qwen3-TTS choices while it is not installed."""
        installed = self._qwen3_panel.is_installed()
        self._voice_combo.setEnabled(installed)
        for engine_combo in (self._quick_engine_combo, self._doc_engine_combo):
            index = engine_combo.findData(_QWEN3_ENGINE)
            model = engine_combo.model()
            if index < 0 or not isinstance(model, QStandardItemModel):
                continue
            default_label = next(label for label, value in _ENGINE_CHOICES if value == _QWEN3_ENGINE)
            engine_combo.setItemText(index, default_label if installed else _QWEN3_MISSING_LABEL)
            item = model.item(index)
            if item is not None:
                item.setEnabled(installed)

    def present(self) -> None:
        """Bring the window to the front; refresh it from saved settings only if it was hidden."""
        if not self.isVisible():
            self.load_from_settings()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _fill_output_combo(self, stored_device: str) -> None:
        """Fill the sound output drop-down with "Automatic" plus the
        currently available outputs, keeping a stored but no-longer-listed
        device name selectable."""
        self._output_combo.clear()
        self._output_combo.addItem(_AUTOMATIC_LABEL)
        self._output_combo.addItems(self._output_names())
        if stored_device != _AUTOMATIC_VALUE and self._output_combo.findText(stored_device) < 0:
            self._output_combo.addItem(stored_device)
        if stored_device == _AUTOMATIC_VALUE:
            self._output_combo.setCurrentIndex(0)
        else:
            self._output_combo.setCurrentText(stored_device)

    def _selected_output_value(self) -> str:
        """The config value for the currently chosen output: "auto" for
        the first item, else its name."""
        if self._output_combo.currentIndex() == 0:
            return _AUTOMATIC_VALUE
        return self._output_combo.currentText()

    def _update_speed_label(self, slider_value: int) -> None:
        """Show the slider's value as a speed multiplier, e.g. "1.15x"."""
        self._speed_label.setText(f"{slider_value / _SPEED_SCALE:.2f}x")

    def _on_speed_slider_changed(self, value: int) -> None:
        """Update the speed label live as the slider moves."""
        self._update_speed_label(value)

    def _on_test_sound_clicked(self) -> None:
        """Play a test sound on the selected output and show the result."""
        try:
            message = self._on_test_sound(self._selected_output_value())
        except Exception as exc:  # noqa: BLE001 - any engine failure is reported, not raised
            self._message_label.setText(f"Sound test failed: {exc}")
            return
        self._message_label.setText(message)

    def _on_preview_clicked(self) -> None:
        """Read a short sample using the saved voice settings."""
        try:
            self._on_preview_voice()
        except Exception as exc:  # noqa: BLE001 - any engine failure is reported, not raised
            self._message_label.setText(f"Preview failed: {exc}")

    def _on_browse_clicked(self, field: str) -> None:
        """Let the user pick a model location; cancelling keeps the old one."""
        label, _noun, is_folder = _MODEL_PATHS[field]
        current = self._path_edits[field].text()
        if is_folder:
            chosen = QFileDialog.getExistingDirectory(self, label, current)
        else:
            chosen, _filter = QFileDialog.getOpenFileName(self, label, current)
        if chosen:
            self._path_edits[field].setText(chosen)

    def _missing_model_location(self) -> str | None:
        """A "not saved" message for the first changed model location that
        does not exist, or None when every changed location is fine.

        Unchanged locations are not checked, so a PC where a model was never
        installed can still save its other settings.
        """
        engines = self._settings.config.engines
        for field in MODEL_PATH_FIELDS:
            text = self._path_edits[field].text().strip()
            if _expanded(text) == _expanded(getattr(engines, field)):
                continue
            _label, noun, is_folder = _MODEL_PATHS[field]
            path = Path(text).expanduser()
            exists = bool(text) and (path.is_dir() if is_folder else path.is_file())
            if not exists:
                return f"Not saved: the {noun} was not found at {text or _EMPTY_PATH_TEXT}."
        return None

    def _config_from_widgets(self) -> VocalisConfig:
        """Build a new config from the current settings and the widgets."""
        config = self._settings.config
        shortcuts = config.shortcuts.model_copy(
            update={
                "numlock_key": self._ptt_combo.currentText(),
                "quick_reader": format_ctrl_alt_shortcut(self._quick_combo.currentText()),
                "document_reader": format_ctrl_alt_shortcut(self._doc_combo.currentText()),
                "ocr_reader": format_ctrl_alt_shortcut(self._ocr_combo.currentText()),
            }
        )
        engines = config.engines.model_copy(
            update={
                "qwen3_tts_speaker": self._voice_combo.currentText(),
                "quick_reader_engine": self._quick_engine_combo.currentData(),
                "document_reader_engine": self._doc_engine_combo.currentData(),
                "kokoro_voice": self._kokoro_voice_combo.currentText(),
                **{field: _expanded(self._path_edits[field].text()) for field in MODEL_PATH_FIELDS},
            }
        )
        return config.model_copy(
            update={
                "notifications_enabled": self._notifications_box.isChecked(),
                "floating_widget_enabled": self._widget_box.isChecked(),
                "dictation_overlay_enabled": self._overlay_box.isChecked(),
                "dictation_output": self._dictation_output_combo.currentData(),
                "shortcuts": shortcuts,
                "engines": engines,
                "audio_output_device": self._selected_output_value(),
                "speaking_speed": self._speed_slider.value() / _SPEED_SCALE,
                "max_cpu_threads": self._cpu_threads_spin.value(),
                "max_ram_mb": self._max_ram_spin.value(),
                "gpu_vram_ceiling_mb": self._vram_ceiling_spin.value(),
                "model_idle_timeout_seconds": float(self._idle_timeout_spin.value()),
            }
        )

    def _on_save_clicked(self) -> None:
        """Validate and save; show the outcome in the message label."""
        problem = self._missing_model_location()
        if problem is not None:
            self._message_label.setText(problem)
            return
        try:
            self._settings.update(self._config_from_widgets())
        except (ShortcutError, ConfigError) as exc:
            self._message_label.setText(f"Not saved: {exc}")
            return
        message = "Saved."
        if self._settings.shortcuts_need_restart():
            message += _RESTART_NOTE
        if self._settings.engines_need_restart():
            message += _ENGINE_RESTART_NOTE
        self._message_label.setText(message)
        self._on_applied()
