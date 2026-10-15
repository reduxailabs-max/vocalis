"""The Settings window: switches, shortcut drop-downs and the voice choice.

Reads from and writes to the shared LiveSettings. Shortcuts and the voice
are chosen from drop-down lists so typos cannot happen. Closing the window
only hides it; the program keeps running in the tray.
"""
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
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
from vocalis.settings import VOICES, LiveSettings

_RESTART_NOTE = " Restart Vocalis to use the new shortcuts."
_AUTOMATIC_LABEL = "Automatic (recommended)"
_AUTOMATIC_VALUE = "auto"
_SPEED_SLIDER_MINIMUM = 70
_SPEED_SLIDER_MAXIMUM = 160
_SPEED_SCALE = 100.0


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
        self._ptt_combo = _combo(PUSH_TO_TALK_CHOICES)
        self._quick_combo = _combo(CTRL_ALT_CHOICES)
        self._doc_combo = _combo(CTRL_ALT_CHOICES)
        self._ocr_combo = _combo(CTRL_ALT_CHOICES)
        self._voice_combo = _combo(VOICES)
        self._output_combo = QComboBox()
        self._test_sound_button = QPushButton("Test sound")
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setMinimum(_SPEED_SLIDER_MINIMUM)
        self._speed_slider.setMaximum(_SPEED_SLIDER_MAXIMUM)
        self._speed_label = QLabel("")
        self._preview_button = QPushButton("Preview voice (uses saved settings)")
        self._save_button = QPushButton("Save")
        self._close_button = QPushButton("Close")
        self._message_label = QLabel("")
        self._message_label.setWordWrap(True)
        self._build_layout()
        self._speed_slider.valueChanged.connect(self._on_speed_slider_changed)
        self._test_sound_button.clicked.connect(self._on_test_sound_clicked)
        self._preview_button.clicked.connect(self._on_preview_clicked)
        self._save_button.clicked.connect(self._on_save_clicked)
        self._close_button.clicked.connect(self.hide)
        self.load_from_settings()

    def _build_layout(self) -> None:
        """Place the widgets: checkboxes, form rows, message, buttons."""
        form = QFormLayout()
        form.addRow("Dictation (hold)", self._ptt_combo)
        form.addRow("Quick Reader  Ctrl+Alt+", self._quick_combo)
        form.addRow("Document Reader  Ctrl+Alt+", self._doc_combo)
        form.addRow("OCR Reader  Ctrl+Alt+", self._ocr_combo)
        form.addRow("Voice", self._voice_combo)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_combo)
        output_row.addWidget(self._test_sound_button)
        form.addRow("Sound output", output_row)
        speed_row = QHBoxLayout()
        speed_row.addWidget(self._speed_slider)
        speed_row.addWidget(self._speed_label)
        form.addRow("Speaking speed", speed_row)
        buttons = QHBoxLayout()
        buttons.addWidget(self._preview_button)
        buttons.addStretch(1)
        buttons.addWidget(self._save_button)
        buttons.addWidget(self._close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self._notifications_box)
        layout.addWidget(self._widget_box)
        layout.addLayout(form)
        layout.addWidget(self._message_label)
        layout.addLayout(buttons)

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
        self._fill_output_combo(config.audio_output_device)
        self._speed_slider.setValue(round(config.speaking_speed * _SPEED_SCALE))
        self._update_speed_label(self._speed_slider.value())

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
        engines = config.engines.model_copy(update={"qwen3_tts_speaker": self._voice_combo.currentText()})
        return config.model_copy(
            update={
                "notifications_enabled": self._notifications_box.isChecked(),
                "floating_widget_enabled": self._widget_box.isChecked(),
                "shortcuts": shortcuts,
                "engines": engines,
                "audio_output_device": self._selected_output_value(),
                "speaking_speed": self._speed_slider.value() / _SPEED_SCALE,
            }
        )

    def _on_save_clicked(self) -> None:
        """Validate and save; show the outcome in the message label."""
        try:
            self._settings.update(self._config_from_widgets())
        except (ShortcutError, ConfigError) as exc:
            self._message_label.setText(f"Not saved: {exc}")
            return
        message = "Saved."
        if self._settings.shortcuts_need_restart():
            message += _RESTART_NOTE
        self._message_label.setText(message)
        self._on_applied()
