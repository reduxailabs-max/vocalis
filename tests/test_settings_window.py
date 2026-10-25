"""Tests for vocalis.ui.settings_window.SettingsWindow."""
from pathlib import Path

from vocalis.config import ConfigError, VocalisConfig, load_config
from vocalis.settings import LiveSettings
from vocalis.ui.settings_window import SettingsWindow


def make(  # type: ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
    config: VocalisConfig | None = None,
    saver=None,
    output_names=None,
    on_test_sound=None,
    on_preview_voice=None,
):
    """Build a window over a LiveSettings that saves into tmp_path."""
    path = tmp_path / "config.toml"
    if saver is None:
        settings = LiveSettings(config or VocalisConfig(), path)
    else:
        settings = LiveSettings(config or VocalisConfig(), path, saver=saver)
    applied: list[int] = []
    kwargs = {}
    if output_names is not None:
        kwargs["output_names"] = output_names
    if on_test_sound is not None:
        kwargs["on_test_sound"] = on_test_sound
    if on_preview_voice is not None:
        kwargs["on_preview_voice"] = on_preview_voice
    window = SettingsWindow(settings, on_applied=lambda: applied.append(1), **kwargs)
    qtbot.addWidget(window)
    return settings, window, applied, path


def test_widgets_show_the_current_settings(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    assert window.windowTitle() == "Vocalis — Settings"
    assert window._notifications_box.isChecked()  # noqa: SLF001
    assert window._widget_box.isChecked()  # noqa: SLF001
    assert window._ptt_combo.currentText() == "NUMLOCK"  # noqa: SLF001
    assert window._quick_combo.currentText() == "R"  # noqa: SLF001
    assert window._doc_combo.currentText() == "D"  # noqa: SLF001
    assert window._ocr_combo.currentText() == "O"  # noqa: SLF001
    assert window._voice_combo.currentText() == "Ryan"  # noqa: SLF001


def test_save_writes_the_file_applies_settings_and_calls_on_applied(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings, window, applied, path = make(qtbot, tmp_path)
    window._notifications_box.setChecked(False)  # noqa: SLF001
    window._voice_combo.setCurrentText("Dylan")  # noqa: SLF001
    window._save_button.click()  # noqa: SLF001
    saved = load_config(path)
    assert saved.notifications_enabled is False
    assert saved.engines.qwen3_tts_speaker == "Dylan"
    assert settings.voice() == "Dylan"
    assert settings.notifications_enabled() is False
    assert applied == [1]
    assert window._message_label.text() == "Saved."  # noqa: SLF001


def test_changing_a_shortcut_tells_the_user_to_restart(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings, window, _applied, path = make(qtbot, tmp_path)
    window._ocr_combo.setCurrentText("P")  # noqa: SLF001
    window._save_button.click()  # noqa: SLF001
    assert window._message_label.text().endswith("Restart Vocalis to use the new shortcuts.")  # noqa: SLF001
    assert settings.config.shortcuts.ocr_reader == "Ctrl+Alt+P"
    assert load_config(path).shortcuts.ocr_reader == "Ctrl+Alt+P"


def test_duplicate_shortcuts_are_refused_with_the_reason(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings, window, applied, path = make(qtbot, tmp_path)
    before = settings.config
    window._quick_combo.setCurrentText("O")  # noqa: SLF001
    window._save_button.click()  # noqa: SLF001
    assert window._message_label.text().startswith("Not saved:")  # noqa: SLF001
    assert not path.exists()
    assert settings.config == before
    assert applied == []


def test_save_failure_is_shown_and_nothing_changes(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    def failing_saver(_config: VocalisConfig, _path: Path) -> None:
        raise ConfigError("disk full")

    settings, window, applied, path = make(qtbot, tmp_path, saver=failing_saver)
    before = settings.config
    window._notifications_box.setChecked(False)  # noqa: SLF001
    window._save_button.click()  # noqa: SLF001
    assert "disk full" in window._message_label.text()  # noqa: SLF001
    assert window._message_label.text().startswith("Not saved:")  # noqa: SLF001
    assert settings.config == before
    assert not path.exists()
    assert applied == []


def test_unknown_stored_voice_is_still_selectable_and_kept(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig()
    config = config.model_copy(update={"engines": config.engines.model_copy(update={"qwen3_tts_speaker": "Custom"})})
    settings, window, _applied, path = make(qtbot, tmp_path, config=config)
    assert window._voice_combo.currentText() == "Custom"  # noqa: SLF001
    window._save_button.click()  # noqa: SLF001
    assert window._message_label.text() == "Saved."  # noqa: SLF001
    assert settings.voice() == "Custom"
    assert load_config(path).engines.qwen3_tts_speaker == "Custom"


def test_unreadable_stored_shortcut_falls_back_to_the_default_choice(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig()
    config = config.model_copy(
        update={"shortcuts": config.shortcuts.model_copy(update={"ocr_reader": "Ctrl+Alt+Banana", "numlock_key": "F13"})}
    )
    _settings, window, _applied, _path = make(qtbot, tmp_path, config=config)
    assert window._ocr_combo.currentText() == "O"  # noqa: SLF001
    assert window._ptt_combo.currentText() == "NUMLOCK"  # noqa: SLF001


def test_close_button_hides_without_quitting(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    window.show()
    assert window.isVisible()
    window._close_button.click()  # noqa: SLF001
    assert not window.isVisible()


def test_reloading_the_window_clears_the_old_message(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    window._save_button.click()  # noqa: SLF001
    assert window._message_label.text() == "Saved."  # noqa: SLF001

    window.load_from_settings()

    assert window._message_label.text() == ""  # noqa: SLF001


def test_sound_and_speed_defaults_are_shown(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, window, _applied, _path = make(qtbot, tmp_path, output_names=lambda: ["Speakers", "Headset"])
    assert window._output_combo.currentText() == "Automatic (recommended)"  # noqa: SLF001
    assert window._speed_slider.value() == 115  # noqa: SLF001
    assert window._speed_label.text() == "1.15x"  # noqa: SLF001


def test_moving_the_speed_slider_updates_the_label_live(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    window._speed_slider.setValue(130)  # noqa: SLF001
    assert window._speed_label.text() == "1.30x"  # noqa: SLF001


def test_saving_writes_output_and_speed_and_applies_live(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings, window, _applied, path = make(qtbot, tmp_path, output_names=lambda: ["Speakers", "Headset"])
    window._output_combo.setCurrentText("Headset")  # noqa: SLF001
    window._speed_slider.setValue(140)  # noqa: SLF001
    window._save_button.click()  # noqa: SLF001
    saved = load_config(path)
    assert saved.audio_output_device == "Headset"
    assert saved.speaking_speed == 1.40
    assert settings.audio_output_device() == "Headset"
    assert settings.speaking_speed() == 1.40


def test_resource_limit_defaults_are_shown(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    assert window._cpu_threads_spin.value() == 0  # noqa: SLF001
    assert window._max_ram_spin.value() == 0  # noqa: SLF001
    assert window._vram_ceiling_spin.value() == 6144  # noqa: SLF001


def test_saving_writes_the_resource_limits(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings, window, _applied, path = make(qtbot, tmp_path)
    window._cpu_threads_spin.setValue(4)  # noqa: SLF001
    window._max_ram_spin.setValue(2048)  # noqa: SLF001
    window._vram_ceiling_spin.setValue(3000)  # noqa: SLF001
    window._on_save_clicked()  # noqa: SLF001
    saved = load_config(path)
    assert saved.max_cpu_threads == 4
    assert saved.max_ram_mb == 2048
    assert saved.gpu_vram_ceiling_mb == 3000
    assert settings.max_cpu_threads() == 4
    assert settings.max_ram_mb() == 2048
    assert settings.gpu_vram_ceiling_mb() == 3000


def test_resource_limits_round_trip_through_reload(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig(max_cpu_threads=2, max_ram_mb=1024, gpu_vram_ceiling_mb=5000)
    _settings, window, _applied, _path = make(qtbot, tmp_path, config=config)
    assert window._cpu_threads_spin.value() == 2  # noqa: SLF001
    assert window._max_ram_spin.value() == 1024  # noqa: SLF001
    assert window._vram_ceiling_spin.value() == 5000  # noqa: SLF001


def test_saving_with_automatic_selected_stores_auto(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings, window, _applied, path = make(qtbot, tmp_path, output_names=lambda: ["Speakers"])
    window._save_button.click()  # noqa: SLF001
    assert load_config(path).audio_output_device == "auto"
    assert settings.audio_output_device() == "auto"


def test_unknown_stored_output_name_is_still_selectable_and_kept(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig().model_copy(update={"audio_output_device": "Old Device"})
    settings, window, _applied, path = make(qtbot, tmp_path, config=config, output_names=lambda: ["Speakers"])
    assert window._output_combo.currentText() == "Old Device"  # noqa: SLF001
    window._save_button.click()  # noqa: SLF001
    assert load_config(path).audio_output_device == "Old Device"
    assert settings.audio_output_device() == "Old Device"


def test_test_sound_button_passes_auto_and_shows_result(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def on_test_sound(output: str) -> str:
        calls.append(output)
        return "Sound played on Speakers."

    _settings, window, _applied, _path = make(qtbot, tmp_path, output_names=lambda: ["Speakers"], on_test_sound=on_test_sound)
    window._test_sound_button.click()  # noqa: SLF001
    assert calls == ["auto"]
    assert window._message_label.text() == "Sound played on Speakers."  # noqa: SLF001


def test_test_sound_button_passes_the_chosen_device_name(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def on_test_sound(output: str) -> str:
        calls.append(output)
        return "ok"

    _settings, window, _applied, _path = make(qtbot, tmp_path, output_names=lambda: ["Speakers", "Headset"], on_test_sound=on_test_sound)
    window._output_combo.setCurrentText("Headset")  # noqa: SLF001
    window._test_sound_button.click()  # noqa: SLF001
    assert calls == ["Headset"]


def test_test_sound_button_shows_failure_reason(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    def on_test_sound(_output: str) -> str:
        raise RuntimeError("no such device")

    _settings, window, _applied, _path = make(qtbot, tmp_path, on_test_sound=on_test_sound)
    window._test_sound_button.click()  # noqa: SLF001
    assert window._message_label.text() == "Sound test failed: no such device"  # noqa: SLF001


def test_preview_button_calls_the_callback_once(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    calls: list[int] = []
    _settings, window, _applied, _path = make(qtbot, tmp_path, on_preview_voice=lambda: calls.append(1))
    window._preview_button.click()  # noqa: SLF001
    assert calls == [1]


def test_preview_button_shows_failure_reason(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    def on_preview_voice() -> None:
        raise RuntimeError("engine busy")

    _settings, window, _applied, _path = make(qtbot, tmp_path, on_preview_voice=on_preview_voice)
    window._preview_button.click()  # noqa: SLF001
    assert window._message_label.text() == "Preview failed: engine busy"  # noqa: SLF001


def test_present_shows_and_reloads_saved_values_when_hidden(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings, window, _applied, _path = make(qtbot, tmp_path)
    window._voice_combo.setCurrentText("Aiden")  # noqa: SLF001 - unsaved edit while hidden
    window.present()
    assert window.isVisible()
    assert window._voice_combo.currentText() == settings.voice()  # noqa: SLF001


def test_present_keeps_unsaved_edits_when_already_open(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    window.present()
    window._voice_combo.setCurrentText("Aiden")  # noqa: SLF001
    window.present()
    assert window._voice_combo.currentText() == "Aiden"  # noqa: SLF001


def test_present_restores_a_minimized_window(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    window.present()
    window.showMinimized()
    window.present()
    assert not window.isMinimized()


def test_engine_pickers_default_to_automatic(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Both voice-engine drop-downs start on Automatic."""
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    assert window._quick_engine_combo.currentData() == "auto"  # noqa: SLF001
    assert window._doc_engine_combo.currentData() == "auto"  # noqa: SLF001
    assert window._kokoro_voice_combo.currentText() == "af_heart"  # noqa: SLF001


def test_engine_pickers_show_the_saved_choices(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Saved engine choices and Kokoro voice are shown when the window opens."""
    from vocalis.config import EnginesConfig

    config = VocalisConfig(
        engines=EnginesConfig(quick_reader_engine="kokoro", document_reader_engine="piper", kokoro_voice="bf_emma")
    )
    _settings, window, _applied, _path = make(qtbot, tmp_path, config=config)
    assert window._quick_engine_combo.currentData() == "kokoro"  # noqa: SLF001
    assert window._doc_engine_combo.currentData() == "piper"  # noqa: SLF001
    assert window._kokoro_voice_combo.currentText() == "bf_emma"  # noqa: SLF001


def test_saving_writes_the_chosen_engines(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Choosing engines and saving updates the live settings."""
    settings, window, _applied, _path = make(qtbot, tmp_path)
    window._quick_engine_combo.setCurrentIndex(window._quick_engine_combo.findData("qwen3_tts"))  # noqa: SLF001
    window._doc_engine_combo.setCurrentIndex(window._doc_engine_combo.findData("kokoro"))  # noqa: SLF001
    window._kokoro_voice_combo.setCurrentText("am_adam")  # noqa: SLF001
    window._on_save_clicked()  # noqa: SLF001
    assert settings.quick_reader_engine() == "qwen3_tts"
    assert settings.document_reader_engine() == "kokoro"
    assert settings.kokoro_voice() == "am_adam"


def test_an_unlisted_stored_kokoro_voice_stays_selectable(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A voice typed into the config file by hand is kept, not silently replaced."""
    from vocalis.config import EnginesConfig

    config = VocalisConfig(engines=EnginesConfig(kokoro_voice="af_river"))
    _settings, window, _applied, _path = make(qtbot, tmp_path, config=config)
    assert window._kokoro_voice_combo.currentText() == "af_river"  # noqa: SLF001
