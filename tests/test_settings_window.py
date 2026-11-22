"""Tests for vocalis.ui.settings_window.SettingsWindow."""
from pathlib import Path
from unittest.mock import patch

from vocalis.config import ConfigError, VocalisConfig, load_config
from vocalis.settings import MODEL_PATH_FIELDS, LiveSettings
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


_TAB_NAMES = ["General", "Voice & Engine", "Performance", "Shortcuts", "Model files"]


def _tab_of(window: SettingsWindow, widget: object) -> str:
    """The name of the tab that holds a widget."""
    tabs = window._tabs  # noqa: SLF001
    for index in range(tabs.count()):
        if tabs.widget(index).isAncestorOf(widget):  # type: ignore[arg-type,union-attr]
            return tabs.tabText(index).replace("&&", "&")
    raise AssertionError("widget is on no tab")


def test_model_path_fields_are_pinned() -> None:
    assert set(MODEL_PATH_FIELDS) == {
        "parakeet_model_dir",
        "qwen3_tts_model_dir",
        "piper_binary_path",
        "piper_voice_model_path",
        "kokoro_model_path",
        "kokoro_voices_path",
    }


def test_settings_are_grouped_into_five_tabs(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, window, _applied, _path = make(qtbot, tmp_path)
    tabs = window._tabs  # noqa: SLF001
    assert [tabs.tabText(i).replace("&&", "&") for i in range(tabs.count())] == _TAB_NAMES


def test_each_control_is_on_its_tab(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, _path = make(qtbot, tmp_path)
    expected = {
        "General": [w._notifications_box, w._widget_box, w._output_combo, w._test_sound_button],  # noqa: SLF001
        "Voice & Engine": [w._voice_combo, w._quick_engine_combo, w._doc_engine_combo,  # noqa: SLF001
                           w._kokoro_voice_combo, w._speed_slider, w._preview_button],  # noqa: SLF001
        "Performance": [w._cpu_threads_spin, w._max_ram_spin, w._vram_ceiling_spin,  # noqa: SLF001
                        w._idle_timeout_spin],  # noqa: SLF001
        "Shortcuts": [w._ptt_combo, w._quick_combo, w._doc_combo, w._ocr_combo],  # noqa: SLF001
        "Model files": [*w._path_edits.values(), *w._browse_buttons.values()],  # noqa: SLF001
    }
    for tab, widgets in expected.items():
        for widget in widgets:
            assert _tab_of(w, widget) == tab


def test_save_close_and_message_are_outside_the_tabs(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, _path = make(qtbot, tmp_path)
    for widget in (w._save_button, w._close_button, w._message_label):  # noqa: SLF001
        assert not w._tabs.isAncestorOf(widget)  # noqa: SLF001
    assert w._save_button.property("role") == "primary"  # noqa: SLF001


def test_idle_timeout_is_shown_and_saved(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig(model_idle_timeout_seconds=45.0)
    settings, w, _applied, path = make(qtbot, tmp_path, config=config)
    assert w._idle_timeout_spin.value() == 45  # noqa: SLF001
    assert w._idle_timeout_spin.minimum() == 1 and w._idle_timeout_spin.maximum() == 3600  # noqa: SLF001
    w._idle_timeout_spin.setValue(120)  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert load_config(path).model_idle_timeout_seconds == 120.0
    assert w._message_label.text() == "Saved. Restart Vocalis to use the new model settings."  # noqa: SLF001


def test_model_locations_are_shown(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig()
    _settings, w, _applied, _path = make(qtbot, tmp_path, config=config)
    assert set(w._path_edits) == set(MODEL_PATH_FIELDS)  # noqa: SLF001
    for field in MODEL_PATH_FIELDS:
        assert w._path_edits[field].text() == getattr(config.engines, field)  # noqa: SLF001


def test_an_existing_new_model_file_is_saved_with_a_restart_note(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = tmp_path / "kokoro.onnx"
    model.write_bytes(b"fake model for test")
    _settings, w, applied, path = make(qtbot, tmp_path)
    w._path_edits["kokoro_model_path"].setText(str(model))  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert load_config(path).engines.kokoro_model_path == str(model)
    assert w._message_label.text() == "Saved. Restart Vocalis to use the new model settings."  # noqa: SLF001
    assert applied == [1]


def test_an_existing_new_model_folder_is_saved(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "parakeet"
    folder.mkdir()
    _settings, w, _applied, path = make(qtbot, tmp_path)
    w._path_edits["parakeet_model_dir"].setText(str(folder))  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert load_config(path).engines.parakeet_model_dir == str(folder)


def test_a_changed_model_location_that_does_not_exist_is_refused(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    missing = tmp_path / "nope.onnx"
    _settings, w, applied, path = make(qtbot, tmp_path)
    w._path_edits["kokoro_model_path"].setText(str(missing))  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert w._message_label.text() == f"Not saved: the Kokoro model file was not found at {missing}."  # noqa: SLF001
    assert not path.exists()
    assert applied == []


def test_a_file_where_a_folder_is_needed_is_refused(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    a_file = tmp_path / "not-a-folder.txt"
    a_file.write_text("x")
    _settings, w, _applied, path = make(qtbot, tmp_path)
    w._path_edits["qwen3_tts_model_dir"].setText(str(a_file))  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert w._message_label.text().startswith("Not saved: the Qwen3-TTS model folder was not found at")  # noqa: SLF001
    assert not path.exists()


def test_an_emptied_model_location_is_refused(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, applied, path = make(qtbot, tmp_path)
    w._path_edits["piper_binary_path"].setText("   ")  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert w._message_label.text() == "Not saved: the Piper program was not found at (empty)."  # noqa: SLF001
    assert not path.exists()
    assert applied == []


def test_unchanged_missing_model_locations_do_not_block_saving(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The default locations do not exist on a test machine; saving other settings still works."""
    _settings, w, _applied, path = make(qtbot, tmp_path)
    w._notifications_box.setChecked(False)  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert w._message_label.text() == "Saved."  # noqa: SLF001
    assert load_config(path).notifications_enabled is False


def test_browse_fills_a_file_location(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, _path = make(qtbot, tmp_path)
    with patch(
        "vocalis.ui.settings_window.QFileDialog.getOpenFileName", return_value=("/picked/voices.bin", "")
    ):
        w._browse_buttons["kokoro_voices_path"].click()  # noqa: SLF001
    assert w._path_edits["kokoro_voices_path"].text() == "/picked/voices.bin"  # noqa: SLF001


def test_browse_fills_a_folder_location(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, _path = make(qtbot, tmp_path)
    with patch("vocalis.ui.settings_window.QFileDialog.getExistingDirectory", return_value="/picked/dir"):
        w._browse_buttons["parakeet_model_dir"].click()  # noqa: SLF001
    assert w._path_edits["parakeet_model_dir"].text() == "/picked/dir"  # noqa: SLF001


def test_cancelling_browse_keeps_the_old_location(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, _path = make(qtbot, tmp_path)
    before = w._path_edits["piper_voice_model_path"].text()  # noqa: SLF001
    with patch("vocalis.ui.settings_window.QFileDialog.getOpenFileName", return_value=("", "")):
        w._browse_buttons["piper_voice_model_path"].click()  # noqa: SLF001
    assert w._path_edits["piper_voice_model_path"].text() == before  # noqa: SLF001


def test_a_home_folder_model_location_is_saved_expanded(qtbot, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "x").write_bytes(b"fake model for test")
    _settings, w, _applied, path = make(qtbot, tmp_path)
    w._path_edits["kokoro_model_path"].setText("~/x")  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert load_config(path).engines.kokoro_model_path == str(tmp_path / "x")


def test_a_padded_model_location_is_saved_trimmed(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = tmp_path / "kokoro.onnx"
    model.write_bytes(b"fake model for test")
    _settings, w, _applied, path = make(qtbot, tmp_path)
    w._path_edits["kokoro_model_path"].setText(f"  {model}  ")  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert load_config(path).engines.kokoro_model_path == str(model)


def test_dictation_output_choice_is_shown_and_saved(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, path = make(qtbot, tmp_path, config=VocalisConfig())
    assert w._dictation_output_combo.currentData() == "type"  # noqa: SLF001
    w._dictation_output_combo.setCurrentIndex(w._dictation_output_combo.findData("clipboard"))  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert load_config(path).dictation_output == "clipboard"


def test_saved_clipboard_only_choice_is_loaded(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, _path = make(qtbot, tmp_path, config=VocalisConfig(dictation_output="clipboard"))
    assert w._dictation_output_combo.currentData() == "clipboard"  # noqa: SLF001


def test_type_and_copy_choice_can_be_saved(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, path = make(qtbot, tmp_path, config=VocalisConfig())
    w._dictation_output_combo.setCurrentIndex(w._dictation_output_combo.findData("both"))  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert load_config(path).dictation_output == "both"


def test_speech_languages_choice_is_shown_and_saved(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _settings, w, _applied, path = make(qtbot, tmp_path, config=VocalisConfig())
    assert w._alphabet_combo.currentData() == "latin"  # noqa: SLF001
    w._alphabet_combo.setCurrentIndex(w._alphabet_combo.findData("all"))  # noqa: SLF001
    w._save_button.click()  # noqa: SLF001
    assert load_config(path).dictation_alphabet == "all"
