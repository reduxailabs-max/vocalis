"""Tests for vocalis.settings.LiveSettings."""
import threading
from pathlib import Path

import pytest

from vocalis.config import ConfigError, ShortcutsConfig, VocalisConfig, load_config
from vocalis.hotkeys.shortcut_parser import ShortcutError
from vocalis.settings import VOICES, LiveSettings


def make(tmp_path: Path, config: VocalisConfig | None = None) -> LiveSettings:
    return LiveSettings(config or VocalisConfig(), tmp_path / "config.toml")


def test_voices_are_the_nine_presets_with_ryan_default() -> None:
    assert VOICES == ("Serena", "Vivian", "Uncle_Fu", "Ryan", "Aiden", "Ono_Anna", "Sohee", "Eric", "Dylan")
    assert make(Path("/nonexistent")).voice() == "Ryan"


def test_update_saves_the_file_then_changes_what_features_read(tmp_path: Path) -> None:
    settings = make(tmp_path)
    new = settings.config.model_copy(
        update={"notifications_enabled": False, "floating_widget_enabled": False,
                "engines": settings.config.engines.model_copy(update={"qwen3_tts_speaker": "Dylan"})}
    )
    settings.update(new)
    assert settings.notifications_enabled() is False
    assert settings.floating_widget_enabled() is False
    assert settings.voice() == "Dylan"
    assert load_config(tmp_path / "config.toml").engines.qwen3_tts_speaker == "Dylan"


def test_duplicate_shortcuts_are_refused_and_nothing_changes(tmp_path: Path) -> None:
    settings = make(tmp_path)
    bad = settings.config.model_copy(update={"shortcuts": ShortcutsConfig(quick_reader="Ctrl+Alt+O")})
    with pytest.raises(ShortcutError):
        settings.update(bad)
    assert settings.config.shortcuts.quick_reader == "Ctrl+Alt+R"
    assert not (tmp_path / "config.toml").exists()


def test_failed_save_keeps_the_old_settings(tmp_path: Path) -> None:
    def failing_saver(config: VocalisConfig, path: Path) -> None:
        raise ConfigError("disk full")

    settings = LiveSettings(VocalisConfig(), tmp_path / "config.toml", saver=failing_saver)
    with pytest.raises(ConfigError):
        settings.update(settings.config.model_copy(update={"notifications_enabled": False}))
    assert settings.notifications_enabled() is True


def test_shortcut_change_needs_a_restart_and_only_then(tmp_path: Path) -> None:
    settings = make(tmp_path)
    assert settings.shortcuts_need_restart() is False
    settings.update(settings.config.model_copy(update={"notifications_enabled": False}))
    assert settings.shortcuts_need_restart() is False
    settings.update(settings.config.model_copy(update={"shortcuts": ShortcutsConfig(ocr_reader="Ctrl+Alt+P")}))
    assert settings.shortcuts_need_restart() is True


def test_readers_on_other_threads_never_see_a_half_updated_config(tmp_path: Path) -> None:
    settings = make(tmp_path)
    seen: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            seen.append(settings.voice())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    for voice in ("Dylan", "Eric", "Ryan"):
        settings.update(settings.config.model_copy(update={"engines": settings.config.engines.model_copy(update={"qwen3_tts_speaker": voice})}))
    stop.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert set(seen) <= {"Ryan", "Dylan", "Eric"}


def test_equivalent_shortcut_spellings_do_not_need_a_restart(tmp_path: Path) -> None:
    started = VocalisConfig(shortcuts=ShortcutsConfig(numlock_key="numlock", quick_reader="ctrl+alt+r"))
    settings = make(tmp_path, started)
    settings.update(started.model_copy(update={"shortcuts": ShortcutsConfig()}))
    assert settings.shortcuts_need_restart() is False


def test_a_different_key_needs_a_restart_even_with_other_spellings(tmp_path: Path) -> None:
    started = VocalisConfig(shortcuts=ShortcutsConfig(quick_reader="ctrl+alt+r"))
    settings = make(tmp_path, started)
    settings.update(started.model_copy(update={"shortcuts": ShortcutsConfig(quick_reader="Ctrl+Alt+T")}))
    assert settings.shortcuts_need_restart() is True


def test_started_with_a_bad_value_then_saving_the_defaults_needs_no_restart(tmp_path: Path) -> None:
    started = VocalisConfig(shortcuts=ShortcutsConfig(quick_reader="Ctrl+Alt+nonsense"))
    settings = make(tmp_path, started)
    settings.update(started.model_copy(update={"shortcuts": ShortcutsConfig()}))
    assert settings.shortcuts_need_restart() is False


def test_current_shortcuts_that_fail_to_parse_need_a_restart(tmp_path: Path) -> None:
    settings = make(tmp_path)
    settings._config = settings.config.model_copy(  # noqa: SLF001
        update={"shortcuts": ShortcutsConfig(quick_reader="Ctrl+Alt+nonsense")}
    )
    assert settings.shortcuts_need_restart() is True


def test_sound_output_and_speed_accessors_follow_updates(tmp_path: Path) -> None:
    settings = make(tmp_path)
    assert settings.audio_output_device() == "auto"
    assert settings.speaking_speed() == 1.15
    settings.update(settings.config.model_copy(update={"audio_output_device": "hw:2,0", "speaking_speed": 1.5}))
    assert settings.audio_output_device() == "hw:2,0"
    assert settings.speaking_speed() == 1.5
