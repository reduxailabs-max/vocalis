"""Tests for vocalis.config: pydantic models and TOML load/save."""
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from vocalis.config import (
    ConfigError,
    EnginesConfig,
    VocalisConfig,
    default_config_path,
    load_config,
    save_config,
)


def test_defaults_match_spec() -> None:
    config = VocalisConfig()
    assert config.shortcuts.numlock_key == "NUMLOCK"
    assert config.shortcuts.quick_reader == "Ctrl+Alt+R"
    assert config.shortcuts.document_reader == "Ctrl+Alt+D"
    assert config.shortcuts.ocr_reader == "Ctrl+Alt+O"
    assert config.engines.stt == "parakeet"
    assert config.notifications_enabled is True
    assert config.floating_widget_enabled is True
    assert config.model_idle_timeout_seconds == 30.0
    assert config.gpu_vram_ceiling_mb == 6144


def test_default_config_path_is_under_home_config() -> None:
    path = default_config_path()
    assert path == Path.home() / ".config" / "vocalis" / "config.toml"


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.toml"
    config = load_config(missing)
    assert config == VocalisConfig()


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = VocalisConfig(model_idle_timeout_seconds=45.0, notifications_enabled=False)
    save_config(original, path)
    loaded = load_config(path)
    assert loaded == original


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "config.toml"
    save_config(VocalisConfig(), path)
    assert path.exists()


def test_load_invalid_toml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("this is not [ valid toml")
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_invalid_field_type_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad_value.toml"
    path.write_text('model_idle_timeout_seconds = "not-a-number"\n')
    with pytest.raises(ConfigError):
        load_config(path)


def test_zero_idle_timeout_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        VocalisConfig(model_idle_timeout_seconds=0)


def test_negative_gpu_ceiling_is_rejected() -> None:
    with pytest.raises(Exception):
        VocalisConfig(gpu_vram_ceiling_mb=-1)


def test_load_non_utf8_file_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad_encoding.toml"
    path.write_bytes(b"\xff\xfe\x00invalid")
    with pytest.raises(ConfigError):
        load_config(path)


def test_default_toml_matches_model_defaults() -> None:
    project_root = Path(__file__).resolve().parent.parent
    default_toml_path = project_root / "config" / "default_config.toml"
    loaded = load_config(default_toml_path)
    assert loaded == VocalisConfig()


def test_parakeet_model_dir_defaults_under_home() -> None:
    config = VocalisConfig()
    assert config.engines.parakeet_model_dir.endswith(
        "/.local/share/voxtype/models/parakeet-tdt-0.6b-v3-int8"
    )
    assert str(Path.home()) in config.engines.parakeet_model_dir


def test_engines_config_has_qwen3_tts_and_piper_defaults() -> None:
    config = VocalisConfig()
    assert config.engines.qwen3_tts_speaker == "Ryan"
    assert "qwen3-tts-customvoice-0.6b" in config.engines.qwen3_tts_model_dir
    assert "piper" in config.engines.piper_binary_path
    assert "piper-voices" in config.engines.piper_voice_model_path


def test_save_config_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "vocalis" / "config.toml"
    save_config(VocalisConfig(notifications_enabled=False), path)
    assert load_config(path).notifications_enabled is False
    assert [p.name for p in path.parent.iterdir()] == ["config.toml"]


def test_failed_save_keeps_the_old_file_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    save_config(VocalisConfig(notifications_enabled=True), path)
    with patch("vocalis.config.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(ConfigError):
            save_config(VocalisConfig(notifications_enabled=False), path)
    assert load_config(path).notifications_enabled is True
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]


def test_save_config_cleans_up_and_reports_a_non_os_write_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    save_config(VocalisConfig(notifications_enabled=False), path)
    before = path.read_bytes()

    with patch("vocalis.config.tomli_w.dump", side_effect=TypeError("cannot encode")):
        with pytest.raises(ConfigError, match="cannot encode"):
            save_config(VocalisConfig(), path)

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]


def test_sound_and_pace_defaults() -> None:
    config = VocalisConfig()
    assert config.audio_output_device == "auto"
    assert config.speaking_speed == 1.15


@pytest.mark.parametrize("bad_value", ["0.5", "9.0", "-1", "nan"])
def test_out_of_range_speaking_speed_is_rejected_naming_the_field(tmp_path: Path, bad_value: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"speaking_speed = {bad_value}\n")
    with pytest.raises(ConfigError, match="speaking_speed"):
        load_config(path)


@pytest.mark.parametrize("edge_value", ["0.7", "1.6"])
def test_speaking_speed_limits_are_accepted(tmp_path: Path, edge_value: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"speaking_speed = {edge_value}\n")
    assert load_config(path).speaking_speed == float(edge_value)


@pytest.mark.parametrize("blank", ["", "   "])
def test_empty_audio_output_device_is_rejected(tmp_path: Path, blank: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'audio_output_device = "{blank}"\n')
    with pytest.raises(ConfigError, match="audio_output_device"):
        load_config(path)


def test_sound_and_pace_settings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    save_config(VocalisConfig(audio_output_device="hw:1,0", speaking_speed=1.4), path)
    loaded = load_config(path)
    assert loaded.audio_output_device == "hw:1,0"
    assert loaded.speaking_speed == 1.4


def test_engine_choice_defaults_to_auto_for_both_readers() -> None:
    """Both readers start on Automatic voice-engine choice."""
    engines = VocalisConfig().engines
    assert engines.quick_reader_engine == "auto"
    assert engines.document_reader_engine == "auto"


def test_engine_choice_rejects_an_unknown_engine_name() -> None:
    """A typo'd engine name is a validation error, not a silent fallback."""
    with pytest.raises(ValidationError):
        EnginesConfig(quick_reader_engine="banana")  # type: ignore[arg-type]


def test_old_config_with_the_dead_quick_tts_keys_loads_and_means_auto(tmp_path: Path) -> None:
    """Config files written by earlier versions still load; their ignored keys mean Auto."""
    path = tmp_path / "config.toml"
    path.write_text('[engines]\nquick_tts = "qwen3_tts"\ndoc_tts = "qwen3_tts"\n', encoding="utf-8")
    config = load_config(path)
    assert config.engines.quick_reader_engine == "auto"
    assert config.engines.document_reader_engine == "auto"


def test_kokoro_defaults_live_under_the_vocalis_models_folder() -> None:
    """Kokoro's model files default to ~/.local/share/vocalis/models/kokoro/."""
    engines = VocalisConfig().engines
    assert engines.kokoro_model_path.endswith("/.local/share/vocalis/models/kokoro/kokoro-v1.0.onnx")
    assert engines.kokoro_voices_path.endswith("/.local/share/vocalis/models/kokoro/voices-v1.0.bin")
    assert engines.kokoro_voice == "af_heart"


def test_default_speaker_is_english_native() -> None:
    """Ryan (English) is the configured default; this pins that choice."""
    assert EnginesConfig().qwen3_tts_speaker == "Ryan"


def test_a_non_english_native_speaker_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A non-English-native Qwen3-TTS speaker is logged, not rejected."""
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        EnginesConfig(qwen3_tts_speaker="Serena")
    assert any("Serena" in record.message for record in caplog.records)


def test_english_native_speakers_log_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Aiden is English-native, so constructing with it logs nothing."""
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        EnginesConfig(qwen3_tts_speaker="Aiden")
    assert not caplog.records


def test_speaker_check_is_case_insensitive(caplog: pytest.LogCaptureFixture) -> None:
    """The English-native check ignores letter case, so 'ryan' logs nothing."""
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        EnginesConfig(qwen3_tts_speaker="ryan")
    assert not caplog.records


def test_max_cpu_threads_and_max_ram_default_to_no_limit() -> None:
    config = VocalisConfig()
    assert config.max_cpu_threads == 0
    assert config.max_ram_mb == 0


def test_max_cpu_threads_and_max_ram_accept_a_positive_value() -> None:
    config = VocalisConfig(max_cpu_threads=4, max_ram_mb=4096)
    assert config.max_cpu_threads == 4
    assert config.max_ram_mb == 4096


def test_max_cpu_threads_rejects_a_negative_value() -> None:
    with pytest.raises(ValidationError):
        VocalisConfig(max_cpu_threads=-1)


def test_max_ram_mb_rejects_a_negative_value() -> None:
    with pytest.raises(ValidationError):
        VocalisConfig(max_ram_mb=-1)
