"""Typed, validated configuration for vocalis, backed by a TOML file.

All config values that affect behavior (shortcuts, engine choices,
notification/widget toggles, GPU idle policy) live in one pydantic model so
invalid config is rejected loudly at load time instead of causing confusing
failures deep inside a feature.
"""
import logging
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Annotated, Literal

import tomli_w
from pydantic import BaseModel, Field, StringConstraints, ValidationError, model_validator

from vocalis.audio.pace import MAX_SPEED, MIN_SPEED

logger = logging.getLogger("vocalis")

ENGLISH_NATIVE_QWEN_SPEAKERS = frozenset({"ryan", "aiden"})


class ConfigError(Exception):
    """Raised when the config file cannot be read, parsed, or validated."""


class ShortcutsConfig(BaseModel):
    """Keyboard shortcut bindings for each vocalis feature."""

    numlock_key: str = "NUMLOCK"
    quick_reader: str = "Ctrl+Alt+R"
    document_reader: str = "Ctrl+Alt+D"
    ocr_reader: str = "Ctrl+Alt+O"


def _default_parakeet_model_dir() -> str:
    """Default location of the Parakeet model files.

    Points at the existing Voxtype installation's model directory — Phase 2
    reuses those exact files rather than re-downloading them. A future
    packaging phase will add a proper first-run download for machines that
    don't already have a Voxtype install.
    """
    return str(Path.home() / ".local" / "share" / "voxtype" / "models" / "parakeet-tdt-0.6b-v3-int8")


def _default_qwen3_tts_model_dir() -> str:
    """Default location of the downloaded Qwen3-TTS CustomVoice model
    files (see Phase 3's Task 2 one-time download step)."""
    return str(Path.home() / ".local" / "share" / "vocalis" / "models" / "qwen3-tts-customvoice-0.6b")


def _default_piper_binary_path() -> str:
    """Default location of the existing Piper binary (already installed
    from the legacy Voxtype setup)."""
    return str(Path.home() / ".local" / "bin" / "piper" / "piper")


def _default_piper_voice_model_path() -> str:
    """Default location of the existing Piper voice model (already
    installed from the legacy Voxtype setup)."""
    return str(Path.home() / ".local" / "share" / "piper-voices" / "en_US-lessac-medium.onnx")


EngineChoice = Literal["auto", "qwen3_tts", "kokoro", "piper"]


def _default_kokoro_model_path() -> str:
    """Default location of the Kokoro voice model file."""
    return str(Path.home() / ".local" / "share" / "vocalis" / "models" / "kokoro" / "kokoro-v1.0.onnx")


def _default_kokoro_voices_path() -> str:
    """Default location of Kokoro's voices file."""
    return str(Path.home() / ".local" / "share" / "vocalis" / "models" / "kokoro" / "voices-v1.0.bin")


class EnginesConfig(BaseModel):
    """Which engine implementation each feature should use."""

    stt: str = "parakeet"
    quick_reader_engine: EngineChoice = "auto"
    document_reader_engine: EngineChoice = "auto"
    parakeet_model_dir: str = Field(default_factory=_default_parakeet_model_dir)
    qwen3_tts_model_dir: str = Field(default_factory=_default_qwen3_tts_model_dir)
    qwen3_tts_speaker: str = "Ryan"
    piper_binary_path: str = Field(default_factory=_default_piper_binary_path)
    piper_voice_model_path: str = Field(default_factory=_default_piper_voice_model_path)
    kokoro_model_path: str = Field(default_factory=_default_kokoro_model_path)
    kokoro_voices_path: str = Field(default_factory=_default_kokoro_voices_path)
    kokoro_voice: str = "af_heart"

    @model_validator(mode="after")
    def _warn_if_speaker_is_not_english_native(self) -> "EnginesConfig":
        """Log a warning when the Qwen3-TTS speaker is not English-native.

        Qwen3-TTS always speaks as English (its spoken language is not
        yet separately configurable), but most of its nine preset voices
        are native to Chinese, Japanese or Korean and sound noticeably
        worse when forced to speak English. This never blocks the
        config -- it only logs, so a deliberate choice still works.
        """
        if self.qwen3_tts_speaker.strip().lower() not in ENGLISH_NATIVE_QWEN_SPEAKERS:
            logger.warning(
                "Qwen3-TTS speaker %r is not English-native (English-native choices: Ryan, Aiden); "
                "it will still be asked to speak English, which may sound worse than its own language.",
                self.qwen3_tts_speaker,
            )
        return self


class VocalisConfig(BaseModel):
    """Top-level vocalis configuration.

    audio_output_device is "auto" or the name of a sound output; blank is
    rejected (validated after trimming spaces).
    """

    shortcuts: ShortcutsConfig = ShortcutsConfig()
    engines: EnginesConfig = EnginesConfig()
    notifications_enabled: bool = True
    floating_widget_enabled: bool = True
    audio_output_device: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = "auto"
    speaking_speed: float = Field(default=1.15, ge=MIN_SPEED, le=MAX_SPEED)
    model_idle_timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
    gpu_vram_ceiling_mb: int = Field(default=6144, gt=0)


def default_config_path() -> Path:
    """Return the standard per-user config file location.

    Returns:
        `~/.config/vocalis/config.toml`.
    """
    return Path.home() / ".config" / "vocalis" / "config.toml"


def load_config(path: Path) -> VocalisConfig:
    """Load and validate config from a TOML file.

    If the file does not exist, returns default settings rather than
    raising — a missing config file is a normal first-run state, not an
    error.

    Args:
        path: Location of the TOML config file.

    Returns:
        A validated VocalisConfig.

    Raises:
        ConfigError: If the file exists but is not valid TOML, or its
            contents fail pydantic validation.
    """
    if not path.exists():
        return VocalisConfig()

    try:
        raw_text = path.read_bytes()
        data = tomllib.loads(raw_text.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in config file {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc

    try:
        return VocalisConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config values in {path}: {exc}") from exc


def save_config(config: VocalisConfig, path: Path) -> None:
    """Write config to a TOML file, creating parent directories as needed.

    Args:
        config: The validated config to persist.
        path: Destination file path.

    The file is written to a temporary file in the same folder and renamed
    into place, so a crash while saving cannot leave a half-written config.

    Raises:
        ConfigError: If the file cannot be written for any reason
            (e.g. permission denied, or a value that cannot be encoded).
    """
    tmp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "wb") as handle:
            tomli_w.dump(config.model_dump(mode="json"), handle)
        os.replace(tmp_path, path)
    except Exception as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise ConfigError(f"Could not write config file {path}: {exc}") from exc
