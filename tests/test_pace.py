"""Tests for vocalis.audio.pace (real librosa, tiny synthetic audio, no models)."""
import logging

import numpy as np
import numpy.typing as npt
import pytest

from vocalis.audio.pace import MAX_SPEED, MIN_SPEED, apply_speed, clamp_speed, paced_audio

RATE = 24000


def sine(freq: float = 440.0, seconds: float = 1.0) -> npt.NDArray[np.float32]:
    """Make a quiet sine wave of the given pitch and length."""
    t = np.arange(int(RATE * seconds)) / RATE
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_frequency(audio: npt.NDArray[np.float32]) -> float:
    """Return the strongest frequency (Hz) found in the audio."""
    spectrum = np.abs(np.fft.rfft(audio))
    return float(np.argmax(spectrum) * RATE / len(audio))


def test_clamp_keeps_values_inside_the_limits() -> None:
    """Out-of-range speeds are limited and NaN becomes 1.0."""
    assert clamp_speed(0.1) == MIN_SPEED
    assert clamp_speed(9.0) == MAX_SPEED
    assert clamp_speed(1.2) == 1.2
    assert clamp_speed(float("nan")) == 1.0


def test_speed_one_returns_the_same_audio_object() -> None:
    """Neutral speed hands back the very same array."""
    audio = sine()
    assert apply_speed(audio, RATE, 1.0) is audio


def test_faster_speed_shortens_and_slower_lengthens_proportionally() -> None:
    """Length changes in proportion to the speed and stays float32."""
    audio = sine(seconds=2.0)
    fast = apply_speed(audio, RATE, 1.5)
    slow = apply_speed(audio, RATE, 0.8)
    assert abs(len(fast) - len(audio) / 1.5) < 0.03 * len(audio)
    assert abs(len(slow) - len(audio) / 0.8) < 0.03 * len(audio)
    assert fast.dtype == np.float32 and slow.dtype == np.float32


def test_pitch_is_preserved() -> None:
    """A 440 Hz tone is still about 440 Hz after speeding up."""
    fast = apply_speed(sine(440.0, 2.0), RATE, 1.4)
    assert abs(dominant_frequency(fast) - 440.0) < 12.0


def test_silence_stays_silent() -> None:
    """Silence in gives silence out."""
    out = apply_speed(np.zeros(RATE, dtype=np.float32), RATE, 1.3)
    assert float(np.max(np.abs(out))) < 1e-6


def test_empty_and_tiny_inputs_are_returned_unchanged() -> None:
    """Empty or very short audio is returned as the same object."""
    empty = np.zeros(0, dtype=np.float32)
    tiny = np.zeros(100, dtype=np.float32)
    assert apply_speed(empty, RATE, 1.3) is empty
    assert apply_speed(tiny, RATE, 1.3) is tiny


def test_out_of_range_speed_is_clamped_not_rejected() -> None:
    """A too-large speed behaves like the maximum speed."""
    audio = sine(seconds=2.0)
    assert abs(len(apply_speed(audio, RATE, 9.0)) - len(audio) / MAX_SPEED) < 0.03 * len(audio)


def test_paced_audio_with_no_provider_returns_the_same_object() -> None:
    """No provider means no pacing at all -- the identical array comes back."""
    audio = sine(seconds=2.0)
    assert paced_audio(audio, RATE, None) is audio


def test_paced_audio_applies_the_providers_speed() -> None:
    """A provider returning 1.5 shortens the audio by about 1/1.5."""
    audio = sine(seconds=2.0)
    result = paced_audio(audio, RATE, lambda: 1.5)
    assert abs(len(result) - len(audio) / 1.5) < 0.03 * len(audio)


def test_paced_audio_survives_a_raising_provider(caplog: pytest.LogCaptureFixture) -> None:
    """A provider that raises must not break the reading: the original
    audio comes back unchanged, and a warning is logged."""
    audio = sine(seconds=2.0)

    def exploding_provider() -> float:
        raise RuntimeError("settings unavailable")

    with caplog.at_level(logging.WARNING, logger="vocalis"):
        result = paced_audio(audio, RATE, exploding_provider)

    assert result is audio
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_paced_audio_survives_a_raising_apply_speed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If apply_speed() itself raises (not just the provider), the
    original audio still comes back unchanged and a warning is logged."""
    import vocalis.audio.pace as pace_module

    audio = sine(seconds=2.0)

    def exploding_apply_speed(
        audio: npt.NDArray[np.float32], sample_rate: int, speed: float
    ) -> npt.NDArray[np.float32]:
        raise RuntimeError("librosa exploded")

    monkeypatch.setattr(pace_module, "apply_speed", exploding_apply_speed)

    with caplog.at_level(logging.WARNING, logger="vocalis"):
        result = paced_audio(audio, RATE, lambda: 1.5)

    assert result is audio
    assert any(record.levelno == logging.WARNING for record in caplog.records)
