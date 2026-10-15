"""Tests for vocalis.startup_checks — each check is independent and non-fatal."""
import logging
from pathlib import Path

from vocalis.config import VocalisConfig
from vocalis.startup_checks import (
    check_capture_tools,
    check_parakeet_model,
    check_qwen3_tts_model,
    check_stale_asoundrc,
    check_ydotoold,
    run_startup_checks,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_parakeet_model_present_is_silent(tmp_path: Path) -> None:
    model_dir = tmp_path / "parakeet"
    for name in ("config.json", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "vocab.txt"):
        _touch(model_dir / name)
    assert check_parakeet_model(model_dir) is None


def test_parakeet_model_missing_names_the_folder(tmp_path: Path) -> None:
    model_dir = tmp_path / "does-not-exist"
    message = check_parakeet_model(model_dir)
    assert message is not None and "Parakeet" in message and str(model_dir) in message


def test_parakeet_model_partially_present_is_also_reported(tmp_path: Path) -> None:
    model_dir = tmp_path / "parakeet"
    _touch(model_dir / "config.json")  # only one of the four files
    message = check_parakeet_model(model_dir)
    assert message is not None and "Parakeet" in message


def test_qwen3_tts_model_present_is_silent(tmp_path: Path) -> None:
    model_dir = tmp_path / "qwen"
    _touch(model_dir / "config.json")
    _touch(model_dir / "model.safetensors")
    assert check_qwen3_tts_model(model_dir) is None


def test_qwen3_tts_model_missing_names_the_folder(tmp_path: Path) -> None:
    model_dir = tmp_path / "does-not-exist"
    message = check_qwen3_tts_model(model_dir)
    assert message is not None and "voice" in message.lower() and str(model_dir) in message


def test_ydotoold_socket_present_is_silent(tmp_path: Path) -> None:
    socket_path = tmp_path / ".ydotool_socket"
    _touch(socket_path)
    assert check_ydotoold(socket_path) is None


def test_ydotoold_socket_missing_mentions_typing() -> None:
    message = check_ydotoold(Path("/nonexistent/.ydotool_socket"))
    assert message is not None and "typ" in message.lower()


def test_capture_tools_all_present_is_silent() -> None:
    assert check_capture_tools(which=lambda name: f"/usr/bin/{name}") is None


def test_capture_tools_all_missing_is_reported() -> None:
    message = check_capture_tools(which=lambda name: None)
    assert message == "The OCR Reader needs tesseract and xclip, which are not installed."


def test_capture_tools_one_missing_reads_grammatically() -> None:
    message = check_capture_tools(
        which=lambda name: None if name == "xclip" else "/usr/bin/tesseract"
    )
    assert message == "The OCR Reader needs xclip, which is not installed."


def test_capture_tools_partially_missing_names_only_the_missing_ones() -> None:
    message = check_capture_tools(which=lambda name: "/usr/bin/tesseract" if name == "tesseract" else None)
    assert message is not None and "xclip" in message and "tesseract" not in message


def test_run_startup_checks_returns_every_real_problem_in_a_fixed_order(tmp_path: Path) -> None:
    config = VocalisConfig()
    config.engines.parakeet_model_dir = str(tmp_path / "missing-parakeet")
    config.engines.qwen3_tts_model_dir = str(tmp_path / "missing-qwen")
    messages = run_startup_checks(config)
    assert len(messages) >= 2
    assert "Parakeet" in messages[0]


def test_run_startup_checks_is_silent_when_everything_is_fine(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HOME", str(tmp_path))
    config = VocalisConfig()
    parakeet_dir = tmp_path / "parakeet"
    for name in ("config.json", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "vocab.txt"):
        _touch(parakeet_dir / name)
    config.engines.parakeet_model_dir = str(parakeet_dir)
    qwen_dir = tmp_path / "qwen"
    _touch(qwen_dir / "config.json")
    _touch(qwen_dir / "model.safetensors")
    config.engines.qwen3_tts_model_dir = str(qwen_dir)
    socket_path = tmp_path / ".ydotool_socket"
    _touch(socket_path)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr("vocalis.startup_checks.shutil.which", lambda _name: "/usr/bin/x")
    assert run_startup_checks(config) == []


def test_run_startup_checks_survives_a_check_that_raises(tmp_path: Path, monkeypatch, caplog) -> None:  # type: ignore[no-untyped-def]
    config = VocalisConfig()

    def _boom(_model_dir: Path) -> str | None:
        raise RuntimeError("boom")

    monkeypatch.setattr("vocalis.startup_checks.check_parakeet_model", _boom)
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        messages = run_startup_checks(config)
    assert "boom" in caplog.text
    assert all("Parakeet" not in m for m in messages)


def test_no_asoundrc_is_silent(tmp_path: Path) -> None:
    assert check_stale_asoundrc(tmp_path / ".asoundrc") is None


def test_unrelated_asoundrc_is_silent(tmp_path: Path) -> None:
    path = tmp_path / ".asoundrc"
    path.write_text("pcm.!default { type pulse }\n")
    assert check_stale_asoundrc(path) is None


def test_old_voxtype_asoundrc_is_reported_in_plain_words(tmp_path: Path) -> None:
    path = tmp_path / ".asoundrc"
    path.write_text("# ALSA configuration for VOXTYPE\npcm.voxtype { type plug }\n")
    message = check_stale_asoundrc(path)
    assert message is not None
    assert "Voxtype" in message and ".asoundrc.old" in message and str(path) in message


def test_unreadable_asoundrc_is_silent_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / ".asoundrc"
    path.mkdir()  # a directory: read_text raises IsADirectoryError
    assert check_stale_asoundrc(path) is None


def test_binary_asoundrc_is_silent(tmp_path: Path) -> None:
    path = tmp_path / ".asoundrc"
    path.write_bytes(b"\xff\xfe\x00voxtype-not-really")
    assert check_stale_asoundrc(path) is None or "Voxtype" in (check_stale_asoundrc(path) or "")


def test_run_startup_checks_reports_a_stale_asoundrc_last(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".asoundrc").write_text("pcm.voxtype { }\n")
    messages = run_startup_checks(VocalisConfig())
    assert "Voxtype" in messages[-1]
