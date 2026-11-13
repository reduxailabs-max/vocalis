"""Tests for the optional Qwen3-TTS add-on checks (no real packages are looked at)."""
import importlib.metadata
from pathlib import Path

import pytest

from vocalis import qwen3_addon
from vocalis.config import VocalisConfig
from vocalis.qwen3_addon import (
    normalize,
    qwen3_available,
    qwen3_model_present,
    qwen3_packages_installed,
    qwen3_requirements,
    read_name_list,
    removable_packages,
    runtime_installer_dir,
    write_snapshot,
)


def _requires(lines: list[str] | None):  # type: ignore[no-untyped-def]
    return lambda name: lines


def _specs(present: set[str]):  # type: ignore[no-untyped-def]
    return lambda name: object() if name in present else None


def _config(model_dir: Path) -> VocalisConfig:
    config = VocalisConfig()
    config.engines.qwen3_tts_model_dir = str(model_dir)
    return config


def test_requirements_are_read_from_the_qwen3_extra_only() -> None:
    lines = [
        "pydantic>=2.6,<3",
        'qwen-tts==0.1.1; extra == "qwen3"',
        "torchaudio>=2.0,<3 ; extra == 'qwen3'",
        'pytest>=8,<9; extra == "dev"',
    ]
    assert qwen3_requirements(_requires(lines)) == ["qwen-tts==0.1.1", "torchaudio>=2.0,<3"]


def test_requirements_empty_when_the_package_data_is_missing() -> None:
    assert qwen3_requirements(_requires(None)) == []

    def missing(name: str) -> list[str]:
        raise importlib.metadata.PackageNotFoundError(name)

    assert qwen3_requirements(missing) == []


def test_real_package_data_lists_the_qwen3_extra_when_installed() -> None:
    """The project's own pyproject names the extra (read from the source tree)."""
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert "qwen3 = [" in text
    assert text.index("qwen-tts") > text.index("qwen3 = [")


def test_packages_installed_needs_both_qwen_tts_and_torch() -> None:
    assert qwen3_packages_installed(_specs({"qwen_tts", "torch"})) is True
    assert qwen3_packages_installed(_specs({"torch"})) is False
    assert qwen3_packages_installed(_specs(set())) is False


def test_packages_installed_is_false_when_the_lookup_breaks() -> None:
    def broken(name: str) -> object:
        raise ValueError(name)

    assert qwen3_packages_installed(broken) is False


def test_model_present_needs_every_file(tmp_path: Path) -> None:
    assert qwen3_model_present(tmp_path) is False
    (tmp_path / "config.json").write_text("{}")
    assert qwen3_model_present(tmp_path) is False
    (tmp_path / "model.safetensors").write_text("x")
    assert qwen3_model_present(tmp_path) is True


def test_available_needs_packages_and_model(tmp_path: Path) -> None:
    both = _specs({"qwen_tts", "torch"})
    assert qwen3_available(_config(tmp_path), both) is False
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_text("x")
    assert qwen3_available(_config(tmp_path), both) is True
    assert qwen3_available(_config(tmp_path), _specs(set())) is False


def test_normalize_ignores_case_and_separators() -> None:
    assert normalize("Qwen_TTS") == normalize("qwen-tts") == "qwen-tts"
    assert normalize("zope.interface") == "zope-interface"


def test_snapshot_refreshes_an_older_list_while_qwen3_is_absent(tmp_path: Path) -> None:
    target = tmp_path / "state" / "baseline.txt"
    assert write_snapshot(target, lambda: {"numpy", "pydantic"}, _specs(set())) is True
    assert read_name_list(target) == {"numpy", "pydantic"}
    assert write_snapshot(target, lambda: {"numpy", "other"}, _specs(set())) is True
    assert read_name_list(target) == {"numpy", "other"}


def test_snapshot_keeps_the_old_list_once_qwen3_is_installed(tmp_path: Path) -> None:
    target = tmp_path / "baseline.txt"
    target.write_text("numpy\n")
    assert write_snapshot(target, lambda: {"numpy", "torch"}, _specs({"qwen_tts", "torch"})) is False
    assert read_name_list(target) == {"numpy"}


def test_removable_never_includes_the_speech_runtime(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("numpy\n")
    now = {"numpy", "torch", "onnxruntime", "onnxruntime-gpu"}
    assert removable_packages(baseline, lambda: now) == ["torch"]


def test_snapshot_is_skipped_when_qwen3_is_already_installed(tmp_path: Path) -> None:
    target = tmp_path / "baseline.txt"
    assert write_snapshot(target, lambda: {"numpy"}, _specs({"qwen_tts", "torch"})) is False
    assert not target.exists()


def test_removable_is_what_was_added_after_the_snapshot(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("numpy\npydantic\n")
    now = {"numpy", "pydantic", "torch", "transformers", "vocalis", "pip"}
    assert removable_packages(baseline, lambda: now) == ["torch", "transformers"]


def test_removable_is_empty_without_a_snapshot(tmp_path: Path) -> None:
    assert removable_packages(tmp_path / "missing.txt", lambda: {"torch"}) == []


def test_read_name_list_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_name_list(tmp_path / "nope") == set()


def test_installed_package_names_lists_this_environment() -> None:
    names = qwen3_addon.installed_package_names()
    assert "pydantic" in names and all(name == normalize(name) for name in names)


def test_runtime_installer_dir_is_under_the_vocalis_data_folder() -> None:
    assert runtime_installer_dir().parts[-3:] == ("share", "vocalis", "installer")


def test_main_requirements_prints_one_per_line(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_addon, "qwen3_requirements", lambda: ["a==1", "b>=2"])
    assert qwen3_addon.main(["requirements"]) == 0
    assert capsys.readouterr().out.splitlines() == ["a==1", "b>=2"]


def test_main_requirements_fails_when_the_list_cannot_be_read(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qwen3_addon, "qwen3_requirements", lambda: [])
    assert qwen3_addon.main(["requirements"]) == 1
    assert "Could not read" in capsys.readouterr().err


def test_main_snapshot_and_removable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "baseline.txt"
    monkeypatch.setattr(qwen3_addon, "installed_package_names", lambda: {"numpy"})
    monkeypatch.setattr(qwen3_addon, "qwen3_packages_installed", lambda *a, **k: False)
    assert qwen3_addon.main(["snapshot", str(target)]) == 0
    assert read_name_list(target) == {"numpy"}
    monkeypatch.setattr(qwen3_addon, "installed_package_names", lambda: {"numpy", "torch"})
    assert qwen3_addon.main(["removable", str(target)]) == 0
    assert capsys.readouterr().out.splitlines() == ["torch"]
