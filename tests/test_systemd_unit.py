"""Tests for the packaging/systemd/vocalis.service template (text content only;
never starts a real service)."""
from pathlib import Path

UNIT_PATH = Path(__file__).resolve().parent.parent / "packaging" / "systemd" / "vocalis.service"


def _lines() -> "list[str]":
    return UNIT_PATH.read_text().splitlines()


def test_unit_file_exists() -> None:
    assert UNIT_PATH.exists()


def test_restart_policy_matches_the_design() -> None:
    lines = _lines()
    assert "Restart=on-failure" in lines
    assert "RestartSec=3" in lines
    assert "StartLimitBurst=5" in lines
    assert "StartLimitIntervalSec=60" in lines


def test_wants_the_xdg_autostart_target_not_default_target() -> None:
    lines = _lines()
    assert any(line.startswith("WantedBy=") and "xdg-desktop-autostart.target" in line for line in lines)
    assert not any(line.startswith("WantedBy=default.target") for line in lines)


def test_waits_for_the_sound_system() -> None:
    text = UNIT_PATH.read_text()
    for unit in ("pipewire.service", "pipewire-pulse.service", "wireplumber.service"):
        assert unit in text, f"{unit} must appear in After=/Wants="
    assert any(line.startswith("After=") for line in _lines())
    assert any(line.startswith("Wants=") for line in _lines())


def test_execstart_uses_a_placeholder_not_a_real_path() -> None:
    text = UNIT_PATH.read_text()
    assert "__VOCALIS_EXEC__" in text
    assert "/home/" not in text  # never a real, user-specific path baked in


def test_readme_documents_the_placeholders() -> None:
    readme = UNIT_PATH.parent / "README.md"
    assert readme.exists()
    text = readme.read_text()
    assert "__VOCALIS_EXEC__" in text
