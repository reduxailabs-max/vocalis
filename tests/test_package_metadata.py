"""Tests that the project's license metadata is present and consistent."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_license_file_is_mit_with_the_copyright_holder() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 reduxailabs-max" in text


def test_pyproject_declares_mit_and_ships_the_license_file() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]


def test_readme_mentions_the_license() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## License" in readme and "MIT" in readme
