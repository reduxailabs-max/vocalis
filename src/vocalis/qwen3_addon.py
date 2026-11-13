"""Qwen3-TTS is an optional add-on: this module answers "is it installed?".

Qwen3-TTS needs two things on this PC: its Python helper programs (PyTorch,
transformers ...) and its 2.4 GB voice model. The installer and the Settings
window use these functions to add or remove both, and the voice picker uses
them to hide Qwen3 when it is not there.

The shell installer calls this file as a command::

    python -m vocalis.qwen3_addon requirements
    python -m vocalis.qwen3_addon snapshot FILE
    python -m vocalis.qwen3_addon removable FILE
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from vocalis.config import VocalisConfig
from vocalis.startup_checks import QWEN3_TTS_MODEL_FILES

QWEN3_EXTRA = "qwen3"
QWEN3_IMPORT_NAMES = ("qwen_tts", "torch")
INSTALLER_FOLDER = Path.home() / ".local" / "share" / "vocalis" / "installer"
BASELINE_FILE = Path.home() / ".local" / "share" / "vocalis" / "qwen3-baseline.txt"
_KEPT_ALWAYS = frozenset({"vocalis", "pip", "setuptools", "wheel", "uv"})
_KEPT_PREFIXES = ("onnxruntime",)  # the speech runtime; the installer re-checks which build is needed
_EXTRA_MARKER = re.compile(r"""extra\s*==\s*['"]([^'"]+)['"]""")
_NAME_PATTERN = re.compile(r"[-_.]+")


def runtime_installer_dir() -> Path:
    """Where the installer scripts are kept so Settings can run them later."""
    return INSTALLER_FOLDER


def normalize(name: str) -> str:
    """The canonical form of a package name (case and - _ . do not matter)."""
    return _NAME_PATTERN.sub("-", name).strip().lower()


def qwen3_requirements(
    requires: Callable[[str], list[str] | None] = importlib.metadata.requires,
) -> list[str]:
    """The packages the optional ``qwen3`` extra needs, read from Vocalis's own metadata."""
    try:
        lines = requires("vocalis") or []
    except importlib.metadata.PackageNotFoundError:
        return []
    found: list[str] = []
    for line in lines:
        requirement, _, marker = line.partition(";")
        match = _EXTRA_MARKER.search(marker)
        if match is not None and match.group(1) == QWEN3_EXTRA:
            found.append(requirement.strip())
    return found


def qwen3_packages_installed(find_spec: Callable[[str], object | None] = importlib.util.find_spec) -> bool:
    """Whether Qwen3's helper programs can be imported right now."""
    importlib.invalidate_caches()
    try:
        return all(find_spec(name) is not None for name in QWEN3_IMPORT_NAMES)
    except (ImportError, ValueError):
        return False


def qwen3_model_present(model_dir: Path) -> bool:
    """Whether the Qwen3 voice model files are all in the folder."""
    return all((model_dir / name).is_file() for name in QWEN3_TTS_MODEL_FILES)


def qwen3_available(
    config: VocalisConfig,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> bool:
    """Whether Qwen3-TTS can be used: helper programs AND voice model both present."""
    return qwen3_packages_installed(find_spec) and qwen3_model_present(Path(config.engines.qwen3_tts_model_dir))


def installed_package_names() -> set[str]:
    """Normalized names of every package installed in this Python environment."""
    names: set[str] = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            names.add(normalize(name))
    return names


def read_name_list(path: Path) -> set[str]:
    """Normalized names from a one-per-line file (a missing file is an empty set)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    return {normalize(line) for line in text.splitlines() if line.strip()}


def write_snapshot(
    path: Path,
    installed: Callable[[], set[str]] = installed_package_names,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> bool:
    """Remember which packages exist WITHOUT Qwen3, so removing it later is exact.

    Refreshes an older list (a Vocalis upgrade may add packages), but writes
    nothing when Qwen3's helper programs are already installed (the list
    would then be wrong). Returns whether a file was written.
    """
    if qwen3_packages_installed(find_spec):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(installed())) + "\n", encoding="utf-8")
    return True


def removable_packages(
    baseline: Path,
    installed: Callable[[], set[str]] = installed_package_names,
) -> list[str]:
    """Packages added after the snapshot, i.e. Qwen3's helper programs.

    Empty when there is no snapshot: guessing could remove something Vocalis
    needs.
    """
    before = read_name_list(baseline)
    if not before:
        return []
    added = installed() - before - _KEPT_ALWAYS
    return sorted(name for name in added if not name.startswith(_KEPT_PREFIXES))


def _print_lines(lines: Iterable[str]) -> None:
    """Print one item per line for the shell to read."""
    for line in lines:
        print(line)


def main(argv: list[str] | None = None) -> int:
    """Command line used by install.sh (see the module text)."""
    parser = argparse.ArgumentParser(prog="python -m vocalis.qwen3_addon")
    parser.add_argument("what", choices=["requirements", "snapshot", "removable"])
    parser.add_argument("file", nargs="?", default=str(BASELINE_FILE))
    args = parser.parse_args(argv)
    if args.what == "requirements":
        requirements = qwen3_requirements()
        if not requirements:
            print("Could not read the Qwen3 package list.", file=sys.stderr)
            return 1
        _print_lines(requirements)
    elif args.what == "snapshot":
        write_snapshot(Path(args.file), installed=installed_package_names)
    else:
        _print_lines(removable_packages(Path(args.file), installed=installed_package_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
