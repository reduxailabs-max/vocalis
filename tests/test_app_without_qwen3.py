"""The app must start when the optional Qwen3-TTS helper programs are not installed."""
import subprocess
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")

_PROGRAM = f"""
import sys
sys.path.insert(0, {_SRC!r})  # this checkout's code, not some other installed copy
for name in ("qwen_tts", "torch", "torchaudio", "transformers", "accelerate", "einops"):
    sys.modules[name] = None  # importing any of them now fails, like an install without Qwen3-TTS
import vocalis.cli
import vocalis.ui.settings_window
import vocalis.startup_checks
import vocalis.hardware
print("started")
"""


def test_the_app_modules_import_without_the_qwen3_helper_programs() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROGRAM],
        capture_output=True,
        text=True,
        timeout=120,
        env={"QT_QPA_PLATFORM": "offscreen", "HOME": str(Path.home())},
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "started"


def test_the_core_dependencies_no_longer_list_the_qwen3_helpers() -> None:
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    core = text[text.index("dependencies = ["):text.index("[project.optional-dependencies]")]
    for name in ("qwen-tts", "transformers", "accelerate", "torchaudio", "einops"):
        assert name not in core
    extra = text[text.index("qwen3 = ["):text.index("dev = [")]
    for name in ("qwen-tts", "transformers", "accelerate", "torchaudio", "einops"):
        assert name in extra
