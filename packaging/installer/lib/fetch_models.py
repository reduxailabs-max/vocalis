"""Reuse-or-download four Vocalis assets: the Parakeet dictation model,
the Qwen3-TTS voice model and the Piper fallback voice (all Hugging
Face hosted), and the two Kokoro voice files (a GitHub release). Run with the just-installed vocalis venv's
python so it can import vocalis's own config defaults and file lists
as the single source of truth -- the installer must never hard-code
a second copy of these paths/filenames and risk drifting out of sync.

The Piper BINARY (not hosted on Hugging Face) is handled separately,
in models.sh, via a plain curl+tar download.
"""
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import hf_hub_download, snapshot_download

from vocalis.config import VocalisConfig
from vocalis.startup_checks import PARAKEET_MODEL_FILES, QWEN3_TTS_MODEL_FILES

# PARAKEET_REPO and QWEN3_TTS_REPO were confirmed directly from this
# machine's own Hugging Face cache folder names
# (~/.cache/huggingface/hub/models--*). PIPER_VOICE_FILES' path follows
# rhasspy/piper-voices' well-documented, standard layout but was NOT
# independently re-confirmed over the network while writing this plan
# (no outbound network access at plan-writing time) -- if it's wrong,
# hf_hub_download raises a loud, clear "not found" error rather than
# silently installing the wrong file, so double-check this one path
# against https://huggingface.co/rhasspy/piper-voices during this
# task if network access is available, and fix here if it's changed.
PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
QWEN3_TTS_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
PIPER_VOICE_REPO = "rhasspy/piper-voices"
PIPER_VOICE_FILES = (
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
)

KOKORO_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILE_SIZES = {"kokoro-v1.0.onnx": 325532387, "voices-v1.0.bin": 28214398}
_CHUNK_BYTES = 1024 * 1024
_DOWNLOAD_TIMEOUT_SECONDS = 60


def ensure_parakeet(model_dir: Path) -> None:
    """Download the Parakeet model's files into model_dir, unless already present.

    huggingface_hub downloads to a temp location and only renames into
    place on success, so a half-finished earlier download never looks
    like a complete one here.
    """
    if all((model_dir / name).exists() for name in PARAKEET_MODEL_FILES):
        print(f"Parakeet model already present at {model_dir}")
        return
    print(f"Downloading Parakeet model to {model_dir} (about 650MB) ...")
    for name in PARAKEET_MODEL_FILES:
        hf_hub_download(repo_id=PARAKEET_REPO, filename=name, local_dir=str(model_dir))


def ensure_qwen3_tts(model_dir: Path) -> None:
    """Download the full Qwen3-TTS snapshot into model_dir, unless already present."""
    if all((model_dir / name).exists() for name in QWEN3_TTS_MODEL_FILES):
        print(f"Qwen3-TTS model already present at {model_dir}")
        return
    print(f"Downloading Qwen3-TTS model to {model_dir} (about 2.4GB) ...")
    snapshot_download(repo_id=QWEN3_TTS_REPO, local_dir=str(model_dir))


def ensure_piper_voice(voice_path: Path) -> None:
    """Download the Piper fallback voice to voice_path, unless already present.

    hf_hub_download preserves the repo's nested folder layout under
    local_dir; the two downloaded files are moved up to the flat path
    vocalis's config expects.
    """
    companion = Path(str(voice_path) + ".json")
    if voice_path.exists() and companion.exists():
        print(f"Piper voice already present at {voice_path}")
        return
    print(f"Downloading Piper voice to {voice_path.parent} ...")
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    for filename in PIPER_VOICE_FILES:
        downloaded = Path(hf_hub_download(repo_id=PIPER_VOICE_REPO, filename=filename, local_dir=str(voice_path.parent)))
        target = voice_path.parent / Path(filename).name
        if downloaded != target:
            downloaded.rename(target)


def _open_url(url: str) -> Any:
    """Open a URL with a timeout so a stalled connection cannot hang the installer."""
    return urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS)


def _download(url: str, target: Path, expected_size: int, opener: Callable[[str], Any]) -> None:
    """Download url to target via a .part file, checking the size before renaming into place."""
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with opener(url) as response, part.open("wb") as handle:
            while True:
                block = response.read(_CHUNK_BYTES)
                if not block:
                    break
                handle.write(block)
        if part.stat().st_size != expected_size:
            raise RuntimeError(f"{target.name} downloaded incompletely (expected {expected_size} bytes)")
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)


def ensure_kokoro(
    model_path: Path,
    voices_path: Path,
    opener: Callable[[str], Any] = _open_url,
    expected_sizes: dict[str, int] = KOKORO_FILE_SIZES,
) -> None:
    """Download the Kokoro model and voices files, unless both are already present."""
    if model_path.exists() and voices_path.exists():
        print(f"Kokoro voice already present at {model_path.parent}")
        return
    print(f"Downloading the Kokoro voice to {model_path.parent} (about 350MB) ...")
    for path in (model_path, voices_path):
        if not path.exists():
            _download(KOKORO_BASE_URL + path.name, path, expected_sizes[path.name], opener)


def main() -> int:
    """Ensure all four downloadable assets are present, at vocalis's own default paths."""
    config = VocalisConfig()
    ensure_parakeet(Path(config.engines.parakeet_model_dir))
    ensure_qwen3_tts(Path(config.engines.qwen3_tts_model_dir))
    ensure_piper_voice(Path(config.engines.piper_voice_model_path))
    ensure_kokoro(Path(config.engines.kokoro_model_path), Path(config.engines.kokoro_voices_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
