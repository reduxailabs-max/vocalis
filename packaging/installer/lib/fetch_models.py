"""Reuse-or-download the three Hugging-Face-hosted Vocalis assets:
the Parakeet dictation model, the Qwen3-TTS voice model, and the
Piper fallback voice. Run with the just-installed vocalis venv's
python so it can import vocalis's own config defaults and file lists
as the single source of truth -- the installer must never hard-code
a second copy of these paths/filenames and risk drifting out of sync.

The Piper BINARY (not hosted on Hugging Face) is handled separately,
in models.sh, via a plain curl+tar download.
"""
import sys
from pathlib import Path

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


def main() -> int:
    """Ensure all three Hugging-Face-hosted assets are present, at vocalis's own default paths."""
    config = VocalisConfig()
    ensure_parakeet(Path(config.engines.parakeet_model_dir))
    ensure_qwen3_tts(Path(config.engines.qwen3_tts_model_dir))
    ensure_piper_voice(Path(config.engines.piper_voice_model_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
