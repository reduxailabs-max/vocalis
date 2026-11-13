"""Download the Vocalis voice and dictation models, resuming interrupted runs.

Run with the just-installed vocalis venv's python so it can import vocalis's
own config defaults and file lists as the single source of truth -- the
installer must never hard-code a second copy of these paths/filenames.

This script only works out *what* is missing (names, sizes, checksums, web
addresses). The actual downloading is done by vocalis.installer.downloader,
which fetches many pieces at once, retries, resumes after a stop, and prints
`PROGRESS done total speed eta` lines for the setup window.

Assets: the Parakeet dictation model, the Qwen3-TTS voice, the Piper fallback
voice (all Hugging Face), the two Kokoro voice files (a GitHub release) and
the Piper program's tarball (a GitHub release; models.sh unpacks it).
"""
import argparse
import shutil
import signal
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

from vocalis.config import VocalisConfig
from vocalis.installer.download_progress import Reporter, SpeedMeter, format_bytes
from vocalis.installer.downloader import (
    SOCKET_TIMEOUT_SECONDS,
    DownloadError,
    DownloadItem,
    Downloader,
)
from vocalis.startup_checks import PARAKEET_MODEL_FILES, QWEN3_TTS_MODEL_FILES

PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
QWEN3_TTS_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
PIPER_VOICE_REPO = "rhasspy/piper-voices"
PIPER_VOICE_FILES = (
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
)
KOKORO_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILE_SIZES = {"kokoro-v1.0.onnx": 325532387, "voices-v1.0.bin": 28214398}
PIPER_TARBALL_URL = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
HEADROOM_BYTES = 500 * 1000 * 1000
_LOOKUP_ATTEMPTS = 4
_SHA256_LENGTH = 64
_IGNORED_REPO_FILES = {".gitattributes"}


def _retry(action: Any, what: str) -> Any:
    """Run a small network lookup, retrying a few times with a growing pause."""
    last: Exception | None = None
    for attempt in range(_LOOKUP_ATTEMPTS):
        try:
            return action()
        except Exception as error:  # noqa: BLE001 - any network problem is retried
            last = error
            if attempt + 1 < _LOOKUP_ATTEMPTS:
                time.sleep(2**attempt)
    raise DownloadError(
        f"Could not reach the download server to look up {what} ({last}). "
        "Check your internet connection and try again."
    )


def hf_item(repo: str, filename: str, target: Path, revision: str | None = None) -> DownloadItem:
    """Describe one Hugging Face file: its web address, exact size and checksum."""
    url = hf_hub_url(repo_id=repo, filename=filename, revision=revision)
    meta = _retry(lambda: get_hf_file_metadata(url), filename)
    etag = (meta.etag or "").strip('"')
    sha = etag.lower() if len(etag) == _SHA256_LENGTH and all(c in "0123456789abcdefABCDEF" for c in etag) else None
    return DownloadItem(url=url, target=target, size=int(meta.size), sha256=sha)


def qwen3_repo_files() -> tuple[str, list[str]]:
    """The Qwen3-TTS repo's current commit and the files in it."""
    info = _retry(lambda: HfApi().model_info(QWEN3_TTS_REPO), "the Qwen3-TTS voice")
    names = [s.rfilename for s in info.siblings or [] if s.rfilename not in _IGNORED_REPO_FILES]
    return str(info.sha), names


def head_size(url: str) -> int:
    """Size in bytes of a web file, from a HEAD request."""

    def ask() -> int:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "vocalis-installer"})
        with urllib.request.urlopen(request, timeout=SOCKET_TIMEOUT_SECONDS) as response:
            return int(response.headers["Content-Length"])

    return int(_retry(ask, Path(url).name))


def _needs(target: Path, size: int | None) -> bool:
    """Whether a file is missing (or, when its size is known, the wrong size)."""
    if not target.exists():
        return True
    return size is not None and target.stat().st_size != size


def _qwen3_items(config: VocalisConfig, skip_qwen3: bool) -> list[DownloadItem]:
    """The optional Qwen3-TTS voice files still missing (nothing at all when skipped)."""
    qwen_dir = Path(config.engines.qwen3_tts_model_dir)
    if skip_qwen3:
        print("NOTE Qwen3-TTS (optional extra voice) was not chosen, so it is not downloaded.")
        return []
    if any(_needs(qwen_dir / name, None) for name in QWEN3_TTS_MODEL_FILES):
        revision, names = qwen3_repo_files()
        return [hf_item(QWEN3_TTS_REPO, name, qwen_dir / name, revision) for name in names]
    print(f"Qwen3-TTS model already present at {qwen_dir}")
    return []


def _base_items(config: VocalisConfig, piper_tarball: Path | None) -> list[DownloadItem]:
    """The always-needed files still missing: Parakeet, Piper, Kokoro and the Piper program."""
    engines = config.engines
    items: list[DownloadItem] = []

    parakeet_dir = Path(engines.parakeet_model_dir)
    if any(_needs(parakeet_dir / name, None) for name in PARAKEET_MODEL_FILES):
        for name in PARAKEET_MODEL_FILES:
            items.append(hf_item(PARAKEET_REPO, name, parakeet_dir / name))
    else:
        print(f"Parakeet model already present at {parakeet_dir}")

    voice = Path(engines.piper_voice_model_path)
    voice_targets = (voice, Path(str(voice) + ".json"))
    if any(_needs(t, None) for t in voice_targets):
        for filename, target in zip(PIPER_VOICE_FILES, voice_targets, strict=True):
            items.append(hf_item(PIPER_VOICE_REPO, filename, target))
    else:
        print(f"Piper voice already present at {voice}")

    kokoro = (Path(engines.kokoro_model_path), Path(engines.kokoro_voices_path))
    if any(_needs(path, KOKORO_FILE_SIZES[path.name]) for path in kokoro):
        for path in kokoro:
            if _needs(path, KOKORO_FILE_SIZES[path.name]):
                path.unlink(missing_ok=True)
                items.append(DownloadItem(KOKORO_BASE_URL + path.name, path, KOKORO_FILE_SIZES[path.name]))
    else:
        print(f"Kokoro voice already present at {kokoro[0].parent}")

    if piper_tarball is not None and not Path(engines.piper_binary_path).exists() and _needs(piper_tarball, None):
        items.append(DownloadItem(PIPER_TARBALL_URL, piper_tarball, head_size(PIPER_TARBALL_URL)))
    return items


def plan_items(
    config: VocalisConfig, skip_qwen3: bool, piper_tarball: Path | None, only_qwen3: bool = False
) -> list[DownloadItem]:
    """Work out the files still to download. Touches the network only if something is missing.

    With only_qwen3 just the optional Qwen3-TTS voice is planned (used when it
    is added later from Settings).
    """
    if only_qwen3:
        return _qwen3_items(config, skip_qwen3=False)
    return _base_items(config, piper_tarball) + _qwen3_items(config, skip_qwen3)


def _free_bytes(path: Path) -> int:
    """Free disk space at the nearest folder that already exists."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def run_downloads(items: list[DownloadItem], stop: threading.Event) -> None:
    """Show the size, check disk space, then download everything."""
    if not items:
        return
    total = sum(i.size for i in items)
    probe = Downloader(items)
    resumed = probe.resumed_bytes()
    left = total - resumed
    earlier = f" ({format_bytes(resumed)} already downloaded earlier)" if resumed else ""
    print(f"NOTE Downloading {format_bytes(total)}{earlier}")
    needed = left + HEADROOM_BYTES
    free = _free_bytes(items[0].target.parent)
    if free < needed:
        raise DownloadError(
            f"There is not enough free disk space. About {format_bytes(needed)} is needed "
            f"but only {format_bytes(free)} is free. Free some space and try again."
        )
    meter = SpeedMeter(total, already_done=resumed)
    reporter = Reporter(meter)
    Downloader(items, stop=stop, meter=meter, reporter=reporter).run()


def main(argv: list[str] | None = None) -> int:
    """Ensure the downloadable assets are present, at vocalis's own default paths.

    Pass --skip-qwen3 to leave out the optional Qwen3-TTS voice.
    Pass --only-qwen3 to download only that voice (Settings uses this).
    Pass --piper-tarball PATH to also fetch the Piper program's archive to PATH.
    """
    parser = argparse.ArgumentParser(description="Download the Vocalis voice and dictation models.")
    parser.add_argument(
        "--skip-qwen3", action="store_true", help="do not download the optional Qwen3-TTS voice"
    )
    parser.add_argument("--only-qwen3", action="store_true", help="download only the optional Qwen3-TTS voice")
    parser.add_argument("--piper-tarball", type=Path, default=None, help="also download the Piper program archive here")
    args = parser.parse_args(argv)
    stop = threading.Event()
    in_main_thread = threading.current_thread() is threading.main_thread()
    if in_main_thread:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda *_: stop.set())
    try:
        run_downloads(plan_items(VocalisConfig(), args.skip_qwen3, args.piper_tarball, args.only_qwen3), stop)
    except DownloadError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
