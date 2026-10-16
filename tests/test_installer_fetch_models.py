"""Tests for packaging/installer/lib/fetch_models.py.

Loaded by file path (it lives outside the vocalis package, under the
installer's own tooling folder) with huggingface_hub's two download
functions patched -- no real network access.
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "packaging" / "installer" / "lib" / "fetch_models.py"


def _load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("fetch_models", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fetch_models():  # type: ignore[no-untyped-def]
    return _load_module()  # type: ignore[no-untyped-call]


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_ensure_parakeet_skips_download_when_all_files_present(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model_dir = tmp_path / "parakeet"
    for name in fetch_models.PARAKEET_MODEL_FILES:
        _touch(model_dir / name)
    with patch.object(fetch_models, "hf_hub_download") as mock_download:
        fetch_models.ensure_parakeet(model_dir)
    mock_download.assert_not_called()


def test_ensure_parakeet_downloads_every_file_when_missing(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model_dir = tmp_path / "parakeet"
    with patch.object(fetch_models, "hf_hub_download") as mock_download:
        fetch_models.ensure_parakeet(model_dir)
    assert mock_download.call_count == len(fetch_models.PARAKEET_MODEL_FILES)
    for call in mock_download.call_args_list:
        assert call.kwargs["repo_id"] == fetch_models.PARAKEET_REPO
        assert call.kwargs["local_dir"] == str(model_dir)


def test_ensure_qwen3_tts_skips_download_when_all_files_present(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model_dir = tmp_path / "qwen"
    for name in fetch_models.QWEN3_TTS_MODEL_FILES:
        _touch(model_dir / name)
    with patch.object(fetch_models, "snapshot_download") as mock_snapshot:
        fetch_models.ensure_qwen3_tts(model_dir)
    mock_snapshot.assert_not_called()


def test_ensure_qwen3_tts_downloads_the_full_snapshot_when_missing(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    model_dir = tmp_path / "qwen"
    with patch.object(fetch_models, "snapshot_download") as mock_snapshot:
        fetch_models.ensure_qwen3_tts(model_dir)
    mock_snapshot.assert_called_once_with(repo_id=fetch_models.QWEN3_TTS_REPO, local_dir=str(model_dir))


def test_ensure_piper_voice_skips_download_when_both_files_present(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    voice_path = tmp_path / "piper-voices" / "en_US-lessac-medium.onnx"
    _touch(voice_path)
    _touch(Path(str(voice_path) + ".json"))
    with patch.object(fetch_models, "hf_hub_download") as mock_download:
        fetch_models.ensure_piper_voice(voice_path)
    mock_download.assert_not_called()


def test_ensure_piper_voice_downloads_and_flattens_both_files_when_missing(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    voice_path = tmp_path / "piper-voices" / "en_US-lessac-medium.onnx"

    def fake_download(repo_id: str, filename: str, local_dir: str) -> str:  # noqa: ANN001
        nested = Path(local_dir) / filename
        _touch(nested)
        return str(nested)

    with patch.object(fetch_models, "hf_hub_download", side_effect=fake_download) as mock_download:
        fetch_models.ensure_piper_voice(voice_path)

    assert mock_download.call_count == len(fetch_models.PIPER_VOICE_FILES)
    assert voice_path.exists()
    assert Path(str(voice_path) + ".json").exists()


def test_main_runs_all_four_ensures_in_order(fetch_models, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    calls = []
    fake_config = MagicMock()
    fake_config.engines.parakeet_model_dir = str(tmp_path / "parakeet")
    fake_config.engines.qwen3_tts_model_dir = str(tmp_path / "qwen")
    fake_config.engines.piper_voice_model_path = str(tmp_path / "piper-voices" / "en_US-lessac-medium.onnx")
    fake_config.engines.kokoro_model_path = str(tmp_path / "kokoro" / "kokoro-v1.0.onnx")
    fake_config.engines.kokoro_voices_path = str(tmp_path / "kokoro" / "voices-v1.0.bin")
    with (
        patch.object(fetch_models, "VocalisConfig", return_value=fake_config),
        patch.object(fetch_models, "ensure_parakeet", side_effect=lambda p: calls.append(("parakeet", p))),
        patch.object(fetch_models, "ensure_qwen3_tts", side_effect=lambda p: calls.append(("qwen3_tts", p))),
        patch.object(fetch_models, "ensure_piper_voice", side_effect=lambda p: calls.append(("piper_voice", p))),
        patch.object(fetch_models, "ensure_kokoro", side_effect=lambda m, v: calls.append(("kokoro", m))),
    ):
        assert fetch_models.main() == 0
    assert [name for name, _ in calls] == ["parakeet", "qwen3_tts", "piper_voice", "kokoro"]


class _FakeResponse:
    """A fake urlopen() result that yields fixed bytes."""

    def __init__(self, data: bytes) -> None:
        import io
        self._buffer = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def test_ensure_kokoro_downloads_both_files_and_checks_their_size(fetch_models, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Both files are downloaded to the configured paths and size-checked."""
    sizes = {"kokoro-v1.0.onnx": 10, "voices-v1.0.bin": 4}
    urls: list[str] = []

    def opener(url: str) -> _FakeResponse:
        urls.append(url)
        return _FakeResponse(b"x" * sizes[url.rsplit("/", 1)[1]])

    model, voices = tmp_path / "k" / "kokoro-v1.0.onnx", tmp_path / "k" / "voices-v1.0.bin"
    fetch_models.ensure_kokoro(model, voices, opener=opener, expected_sizes=sizes)
    assert model.read_bytes() == b"x" * 10 and voices.read_bytes() == b"x" * 4
    assert all("thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/" in url for url in urls)


def test_ensure_kokoro_skips_files_that_are_already_there(fetch_models, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Nothing is downloaded when both files already exist."""
    model, voices = tmp_path / "kokoro-v1.0.onnx", tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")

    def opener(url: str) -> _FakeResponse:
        raise AssertionError("must not download")

    fetch_models.ensure_kokoro(model, voices, opener=opener)


def test_ensure_kokoro_rejects_a_truncated_download_and_leaves_no_partial_file(fetch_models, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A short download raises and never leaves a half file that would look complete."""
    model, voices = tmp_path / "kokoro-v1.0.onnx", tmp_path / "voices-v1.0.bin"
    with pytest.raises(RuntimeError, match="kokoro-v1.0.onnx"):
        fetch_models.ensure_kokoro(
            model, voices, opener=lambda url: _FakeResponse(b"xx"),
            expected_sizes={"kokoro-v1.0.onnx": 10, "voices-v1.0.bin": 4},
        )
    assert not model.exists()
    assert list(tmp_path.glob("*.part")) == []
