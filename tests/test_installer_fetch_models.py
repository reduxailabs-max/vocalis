"""Tests for packaging/installer/lib/fetch_models.py (no network: lookups and downloads are faked)."""
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "packaging" / "installer" / "lib" / "fetch_models.py"


@pytest.fixture()
def fm():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("fetch_models", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path) -> MagicMock:
    config = MagicMock()
    e = config.engines
    e.parakeet_model_dir = str(tmp_path / "parakeet")
    e.qwen3_tts_model_dir = str(tmp_path / "qwen")
    e.piper_voice_model_path = str(tmp_path / "voices" / "en.onnx")
    e.kokoro_model_path = str(tmp_path / "kokoro" / "kokoro-v1.0.onnx")
    e.kokoro_voices_path = str(tmp_path / "kokoro" / "voices-v1.0.bin")
    e.piper_binary_path = str(tmp_path / "bin" / "piper" / "piper")
    return config


def _touch(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _fake_hf(fm):  # type: ignore[no-untyped-def]  # noqa: ANN202
    calls: list[tuple[str, str, str | None]] = []

    def hf_item(repo, filename, target, revision=None):  # type: ignore[no-untyped-def]
        calls.append((repo, filename, revision))
        return fm.DownloadItem(f"http://hf/{repo}/{filename}", target, 10, None)

    return calls, hf_item


def _present_everything(fm, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    for name in fm.PARAKEET_MODEL_FILES:
        _touch(tmp_path / "parakeet" / name)
    for name in fm.QWEN3_TTS_MODEL_FILES:
        _touch(tmp_path / "qwen" / name)
    _touch(tmp_path / "voices" / "en.onnx")
    _touch(tmp_path / "voices" / "en.onnx.json")
    for name in fm.KOKORO_FILE_SIZES:
        _touch(tmp_path / "kokoro" / name, b"k" * 3)
    # The real Kokoro sizes are huge; use tiny ones for these tests.
    fm.KOKORO_FILE_SIZES = {name: 3 for name in fm.KOKORO_FILE_SIZES}


def test_everything_present_needs_no_network(fm, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _present_everything(fm, tmp_path)
    with patch.object(fm, "hf_item", side_effect=AssertionError("network!")), patch.object(
        fm, "qwen3_repo_files", side_effect=AssertionError("network!")
    ):
        assert fm.plan_items(_config(tmp_path), False, None) == []
    assert "already present" in capsys.readouterr().out


def test_missing_parakeet_files_are_planned(fm, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _present_everything(fm, tmp_path)
    (tmp_path / "parakeet" / fm.PARAKEET_MODEL_FILES[0]).unlink()
    calls, fake = _fake_hf(fm)  # type: ignore[no-untyped-call]
    with patch.object(fm, "hf_item", fake):
        items = fm.plan_items(_config(tmp_path), False, None)
    assert [c[1] for c in calls] == list(fm.PARAKEET_MODEL_FILES)
    assert len(items) == len(fm.PARAKEET_MODEL_FILES)


def test_qwen3_files_are_pinned_to_one_commit(fm, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _present_everything(fm, tmp_path)
    (tmp_path / "qwen" / "model.safetensors").unlink()
    calls, fake = _fake_hf(fm)  # type: ignore[no-untyped-call]
    with patch.object(fm, "hf_item", fake), patch.object(
        fm, "qwen3_repo_files", return_value=("abc123", ["config.json", "model.safetensors", "sub/tok.json"])
    ):
        items = fm.plan_items(_config(tmp_path), False, None)
    assert {c[2] for c in calls} == {"abc123"}
    assert items[-1].target == tmp_path / "qwen" / "sub" / "tok.json"


def test_skip_qwen3_never_asks_for_it(fm, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _present_everything(fm, tmp_path)
    (tmp_path / "qwen" / "model.safetensors").unlink()
    with patch.object(fm, "qwen3_repo_files", side_effect=AssertionError("no")):
        assert fm.plan_items(_config(tmp_path), True, None) == []
    assert "NOTE Skipping the Qwen3-TTS voice" in capsys.readouterr().out


def test_piper_voice_goes_to_flat_names(fm, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _present_everything(fm, tmp_path)
    (tmp_path / "voices" / "en.onnx.json").unlink()
    calls, fake = _fake_hf(fm)  # type: ignore[no-untyped-call]
    with patch.object(fm, "hf_item", fake):
        items = fm.plan_items(_config(tmp_path), False, None)
    assert [i.target for i in items] == [tmp_path / "voices" / "en.onnx", tmp_path / "voices" / "en.onnx.json"]


def test_kokoro_wrong_size_is_fetched_again(fm, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _present_everything(fm, tmp_path)
    _touch(tmp_path / "kokoro" / "voices-v1.0.bin", b"too long!")
    items = fm.plan_items(_config(tmp_path), False, None)
    assert [i.target.name for i in items] == ["voices-v1.0.bin"]
    assert items[0].url == fm.KOKORO_BASE_URL + "voices-v1.0.bin"
    assert not (tmp_path / "kokoro" / "voices-v1.0.bin").exists()


def test_piper_tarball_planned_only_when_binary_missing(fm, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _present_everything(fm, tmp_path)
    tar = tmp_path / "cache" / "piper.tar.gz"
    with patch.object(fm, "head_size", return_value=777):
        items = fm.plan_items(_config(tmp_path), False, tar)
        assert [(i.url, i.size, i.target) for i in items] == [(fm.PIPER_TARBALL_URL, 777, tar)]
        _touch(tmp_path / "bin" / "piper" / "piper")
        assert fm.plan_items(_config(tmp_path), False, tar) == []
        assert fm.plan_items(_config(tmp_path), False, None) == []


def test_hf_item_reads_size_and_lfs_checksum(fm) -> None:  # type: ignore[no-untyped-def]
    sha = "a" * 64
    meta = SimpleNamespace(size=123, etag=f'"{sha}"')
    with patch.object(fm, "hf_hub_url", return_value="http://hf/x"), patch.object(
        fm, "get_hf_file_metadata", return_value=meta
    ):
        item = fm.hf_item("r", "f.bin", Path("/t/f.bin"))
    assert (item.url, item.size, item.sha256) == ("http://hf/x", 123, sha)


def test_hf_item_ignores_non_checksum_etag(fm) -> None:  # type: ignore[no-untyped-def]
    meta = SimpleNamespace(size=5, etag="short")
    with patch.object(fm, "hf_hub_url", return_value="u"), patch.object(fm, "get_hf_file_metadata", return_value=meta):
        assert fm.hf_item("r", "f", Path("/t/f")).sha256 is None


def test_lookup_retries_then_gives_plain_error(fm) -> None:  # type: ignore[no-untyped-def]
    calls = []

    def bad():  # type: ignore[no-untyped-def]
        calls.append(1)
        raise OSError("offline")

    with patch.object(fm.time, "sleep") as sleep, pytest.raises(fm.DownloadError, match="internet connection"):
        fm._retry(bad, "thing")
    assert len(calls) == fm._LOOKUP_ATTEMPTS
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2, 4]


def test_lookup_succeeds_after_a_failure(fm) -> None:  # type: ignore[no-untyped-def]
    state = {"n": 0}

    def flaky() -> str:
        state["n"] += 1
        if state["n"] == 1:
            raise OSError("x")
        return "ok"

    with patch.object(fm.time, "sleep"):
        assert fm._retry(flaky, "t") == "ok"


def test_qwen3_repo_files_skips_gitattributes(fm) -> None:  # type: ignore[no-untyped-def]
    info = SimpleNamespace(sha="c0ffee", siblings=[SimpleNamespace(rfilename=n) for n in (".gitattributes", "a.json")])
    api = MagicMock()
    api.model_info.return_value = info
    with patch.object(fm, "HfApi", return_value=api):
        assert fm.qwen3_repo_files() == ("c0ffee", ["a.json"])


def test_head_size_reads_content_length(fm) -> None:  # type: ignore[no-untyped-def]
    response = MagicMock()
    response.headers = {"Content-Length": "4242"}
    response.__enter__.return_value = response
    with patch.object(fm.urllib.request, "urlopen", return_value=response) as opener:
        assert fm.head_size("http://x/file.tar.gz") == 4242
    assert opener.call_args.kwargs["timeout"] == fm.SOCKET_TIMEOUT_SECONDS


def test_run_downloads_refuses_when_disk_is_too_small(fm, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    item = fm.DownloadItem("http://x", tmp_path / "f", 10_000_000_000, None)
    with patch.object(fm, "_free_bytes", return_value=1_000_000_000), pytest.raises(fm.DownloadError, match="free disk space"):
        fm.run_downloads([item], threading.Event())


def test_run_downloads_announces_size_and_resumed(fm, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    item = fm.DownloadItem("http://x", tmp_path / "f", 3_400_000_000, None)
    runner = MagicMock()
    with patch.object(fm, "_free_bytes", return_value=10**12), patch.object(
        fm, "Downloader", side_effect=[MagicMock(resumed_bytes=lambda: 1_100_000_000), runner]
    ):
        fm.run_downloads([item], threading.Event())
    assert "NOTE Downloading 3.4 GB (1.1 GB already downloaded earlier)" in capsys.readouterr().out
    runner.run.assert_called_once()


def test_run_downloads_with_nothing_does_nothing(fm) -> None:  # type: ignore[no-untyped-def]
    fm.run_downloads([], threading.Event())


def test_free_bytes_walks_up_to_an_existing_folder(fm, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    assert fm._free_bytes(tmp_path / "a" / "b" / "c") > 0


def test_main_passes_flags_and_returns_zero(fm, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with patch.object(fm, "VocalisConfig", return_value=_config(tmp_path)), patch.object(
        fm, "plan_items", return_value=[]
    ) as plan, patch.object(fm, "run_downloads") as run:
        assert fm.main(["--skip-qwen3", "--piper-tarball", str(tmp_path / "p.tgz")]) == 0
    assert plan.call_args.args[1:] == (True, tmp_path / "p.tgz")
    run.assert_called_once()


def test_main_reports_a_download_error_and_returns_one(fm, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    with patch.object(fm, "VocalisConfig", return_value=_config(tmp_path)), patch.object(
        fm, "plan_items", side_effect=fm.DownloadError("no internet")
    ):
        assert fm.main([]) == 1
    assert "no internet" in capsys.readouterr().err
