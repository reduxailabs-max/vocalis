"""Tests for vocalis.hardware — every data source is faked; no real /proc or GPU."""
import logging
from pathlib import Path

from vocalis.hardware import GpuInfo, HardwareInfo, describe, log_hardware_summary, probe_hardware

MEMINFO = "MemTotal:       16000000 kB\nMemFree:  1000000 kB\nMemAvailable:    8000000 kB\n"


def _files(tmp_path: Path, meminfo: str = MEMINFO, loadavg: str = "1.50 1.00 0.50 2/300 42\n") -> tuple[Path, Path]:
    m, l = tmp_path / "meminfo", tmp_path / "loadavg"
    m.write_text(meminfo)
    l.write_text(loadavg)
    return m, l


def test_reads_ram_load_cpu_and_gpu(tmp_path: Path) -> None:
    m, l = _files(tmp_path)
    info = probe_hardware(
        m, l, cpu_count=lambda: 8, gpu_query=lambda: GpuInfo("RTX", 8188, 7000), own_ram_query=lambda: None,
    )
    assert info == HardwareInfo(GpuInfo("RTX", 8188, 7000), 15625, 7812, 8, 1.5, None)


def test_missing_proc_files_give_none_not_an_error(tmp_path: Path) -> None:
    info = probe_hardware(
        tmp_path / "nope", tmp_path / "nope2", cpu_count=lambda: None, gpu_query=lambda: None, own_ram_query=lambda: None,
    )
    assert info == HardwareInfo(None, None, None, None, None, None)


def test_reads_own_process_ram(tmp_path: Path) -> None:
    m, l = _files(tmp_path)
    info = probe_hardware(
        m, l, cpu_count=lambda: 8, gpu_query=lambda: GpuInfo("RTX", 8188, 7000), own_ram_query=lambda: 512,
    )
    assert info.own_process_ram_mb == 512


def test_a_raising_own_ram_query_means_unknown(tmp_path: Path) -> None:
    m, l = _files(tmp_path)

    def boom() -> "int | None":
        raise RuntimeError("resource module unavailable")

    info = probe_hardware(m, l, cpu_count=lambda: 4, gpu_query=lambda: None, own_ram_query=boom)
    assert info.own_process_ram_mb is None


def test_query_own_process_ram_mb_reads_current_vmrss_in_mb(tmp_path: Path) -> None:
    from vocalis.hardware import query_own_process_ram_mb

    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmPeak:\t  999999 kB\nVmRSS:\t  204800 kB\nThreads:\t4\n")
    assert query_own_process_ram_mb(status) == 200


def test_query_own_process_ram_mb_without_vmrss_line_is_unknown(tmp_path: Path) -> None:
    from vocalis.hardware import query_own_process_ram_mb

    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmPeak:\t  999999 kB\n")
    assert query_own_process_ram_mb(status) is None


def test_query_own_process_ram_mb_missing_file_is_unknown(tmp_path: Path) -> None:
    from vocalis.hardware import query_own_process_ram_mb

    assert query_own_process_ram_mb(tmp_path / "nope") is None


def test_query_own_process_ram_mb_reads_this_process_on_this_machine() -> None:
    from vocalis.hardware import query_own_process_ram_mb

    value = query_own_process_ram_mb()
    assert value is not None and value > 0


def test_garbage_proc_content_gives_none(tmp_path: Path) -> None:
    m, l = _files(tmp_path, meminfo="garbage\n", loadavg="x y z\n")
    info = probe_hardware(m, l, cpu_count=lambda: 4, gpu_query=lambda: None)
    assert info.ram_total_mb is None and info.ram_available_mb is None and info.load_1min is None


def test_a_raising_gpu_query_means_no_gpu(tmp_path: Path) -> None:
    m, l = _files(tmp_path)

    def boom() -> GpuInfo | None:
        raise RuntimeError("CUDA driver missing")

    assert probe_hardware(m, l, cpu_count=lambda: 4, gpu_query=boom).gpu is None


def test_a_raising_cpu_count_means_cpu_count_is_none(tmp_path: Path) -> None:
    m, l = _files(tmp_path)

    def boom() -> int | None:
        raise RuntimeError("cpu count unavailable")

    assert probe_hardware(m, l, cpu_count=boom, gpu_query=lambda: None).cpu_count is None


def test_describe_is_one_plain_line() -> None:
    line = describe(HardwareInfo(GpuInfo("RTX 4060", 8188, 7000), 15625, 7812, 8, 1.5))
    assert "RTX 4060" in line and "8188" in line and "8 CPU" in line and "\n" not in line
    assert "no GPU" in describe(HardwareInfo(None, None, None, None, None))


def test_log_hardware_summary_never_raises(monkeypatch, caplog) -> None:  # type: ignore[no-untyped-def]
    def boom(*_a, **_k):  # type: ignore[no-untyped-def]
        raise RuntimeError("probe blew up")

    monkeypatch.setattr("vocalis.hardware.probe_hardware", boom)
    with caplog.at_level(logging.WARNING, logger="vocalis"):
        log_hardware_summary()
    assert "Could not read hardware details" in caplog.text
