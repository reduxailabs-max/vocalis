"""Reads what the computer has right now (GPU memory, RAM, CPU load) so
Vocalis can explain itself in the log today and pick a voice engine later.
Every source is optional and injectable; nothing here ever raises."""
import importlib.util
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger("vocalis")

_KB_PER_MB = 1024
_BYTES_PER_MB = 1024 * 1024
_NVIDIA_SMI_COMMAND = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.free",
    "--format=csv,noheader,nounits",
]
_NVIDIA_SMI_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class GpuInfo:
    """One CUDA GPU: its name and memory in megabytes."""

    name: str
    total_mb: int
    free_mb: int


@dataclass(frozen=True)
class HardwareInfo:
    """A snapshot of the machine; a field is None when it could not be read."""

    gpu: GpuInfo | None
    ram_total_mb: int | None
    ram_available_mb: int | None
    cpu_count: int | None
    load_1min: float | None
    own_process_ram_mb: int | None = None


def query_gpu_with_torch() -> GpuInfo | None:
    """Return the first CUDA GPU's name and memory, or None if there is none."""
    import torch  # slow import, kept local

    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return GpuInfo(torch.cuda.get_device_name(0), total_bytes // _BYTES_PER_MB, free_bytes // _BYTES_PER_MB)


def parse_nvidia_smi_memory(output: str) -> GpuInfo | None:
    """Read the first ``name, total, free`` line of nvidia-smi output, or None if unusable."""
    for line in output.splitlines():
        parts = line.rsplit(",", 2)
        if len(parts) < 3:
            continue
        name, total, free = (part.strip() for part in parts)
        if not (total.isascii() and total.isdigit() and free.isascii() and free.isdigit()):
            return None
        if not name or int(total) <= 0:
            return None
        return GpuInfo(name, int(total), int(free))
    return None


def query_gpu_with_nvidia_smi() -> GpuInfo | None:
    """Ask nvidia-smi about the first graphics card (works without PyTorch); None if unavailable."""
    try:
        result = subprocess.run(
            _NVIDIA_SMI_COMMAND,
            capture_output=True,
            text=True,
            errors="replace",
            check=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return parse_nvidia_smi_memory(result.stdout)


def query_gpu() -> GpuInfo | None:
    """Read the graphics card with PyTorch when it is installed, else with nvidia-smi.

    PyTorch belongs to the optional Qwen3-TTS add-on, so it may be absent.
    """
    if importlib.util.find_spec("torch") is None:
        return query_gpu_with_nvidia_smi()
    return query_gpu_with_torch()


def query_own_process_ram_mb(status_path: Path = Path("/proc/self/status")) -> "int | None":
    """Return Vocalis's own current resident memory in MB (the VmRSS line), or None if unreadable."""
    try:
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) // _KB_PER_MB
    except OSError:
        return None
    return None


def _read_meminfo(path: Path) -> tuple[int | None, int | None]:
    """Return (total, available) RAM in MB from a /proc/meminfo-style file."""
    try:
        values: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[key.strip()] = int(parts[0])
    except OSError:
        return None, None
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    return (
        total // _KB_PER_MB if total is not None else None,
        available // _KB_PER_MB if available is not None else None,
    )


def _read_load(path: Path) -> float | None:
    """Return the 1-minute load average from a /proc/loadavg-style file."""
    try:
        return float(path.read_text(encoding="utf-8", errors="replace").split()[0])
    except (OSError, ValueError, IndexError):
        return None


def probe_hardware(
    meminfo_path: Path = Path("/proc/meminfo"),
    loadavg_path: Path = Path("/proc/loadavg"),
    cpu_count: Callable[[], int | None] = os.cpu_count,
    gpu_query: Callable[[], GpuInfo | None] | None = None,
    own_ram_query: Callable[[], "int | None"] = query_own_process_ram_mb,
) -> HardwareInfo:
    """Take a snapshot of GPU, RAM, CPU and Vocalis's own memory use; unreadable parts become None."""
    query = gpu_query if gpu_query is not None else query_gpu
    try:
        gpu = query()
    except Exception:
        logger.debug("GPU query failed; treating as no GPU", exc_info=True)
        gpu = None
    try:
        cpus = cpu_count()
    except Exception:
        logger.debug("CPU count failed; treating as unknown", exc_info=True)
        cpus = None
    try:
        own_ram = own_ram_query()
    except Exception:
        logger.debug("Own-process RAM query failed; treating as unknown", exc_info=True)
        own_ram = None
    ram_total, ram_available = _read_meminfo(meminfo_path)
    return HardwareInfo(gpu, ram_total, ram_available, cpus, _read_load(loadavg_path), own_ram)


def describe(info: HardwareInfo) -> str:
    """One plain-English line summarising a snapshot, for the log."""
    gpu = (
        f"GPU {info.gpu.name} ({info.gpu.free_mb} of {info.gpu.total_mb} MB free)"
        if info.gpu is not None
        else "no GPU"
    )
    ram = f"RAM {info.ram_available_mb} of {info.ram_total_mb} MB free" if info.ram_total_mb else "RAM unknown"
    cpu = f"{info.cpu_count} CPU cores" if info.cpu_count else "CPU cores unknown"
    load = f"load {info.load_1min:.2f}" if info.load_1min is not None else "load unknown"
    return f"Hardware: {gpu}; {ram}; {cpu}; {load}"


def log_hardware_summary() -> None:
    """Probe the machine and write one INFO line; any failure is a warning only."""
    try:
        logger.info(describe(probe_hardware()))
    except Exception:
        logger.warning("Could not read hardware details", exc_info=True)
