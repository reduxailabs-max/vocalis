"""Decides what the installer should put on this PC, based on the graphics card.

Pure apart from reading ``nvidia-smi``: every rule is unit-testable. The
shell installer calls ``python -m vocalis.installer.plan <what>`` to ask.
"""
import argparse
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from vocalis.engines.auto_select import AutoPolicy

_NVIDIA_SMI_COMMAND = [
    "nvidia-smi",
    "--query-gpu=name,memory.total",
    "--format=csv,noheader,nounits",
]
_NVIDIA_SMI_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class InstalledGpu:
    """The graphics card found in this PC (memory in megabytes)."""

    name: str
    total_mb: int


def parse_nvidia_smi(output: str) -> "InstalledGpu | None":
    """Read the first ``name, memory`` line of nvidia-smi output, or None if unusable."""
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        name, sep, memory = line.rpartition(",")
        name, memory = name.strip(), memory.strip()
        if not sep or not name or not memory.isdigit():
            return None
        return InstalledGpu(name=name, total_mb=int(memory))
    return None


def _run_nvidia_smi(command: list[str]) -> str:
    """Run the command and return its output; raises if it fails or hangs."""
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
    )
    return result.stdout


def read_gpu(run: Callable[[list[str]], str] = _run_nvidia_smi) -> "InstalledGpu | None":
    """Ask nvidia-smi about the graphics card; None if absent, broken, hung or unreadable."""
    try:
        output = run(_NVIDIA_SMI_COMMAND)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(output)


@dataclass(frozen=True)
class InstallPlan:
    """What to install: which runtime flavour and whether to fetch Qwen3-TTS."""

    gpu: "InstalledGpu | None"
    gpu_runtime: bool
    fetch_qwen3: bool

    def summary_lines(self) -> list[str]:
        """Plain-English lines describing this PC and what will be installed."""
        if self.gpu is None:
            lines = ["No NVIDIA graphics card found – using your processor"]
        else:
            lines = [f"Graphics card: {self.gpu.name} ({self.gpu.total_mb} MB memory)"]
        lines.append(
            "Voices to install: Piper, Kokoro, Qwen3-TTS (best quality)"
            if self.fetch_qwen3
            else "Voices to install: Piper and Kokoro"
        )
        lines.append("Piper: fast, light voice that works on any PC")
        lines.append("Kokoro: natural voice, runs on the graphics card when there is room")
        if self.fetch_qwen3:
            lines.append("Qwen3-TTS: the best-sounding voice, needs a strong graphics card")
        else:
            lines.append("Qwen3-TTS: skipped, your graphics card is too small or missing")
        return lines


def make_plan(gpu: "InstalledGpu | None", policy: AutoPolicy = AutoPolicy()) -> InstallPlan:
    """Build the install plan for the given graphics card (or None)."""
    return InstallPlan(
        gpu=gpu,
        gpu_runtime=gpu is not None,
        fetch_qwen3=gpu is not None and gpu.total_mb >= policy.qwen3_min_gpu_total_mb,
    )


def main(argv: "list[str] | None" = None) -> int:
    """Command line: print the installer's answer for ``gpu-runtime`` or ``fetch-flags``."""
    parser = argparse.ArgumentParser(prog="python -m vocalis.installer.plan")
    parser.add_argument("what", choices=["gpu-runtime", "fetch-flags"])
    args = parser.parse_args(argv)
    plan = make_plan(read_gpu())
    if args.what == "gpu-runtime":
        print("yes" if plan.gpu_runtime else "no")
    else:
        print("" if plan.fetch_qwen3 else "--skip-qwen3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
