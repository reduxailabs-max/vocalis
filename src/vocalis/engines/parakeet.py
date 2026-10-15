"""Parakeet speech-to-text engine, implementing the ModelEngine protocol so
it can be managed by ModelManager's single-residency/idle-unload policy.

Uses the `onnx-asr` package to run NVIDIA's Parakeet TDT model (the same
ONNX files the existing Voxtype installation already uses) via ONNX
Runtime, on GPU (CUDA) by default with a CPU fallback available for
testing/debugging.

cuDNN is required for the CUDA execution provider but is not bundled with
onnxruntime-gpu; it's installed as the `nvidia-cudnn-cu12` pip package and
must be preloaded via ctypes before the CUDA session is created, since its
.so files live inside site-packages rather than the system library path.

Since Phase 3's Qwen3-TTS engine added `huggingface-hub` as a project
dependency, that package is now importable process-wide for the first
time. This unlocks a previously-dormant fallback path inside
`onnx_asr.load_model()`: when given a model directory that doesn't
contain valid local files, it now attempts a live Hugging Face Hub
network lookup instead of raising a fast local error, hanging on network
retries. That violates this project's fully-offline, no-cloud-APIs
design rule. Hub/transformers offline mode is forced in
`vocalis/__init__.py` (which always runs before this module can be
imported), not here, so the guard is not import-order-dependent.
"""
import ctypes
import glob
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnx_asr

logger = logging.getLogger("vocalis")

_MODEL_TYPE = "nemo-parakeet-tdt-0.6b-v3"


class ParakeetLoadError(Exception):
    """Raised when the Parakeet model fails to load."""


class ParakeetSTT:
    """Speech-to-text engine backed by the local Parakeet TDT ONNX model."""

    def __init__(
        self,
        model_dir: Path,
        quantization: str = "int8",
        use_gpu: bool = True,
    ) -> None:
        """Create a ParakeetSTT engine (does not load the model yet).

        Args:
            model_dir: Directory containing the encoder/decoder ONNX files,
                vocab.txt, and config.json (the existing Voxtype Parakeet
                model directory).
            quantization: Quantization suffix matching the model files,
                e.g. "int8".
            use_gpu: If True, use CUDAExecutionProvider; if False, force
                CPUExecutionProvider (useful for testing without a GPU).
        """
        self._model_dir = model_dir
        self._quantization = quantization
        self._use_gpu = use_gpu
        self._model: onnx_asr.adapters.TextResultsAsrAdapter | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the ONNX model into memory (and GPU VRAM if use_gpu=True).

        Raises:
            ParakeetLoadError: If the model files are missing or loading
                otherwise fails.
        """
        if self._use_gpu:
            _preload_cudnn()
        providers = ["CUDAExecutionProvider"] if self._use_gpu else ["CPUExecutionProvider"]
        try:
            self._model = onnx_asr.load_model(
                _MODEL_TYPE,
                str(self._model_dir),
                quantization=self._quantization,
                providers=providers,
            )
        except Exception as exc:
            raise ParakeetLoadError(
                f"Failed to load Parakeet model from {self._model_dir}: {exc}"
            ) from exc

    def unload(self) -> None:
        """Release the model reference. Safe to call even if never loaded."""
        self._model = None

    def transcribe(self, audio: npt.NDArray[np.float32], sample_rate: int) -> str:
        """Transcribe a float32 audio waveform to text.

        Args:
            audio: 1-D float32 numpy array of audio samples.
            sample_rate: Sample rate of `audio` in Hz.

        Returns:
            The transcribed text (may be an empty string for silence).

        Raises:
            RuntimeError: If called before load().
        """
        if self._model is None:
            raise RuntimeError("ParakeetSTT.transcribe() called before load()")
        # onnx-asr's recognize() types sample_rate as a Literal of specific
        # values (8000/16000/etc.) since only those are valid; our public
        # API accepts a plain int (the caller-supplied rate), so this call
        # can't be statically verified against that Literal.
        result = self._model.recognize(audio, sample_rate=sample_rate)  # type: ignore[call-overload]
        return str(result)


def _preload_cudnn() -> None:
    """Preload cuDNN shared libraries so onnxruntime's CUDA provider can
    find them, since they live inside the nvidia-cudnn-cu12 pip package's
    site-packages directory rather than the system library path.

    Safe to call multiple times (loading an already-loaded .so is a no-op).
    """
    try:
        import nvidia.cudnn  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "nvidia-cudnn-cu12 not installed; CUDAExecutionProvider will likely fail"
        )
        return

    lib_dir = Path(nvidia.cudnn.__path__[0]) / "lib"
    for lib_path in sorted(glob.glob(str(lib_dir / "libcudnn*.so.9"))):
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
