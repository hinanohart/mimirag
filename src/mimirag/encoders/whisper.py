"""Real OpenAI Whisper ASR baseline (MIT, weights downloaded at runtime).

Lazy-imported; requires `mimirag[asr]`.
"""

from __future__ import annotations

import numpy as np


class WhisperASR:
    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        name: str = "whisper",
    ) -> None:
        try:
            import whisper  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "WhisperASR requires the `asr` extra: "
                "`uv sync --extra asr` or `pip install 'mimirag[asr]'`"
            ) from e
        self.model_size = model_size
        self.device = device
        self.name = name
        self._model = whisper.load_model(model_size, device=device)
        self.accepted_sample_rate: int | None = 16000

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> str:
        if waveform.ndim != 1:
            raise ValueError(f"expected 1-D waveform, got shape {waveform.shape}")
        if waveform.size == 0:
            return ""
        if sample_rate != 16000:
            raise ValueError(f"WhisperASR expects 16000 Hz; got {sample_rate}. Resample upstream.")
        result = self._model.transcribe(waveform.astype(np.float32), fp16=False)
        return str(result.get("text", "")).strip()
