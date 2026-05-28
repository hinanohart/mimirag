"""Real Kyutai Mimi backend (CC-BY-4.0 weights, downloaded at runtime).

This module is intentionally lazy-imported from `encoders/__init__.py`
so the `mimirag[mimi]` extra is only required on the code path that
needs it. The CPU CI matrix exercises `FakeMimiEncoder` instead and
this module is `@pytest.mark.live`.

NOTE: pre-alpha. Live GPU bench is deferred to v0.1.1. This wrapper
exists so end-users can opt-in locally; it is not exercised by the
default CI matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mimirag.models import TokenStream

if TYPE_CHECKING:  # pragma: no cover
    pass


class MimiEncoder:
    """Wraps `transformers.MimiModel` for inference-only use."""

    def __init__(
        self,
        model_id: str = "kyutai/mimi",
        device: str = "cpu",
        name: str = "mimi",
    ) -> None:
        try:
            import torch  # noqa: F401, PLC0415
            from transformers import AutoFeatureExtractor, MimiModel  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "MimiEncoder requires the `mimi` extra: "
                "`uv sync --extra mimi` or `pip install 'mimirag[mimi]'`"
            ) from e

        self.model_id = model_id
        self.device = device
        self.name = name
        self._model = MimiModel.from_pretrained(model_id).to(device).eval()
        self._fe = AutoFeatureExtractor.from_pretrained(model_id)
        cfg = self._model.config
        self.sample_hz = float(getattr(cfg, "frame_rate", 12.5))
        self.n_codebooks = int(getattr(cfg, "num_quantizers", 8))
        self.vocab_size = int(getattr(cfg, "codebook_size", 2048))

    def encode(
        self, waveform: np.ndarray, sample_rate: int, source_id: str
    ) -> tuple[TokenStream, np.ndarray]:
        import torch  # noqa: PLC0415

        if waveform.ndim != 1:
            raise ValueError(f"expected 1-D waveform, got shape {waveform.shape}")
        if waveform.size == 0:
            raise ValueError("waveform is empty")
        target_sr = int(self._fe.sampling_rate)
        if sample_rate != target_sr:
            # Caller is responsible for resampling; we refuse rather than
            # silently downgrading quality.
            raise ValueError(
                f"MimiEncoder expects {target_sr} Hz, got {sample_rate}. "
                "Resample upstream (e.g. with soundfile + librosa)."
            )
        inputs = self._fe(raw_audio=waveform, sampling_rate=target_sr, return_tensors="pt")
        with torch.no_grad():
            enc = self._model.encode(inputs["input_values"].to(self.device))
        codes = enc.audio_codes.detach().cpu().numpy()
        # transformers returns (batch, n_codebooks, n_frames); squeeze batch
        if codes.ndim == 3:
            codes = codes[0]
        tokens = codes.astype(np.int32)
        stream = TokenStream(
            source_id=source_id,
            sample_hz=self.sample_hz,
            n_codebooks=int(tokens.shape[0]),
            n_frames=int(tokens.shape[1]),
            encoder=self.name,
            meta={"backend": "mimi", "model_id": self.model_id},
        )
        return stream, tokens

    def pool(self, stream: TokenStream, tokens: np.ndarray) -> np.ndarray:
        if tokens.ndim != 2 or tokens.shape[0] != self.n_codebooks:
            raise ValueError(f"expected ({self.n_codebooks}, T) tokens, got shape {tokens.shape}")
        if tokens.shape[1] == 0:
            raise ValueError("token stream has 0 frames")
        out = np.zeros((self.n_codebooks * self.vocab_size,), dtype=np.float32)
        for ci in range(self.n_codebooks):
            counts = np.bincount(tokens[ci], minlength=self.vocab_size).astype(np.float32)
            out[ci * self.vocab_size : (ci + 1) * self.vocab_size] = counts / float(tokens.shape[1])
        norm = float(np.linalg.norm(out))
        if norm > 0.0:
            out = out / norm
        return out

    @property
    def pooled_dim(self) -> int:
        return self.n_codebooks * self.vocab_size
