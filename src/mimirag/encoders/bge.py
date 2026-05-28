"""Real BAAI BGE-M3 text embedder (MIT, weights downloaded at runtime).

Lazy-imported; requires `mimirag[text]`.
"""

from __future__ import annotations

import numpy as np


class BGEEncoder:
    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        device: str = "cpu",
        name: str = "bge-m3",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "BGEEncoder requires the `text` extra: "
                "`uv sync --extra text` or `pip install 'mimirag[text]'`"
            ) from e
        self.model_id = model_id
        self.device = device
        self.name = name
        self._model = SentenceTransformer(model_id, device=device)
        # Force a single dummy call to materialize self.dim
        sample = self._model.encode("x", normalize_embeddings=True)
        self.dim = int(np.asarray(sample).shape[-1])

    def encode_text(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError(f"expected str, got {type(text).__name__}")
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)
