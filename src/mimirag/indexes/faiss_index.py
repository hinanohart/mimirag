"""FAISS-CPU inner-product index, fronted with a string-key map.

We keep keys as a parallel `list[str]` because FAISS works on int64
positions. The interface is intentionally small (add / search / len)
so swapping in sqlite-vec, chromadb, or a remote vector DB later is a
single-file change.
"""

from __future__ import annotations

import numpy as np

try:  # FAISS is required (it's a base dependency, not an extra)
    import faiss
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "faiss-cpu is required. `uv sync` should install it; "
        "if it failed, your platform may need a wheel from PyPI directly."
    ) from e


class FaissCpuIndex:
    """Inner-product FAISS index keyed by string IDs.

    Vectors are expected to be L2-normalised by the caller so IP ≈ cosine.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._keys: list[str] = []
        self._key_set: set[str] = set()

    def add(self, key: str, vec: np.ndarray) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty str")
        if key in self._key_set:
            raise ValueError(f"duplicate key: {key!r}")
        if vec.shape != (self.dim,):
            raise ValueError(f"vec shape {vec.shape} != index dim ({self.dim},)")
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32)
        self._index.add(vec.reshape(1, self.dim))
        self._keys.append(key)
        self._key_set.add(key)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if len(self._keys) == 0:
            return []
        if query.shape != (self.dim,):
            raise ValueError(f"query shape {query.shape} != index dim ({self.dim},)")
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        k_eff = min(k, len(self._keys))
        scores, idxs = self._index.search(query.reshape(1, self.dim), k_eff)
        out: list[tuple[str, float]] = []
        for pos, score in zip(idxs[0].tolist(), scores[0].tolist(), strict=True):
            if pos < 0:
                continue
            out.append((self._keys[pos], float(score)))
        return out

    def __len__(self) -> int:
        return len(self._keys)
