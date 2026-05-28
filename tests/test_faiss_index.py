"""Tests for FaissCpuIndex."""

from __future__ import annotations

import numpy as np
import pytest

from mimirag.indexes.faiss_index import FaissCpuIndex


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def test_faiss_add_then_search() -> None:
    idx = FaissCpuIndex(dim=4)
    v1 = _norm(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    v2 = _norm(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    idx.add("a", v1)
    idx.add("b", v2)
    assert len(idx) == 2
    hits = idx.search(v1, k=2)
    assert hits[0][0] == "a"
    assert hits[0][1] > hits[1][1]


def test_faiss_duplicate_key_rejected() -> None:
    idx = FaissCpuIndex(dim=4)
    v = np.zeros(4, dtype=np.float32)
    idx.add("a", v)
    with pytest.raises(ValueError, match="duplicate"):
        idx.add("a", v)


def test_faiss_wrong_dim_rejected() -> None:
    idx = FaissCpuIndex(dim=4)
    with pytest.raises(ValueError, match="dim"):
        idx.add("a", np.zeros(8, dtype=np.float32))


def test_faiss_empty_search_returns_empty() -> None:
    idx = FaissCpuIndex(dim=4)
    out = idx.search(np.zeros(4, dtype=np.float32), k=3)
    assert out == []


def test_faiss_k_truncated_to_available() -> None:
    idx = FaissCpuIndex(dim=4)
    idx.add("a", _norm(np.array([1, 0, 0, 0], dtype=np.float32)))
    out = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), k=10)
    assert len(out) == 1


def test_faiss_k_must_be_positive() -> None:
    idx = FaissCpuIndex(dim=4)
    with pytest.raises(ValueError):
        idx.search(np.zeros(4, dtype=np.float32), k=0)


def test_faiss_empty_key_rejected() -> None:
    idx = FaissCpuIndex(dim=4)
    with pytest.raises(ValueError):
        idx.add("", np.zeros(4, dtype=np.float32))


def test_faiss_dim_must_be_positive() -> None:
    with pytest.raises(ValueError):
        FaissCpuIndex(dim=0)
