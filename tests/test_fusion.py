"""Tests for the RRF fuser."""

from __future__ import annotations

import pytest

from mimirag.fusion import RRFFuser
from mimirag.models import Hit


def _h(doc_id: str, axis: str, rank: int) -> Hit:
    return Hit(doc_id=doc_id, score=0.0, axis=axis, rank=rank)  # type: ignore[arg-type]


def test_rrf_basic_fusion() -> None:
    rrf = RRFFuser(k_rrf=60)
    src1 = [_h("a", "text-only", 0), _h("b", "text-only", 1)]
    src2 = [_h("b", "codec-only", 0), _h("c", "codec-only", 1)]
    fused = rrf.fuse([src1, src2], k=3)
    assert [h.doc_id for h in fused] == ["b", "a", "c"]
    assert all(h.axis == "hybrid" for h in fused)
    assert [h.rank for h in fused] == [0, 1, 2]


def test_rrf_truncates_to_k() -> None:
    rrf = RRFFuser()
    src = [_h(f"d{i}", "text-only", i) for i in range(10)]
    fused = rrf.fuse([src], k=3)
    assert len(fused) == 3


def test_rrf_empty_input() -> None:
    rrf = RRFFuser()
    assert rrf.fuse([], k=5) == []


def test_rrf_rejects_bad_k_rrf() -> None:
    with pytest.raises(ValueError):
        RRFFuser(k_rrf=0)


def test_rrf_rejects_bad_k() -> None:
    rrf = RRFFuser()
    with pytest.raises(ValueError):
        rrf.fuse([[]], k=0)


def test_rrf_score_monotone_in_rank() -> None:
    rrf = RRFFuser(k_rrf=60)
    src = [_h("a", "text-only", 0), _h("b", "text-only", 5)]
    fused = rrf.fuse([src], k=2)
    # a (rank 0) should have a higher fused score than b (rank 5)
    assert fused[0].doc_id == "a"
    assert fused[0].score > fused[1].score
