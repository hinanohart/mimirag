"""Reciprocal-Rank-Fusion (RRF) over per-axis hit lists.

RRF: score(d) = sum over rankers r of 1 / (k_rrf + rank_r(d)).
`k_rrf=60` is the value from Cormack et al. (2009) and is the de-facto
default in modern hybrid-search literature.
"""

from __future__ import annotations

from mimirag.models import Axis, Hit


class RRFFuser:
    name = "rrf"

    def __init__(self, k_rrf: int = 60) -> None:
        if k_rrf <= 0:
            raise ValueError("k_rrf must be > 0")
        self.k_rrf = k_rrf

    def fuse(self, hits_per_source: list[list[Hit]], k: int) -> list[Hit]:
        if k <= 0:
            raise ValueError("k must be > 0")
        if not hits_per_source:
            return []
        scores: dict[str, float] = {}
        for source in hits_per_source:
            for hit in source:
                scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (self.k_rrf + hit.rank)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        # Fused hits have axis="hybrid" by convention.
        axis_out: Axis = "hybrid"
        return [
            Hit(doc_id=doc_id, score=score, axis=axis_out, rank=rank)
            for rank, (doc_id, score) in enumerate(ranked)
        ]
