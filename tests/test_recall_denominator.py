"""Regression test for `recall_at_k`'s `min(k, |relevant|)` denominator.

The v0.1.0a1 release used `|relevant|` as the denominator, which
silently capped the score below 1.0 on a perfect ranking whenever
the corpus had more relevant docs than the cutoff `k` allowed the
ranker to return. v0.1.0a2 switched to the "achievable recall"
convention used by the same-file `ndcg_at_k`.
"""

from __future__ import annotations

from mimirag.bench import recall_at_k


def test_perfect_top_k_reaches_one_when_relevant_exceeds_k() -> None:
    # 5 docs are relevant, but the cutoff is 3; the best a ranker can
    # achieve is "all 3 retrieved items are relevant".
    rel = frozenset({"a", "b", "c", "d", "e"})
    hits = ["a", "b", "c", "x", "y"]
    assert recall_at_k(hits, rel, k=3) == 1.0


def test_partial_top_k_reports_fraction_of_min() -> None:
    rel = frozenset({"a", "b", "c", "d", "e"})
    hits = ["a", "x", "b", "y", "z"]
    # 2 of the 3 top-3 are relevant; achievable was 3 → 2/3
    assert abs(recall_at_k(hits, rel, k=3) - (2.0 / 3.0)) < 1e-9


def test_single_relevant_unchanged() -> None:
    # |relevant|=1, the old denominator and the new one agree.
    assert recall_at_k(["a", "b", "c"], frozenset({"a"}), k=3) == 1.0
    assert recall_at_k(["b", "c", "d"], frozenset({"a"}), k=3) == 0.0


def test_empty_relevant_returns_zero() -> None:
    assert recall_at_k(["a"], frozenset(), k=3) == 0.0


def test_k_larger_than_relevant_uses_relevant_count() -> None:
    # min(k, |relevant|) = |relevant| when k > |relevant|.
    rel = frozenset({"a", "b"})
    hits = ["a", "b", "c", "d", "e"]
    assert recall_at_k(hits, rel, k=5) == 1.0
