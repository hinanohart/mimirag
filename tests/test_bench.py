"""Tests for the bench harness + RESULTS rendering."""

from __future__ import annotations

import numpy as np

from mimirag.bench import (
    AxisStats,
    QueryItem,
    _bootstrap_ci,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    render_results_md,
    run_bench,
)
from mimirag.models import AudioChunk
from mimirag.pipeline import Pipeline


def _wave(seed: int, n: int = 24000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def test_recall_at_k_basic() -> None:
    assert recall_at_k(["a", "b", "c"], frozenset({"b"}), k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], frozenset({"d"}), k=3) == 0.0
    assert recall_at_k(["a", "b", "c"], frozenset({"c"}), k=2) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], frozenset({"a"})) == 1.0
    assert reciprocal_rank(["a", "b", "c"], frozenset({"b"})) == 0.5
    assert reciprocal_rank(["a", "b", "c"], frozenset({"z"})) == 0.0


def test_ndcg_at_k_perfect_is_one() -> None:
    assert ndcg_at_k(["a"], frozenset({"a"}), k=5) == 1.0


def test_ndcg_at_k_handles_empty_relevant() -> None:
    assert ndcg_at_k(["a", "b"], frozenset(), k=5) == 0.0


def test_bootstrap_ci_brackets_mean() -> None:
    values = [0.0, 0.5, 1.0, 0.5, 0.5]
    lo, hi = _bootstrap_ci(values, n_boot=500, seed=42)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi


def test_run_bench_returns_all_axes(fake_pipeline: Pipeline) -> None:
    queries: list[QueryItem] = []
    for i in range(6):
        wave = _wave(seed=200 + i)
        chunk = AudioChunk(
            id=f"d{i}",
            path=f"/synthetic/d{i}.wav",
            sample_rate=24000,
            n_samples=int(wave.size),
        )
        fake_pipeline.ingest(chunk, wave)
        queries.append(
            QueryItem(
                query_id=f"q{i}",
                waveform=wave,
                sample_rate=24000,
                relevant_doc_ids=frozenset({f"d{i}"}),
            )
        )
    stats = run_bench(fake_pipeline, queries, k=3, n_boot=200)
    assert {s.axis for s in stats} == {"text-only", "codec-only", "hybrid", "baseline"}
    for s in stats:
        assert s.n_queries == 6
        assert 0.0 <= s.recall_at_5 <= 1.0
        assert s.recall_at_5_ci[0] <= s.recall_at_5_ci[1]


def test_render_results_md_has_measured_tag() -> None:
    stats = [
        AxisStats(
            axis="hybrid",
            recall_at_5=0.5,
            recall_at_5_ci=(0.3, 0.7),
            mrr=0.4,
            mrr_ci=(0.2, 0.6),
            ndcg_at_5=0.45,
            ndcg_at_5_ci=(0.25, 0.65),
            latency_p50_ms=10.0,
            latency_p95_ms=15.0,
            n_queries=5,
        ),
        AxisStats(
            axis="baseline",
            recall_at_5=0.5,
            recall_at_5_ci=(0.3, 0.7),
            mrr=0.4,
            mrr_ci=(0.2, 0.6),
            ndcg_at_5=0.45,
            ndcg_at_5_ci=(0.25, 0.65),
            latency_p50_ms=5.0,
            latency_p95_ms=8.0,
            n_queries=5,
        ),
    ]
    md = render_results_md(
        stats,
        corpus_desc="tiny (5 docs)",
        backend_desc="fake",
        measured_on="2026-05-29",
    )
    assert "[MEASURED 2026-05-29]" in md
    assert "undetermined" in md.lower()
