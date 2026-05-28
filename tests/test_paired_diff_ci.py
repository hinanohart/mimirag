"""Tests for the paired-bootstrap difference CI used by RESULTS.md."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mimirag.bench import (
    AxisStats,
    _paired_bootstrap_diff_ci,
    render_results_md,
)


def test_paired_diff_ci_identical_inputs_brackets_zero() -> None:
    vals = [0.0, 0.5, 1.0, 0.5, 0.5]
    lo, hi = _paired_bootstrap_diff_ci(vals, vals, n_boot=500, seed=7)
    assert lo == 0.0 and hi == 0.0


def test_paired_diff_ci_a_dominates_b() -> None:
    a = [0.9] * 20
    b = [0.1] * 20
    lo, hi = _paired_bootstrap_diff_ci(a, b, n_boot=500, seed=7)
    assert lo > 0 and hi > 0
    assert math.isclose((lo + hi) / 2, 0.8, abs_tol=0.05)


def test_paired_diff_ci_length_mismatch_raises() -> None:

    with pytest.raises(ValueError, match="equal length"):
        _paired_bootstrap_diff_ci([1.0, 2.0], [1.0], n_boot=10)


def test_paired_diff_ci_empty_returns_nan() -> None:
    lo, hi = _paired_bootstrap_diff_ci([], [], n_boot=10)
    assert math.isnan(lo) and math.isnan(hi)


def _stats(axis: str, recalls: list[float]) -> AxisStats:
    arr = np.asarray(recalls, dtype=np.float64)
    return AxisStats(
        axis=axis,  # type: ignore[arg-type]
        recall_at_5=float(arr.mean()),
        recall_at_5_ci=(float(arr.mean() - 0.1), float(arr.mean() + 0.1)),
        mrr=float(arr.mean()),
        mrr_ci=(0.0, 1.0),
        ndcg_at_5=float(arr.mean()),
        ndcg_at_5_ci=(0.0, 1.0),
        latency_p50_ms=1.0,
        latency_p95_ms=2.0,
        n_queries=len(recalls),
        per_query_recall=tuple(recalls),
    )


def test_render_uses_paired_diff_ci_and_reports_dominance() -> None:
    stats = [
        _stats("hybrid", [1.0] * 10),
        _stats("baseline", [0.0] * 10),
    ]
    md = render_results_md(stats, corpus_desc="x", backend_desc="fake", measured_on="2026-05-29")
    assert "hybrid outperforms baseline" in md
    assert "undetermined" not in md.lower()


def test_render_reports_undetermined_for_identical_recalls() -> None:
    stats = [
        _stats("hybrid", [0.5] * 10),
        _stats("baseline", [0.5] * 10),
    ]
    md = render_results_md(stats, corpus_desc="x", backend_desc="fake", measured_on="2026-05-29")
    assert "undetermined" in md.lower()
