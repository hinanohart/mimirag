"""Benchmark harness with bootstrap 95 % CI on Recall@k / MRR / nDCG.

Honesty rules:
- Every reported number gets a `[MEASURED YYYY-MM-DD]` tag and a
  bootstrap 95 % CI alongside the point estimate.
- If the CI of `(hybrid - baseline)` contains 0, the auto-rendered
  RESULTS.md prints "undetermined" rather than implying superiority.
- The CI is computed at the per-query level (resample queries with
  replacement, B=1000 by default).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np

from mimirag.models import Axis, RetrievalResult
from mimirag.pipeline import Pipeline

AXES: tuple[Axis, ...] = ("text-only", "codec-only", "hybrid", "baseline")


@dataclass(frozen=True)
class QueryItem:
    query_id: str
    waveform: np.ndarray
    sample_rate: int
    relevant_doc_ids: frozenset[str]
    query_text: str | None = None


def recall_at_k(hits: list[str], relevant: frozenset[str], k: int) -> float:
    if not relevant:
        return 0.0
    truncated = hits[:k]
    rel_hit = sum(1 for h in truncated if h in relevant)
    return rel_hit / float(len(relevant))


def reciprocal_rank(hits: list[str], relevant: frozenset[str]) -> float:
    for i, h in enumerate(hits):
        if h in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(hits: list[str], relevant: frozenset[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, h in enumerate(hits[:k]):
        if h in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


def _bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


@dataclass(frozen=True)
class AxisStats:
    axis: Axis
    recall_at_5: float
    recall_at_5_ci: tuple[float, float]
    mrr: float
    mrr_ci: tuple[float, float]
    ndcg_at_5: float
    ndcg_at_5_ci: tuple[float, float]
    latency_p50_ms: float
    latency_p95_ms: float
    n_queries: int


def run_bench(
    pipeline: Pipeline,
    queries: list[QueryItem],
    k: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
) -> list[AxisStats]:
    if not queries:
        raise ValueError("empty query set")
    out: list[AxisStats] = []
    for axis in AXES:
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        latencies: list[float] = []
        for q in queries:
            res: RetrievalResult = pipeline.query(
                waveform=q.waveform,
                sample_rate=q.sample_rate,
                axis=axis,
                k=k,
                query_id=q.query_id,
                query_text=q.query_text,
            )
            doc_ids = [h.doc_id for h in res.hits]
            recalls.append(recall_at_k(doc_ids, q.relevant_doc_ids, k))
            mrrs.append(reciprocal_rank(doc_ids, q.relevant_doc_ids))
            ndcgs.append(ndcg_at_k(doc_ids, q.relevant_doc_ids, k))
            latencies.append(res.latency_ms)
        lat_arr = np.asarray(latencies, dtype=np.float64)
        out.append(
            AxisStats(
                axis=axis,
                recall_at_5=float(np.mean(recalls)),
                recall_at_5_ci=_bootstrap_ci(recalls, n_boot=n_boot, seed=seed),
                mrr=float(np.mean(mrrs)),
                mrr_ci=_bootstrap_ci(mrrs, n_boot=n_boot, seed=seed + 1),
                ndcg_at_5=float(np.mean(ndcgs)),
                ndcg_at_5_ci=_bootstrap_ci(ndcgs, n_boot=n_boot, seed=seed + 2),
                latency_p50_ms=float(np.percentile(lat_arr, 50)),
                latency_p95_ms=float(np.percentile(lat_arr, 95)),
                n_queries=len(queries),
            )
        )
    return out


def render_results_md(
    stats: list[AxisStats],
    corpus_desc: str,
    backend_desc: str,
    measured_on: str | None = None,
) -> str:
    when = measured_on or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    hybrid = next((s for s in stats if s.axis == "hybrid"), None)
    baseline = next((s for s in stats if s.axis == "baseline"), None)
    superiority = ""
    if hybrid is not None and baseline is not None:
        diff_lo = hybrid.recall_at_5_ci[0] - baseline.recall_at_5_ci[1]
        diff_hi = hybrid.recall_at_5_ci[1] - baseline.recall_at_5_ci[0]
        if diff_lo > 0:
            superiority = (
                "**hybrid > baseline** on Recall@5 (95 % CI of the difference "
                "lies strictly above 0)."
            )
        elif diff_hi < 0:
            superiority = (
                "**hybrid < baseline** on Recall@5 (95 % CI of the difference "
                "lies strictly below 0)."
            )
        else:
            superiority = (
                "Recall@5 difference between hybrid and baseline is "
                "**undetermined** at the 95 % bootstrap CI level for this corpus."
            )
    is_fake = "fake" in backend_desc.lower()
    fake_banner = (
        "\n> ⚠️ **Backend: `fake` (deterministic CPU stub).** These numbers "
        "exercise the pipeline end-to-end on the CI matrix but are produced "
        "by hash-seeded encoders, **not** real Mimi/Whisper/BGE-M3 weights. "
        "Treat them as plumbing checks, not as a model-quality result. "
        "Live-backend numbers will appear in `v0.1.1`.\n"
        if is_fake
        else ""
    )
    lines = [
        "# mimirag bench RESULTS",
        "",
        f"[MEASURED {when}] [BACKEND {backend_desc}]",
        fake_banner,
        f"**Corpus**: {corpus_desc}",
        "",
        (
            "| axis | Recall@5 (95% CI) | MRR (95% CI) | nDCG@5 (95% CI) "
            "| latency p50 / p95 (ms) | n queries |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for s in stats:
        lines.append(
            f"| `{s.axis}` "
            f"| {s.recall_at_5:.3f} [{s.recall_at_5_ci[0]:.3f}, {s.recall_at_5_ci[1]:.3f}] "
            f"| {s.mrr:.3f} [{s.mrr_ci[0]:.3f}, {s.mrr_ci[1]:.3f}] "
            f"| {s.ndcg_at_5:.3f} [{s.ndcg_at_5_ci[0]:.3f}, {s.ndcg_at_5_ci[1]:.3f}] "
            f"| {s.latency_p50_ms:.1f} / {s.latency_p95_ms:.1f} "
            f"| {s.n_queries} |"
        )
    if superiority:
        lines.extend(["", superiority])
    lines.extend(
        [
            "",
            "## Honest-marketing notes",
            "",
            "- Numbers above are point estimates with bootstrap 95 % percentile",
            "  CIs over query-level resamples (B=1000).",
            "- The `baseline` axis is computationally identical to `text-only`",
            "  in this implementation; it is reported separately as a",
            "  rhetorical anchor for ablation tables.",
            "- These numbers were produced on the bundled CORPUS only. They",
            "  are **not** a generalisation claim. Independent reproduction",
            "  on a different corpus is encouraged.",
        ]
    )
    return "\n".join(lines) + "\n"
