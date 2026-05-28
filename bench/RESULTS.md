# mimirag bench RESULTS

[MEASURED 2026-05-28] [BACKEND fake]

> ⚠️ **Backend: `fake` (deterministic CPU stub).** These numbers exercise the pipeline end-to-end on the CI matrix but are produced by hash-seeded encoders, **not** real Mimi/Whisper/BGE-M3 weights. Treat them as plumbing checks, not as a model-quality result. Live-backend numbers will appear in `v0.1.1`.

**Corpus**: tests/data/tiny (8 docs, self-retrieval)

| axis | Recall@5 (95% CI) | MRR (95% CI) | nDCG@5 (95% CI) | latency p50 / p95 (ms) | n queries |
|---|---|---|---|---|---|
| `text-only` | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.2 / 0.4 | 8 |
| `codec-only` | 0.625 [0.250, 0.875] | 0.448 [0.156, 0.781] | 0.491 [0.179, 0.804] | 0.4 / 0.6 | 8 |
| `hybrid` | 1.000 [1.000, 1.000] | 0.667 [0.422, 0.906] | 0.749 [0.561, 0.929] | 0.7 / 1.0 | 8 |
| `baseline` | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.1 / 0.2 | 8 |

On Recall@5, the paired-bootstrap 95 % CI of (hybrid - baseline) is [0.000, 0.000], containing 0: the difference between hybrid and baseline is **undetermined** on this corpus.

## Honest-marketing notes

- Numbers above are point estimates with bootstrap 95 % percentile
  CIs over query-level resamples (B=1000).
- The `baseline` axis is computationally identical to `text-only`
  in this implementation; it is reported separately as a
  rhetorical anchor for ablation tables.
- These numbers were produced on the bundled CORPUS only. They
  are **not** a generalisation claim. Independent reproduction
  on a different corpus is encouraged.
