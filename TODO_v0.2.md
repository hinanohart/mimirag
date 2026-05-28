# mimirag v0.2 backlog

Deferred from v0.1.0a1 per architecture (`project_mimirag_architecture_2026-05-29.md`)
and S8 critic findings. Items here are intentional non-goals for the
v0.1 line; ship-blocking issues live in `.MIMIRAG-progress.json`
under `open_blockers`.

## Architecture follow-ups
- **Persisted FAISS index** (S8-A MAJOR-3): add `faiss.write_index` +
  sidecar JSON for `_keys`, then teach both `ingest` and `query` a real
  `--index` flag. v0.1 keeps both indexes in-process and re-ingests on
  every CLI invocation; the README quickstart now says so explicitly.
- **Paired-bootstrap difference CI**: the v0.1 hybrid-vs-baseline
  difference CI is implemented as a paired bootstrap over per-query
  resamples; once we have more than one ablation pair we want a single
  helper (`_paired_diff_ci(metric_per_query_a, metric_per_query_b)`)
  reused across pairs.
- **`sqlite-vec` codec index option** (arch §2/§3 originally listed it):
  pure-Python FAISS-CPU is enough for v0.1; revisit once the public
  corpus path lands.

## Cross-repo follow-up
- **conformlock follow-up PR (status: not yet filed)**: the original
  `mcgate Bus` concept dissolved into the `conformlock` / `saegate` /
  `foldconsensus` trio at S0 self-overlap. The follow-up work is to
  open a PR against `hinanohart/conformlock` that makes its hard-gate
  API consume an arbitrary `Validator` Protocol (matching
  `mimirag.protocols.IndexBackend` style). This is the only
  cross-repo task carried out of v0.1; track here until it lands.

## Releases
- **PyPI publication** (`v0.1.1`): blocked on post-release audit (S11)
  finishing without CRITICAL. Until then we ship by git URL only.
- **Live Mimi GPU bench** (`v0.1.1`): the `mimi` extra works today but
  isn't exercised in the default CI matrix. Add a `@pytest.mark.live`
  GPU job that pulls `kyutai/mimi` weights and confirms the round-trip
  on a single utterance.
- **PolyglotMimi (Omnilingual ASR 1600 languages)**: reassess at v0.2
  per architecture §1. License review for Omnilingual ASR weights
  needed before any code lands.

## Engineering hygiene
- **Rust hot-path**: deferred to "v0.2 以降 bottleneck 実測後". Do not
  add `Cargo.toml` until the v0.1.1 bench shows a concrete bottleneck.
- **`mimirag-core` extraction**: only when a second consumer exists
  (YAGNI per protocols.py:1-5).

## Closed audit findings

### Pre-release internal critic (S8 round 1, S8 round 2) — closed in v0.1.0a1
| finding | severity | landing |
|---|---|---|
| README claims Whisper as Apache-2.0 (it is MIT) | MAJOR | v0.1.0a1 patch chain |
| `bench-dry-run` CI job referenced in README but absent | MAJOR | v0.1.0a1 patch chain |
| `poc/s0_smoke.py` promised at S0 but missing | MAJOR | v0.1.0a1 patch chain |
| `FakeMimiEncoder.pool` / `MimiEncoder.pool` lack 0-frame guard | MAJOR | v0.1.0a1 patch chain |
| `.MIMIRAG-progress.json` / `.MIMIRAG-builder-lock.json` not at repo root | MAJOR | v0.1.0a1 patch chain (gitignored, but present for /compact recovery) |
| `tqdm` unused dependency | MINOR | v0.1.0a1 (removed) |
| diff-CI math drift between docstring and impl | MAJOR | v0.1.0a1 (paired bootstrap) |
| pre-commit hook self-bombs the repo's own tracked content | MAJOR | v0.1.0a1 (diff-only scan + allow-list) |
| TODO_v0.2 phrasing implied conformlock PR was already filed | MINOR | v0.1.0a1 (rewritten) |

### Post-release independent audit (S11) — closed in v0.1.0a2
| finding | severity | landing |
|---|---|---|
| GitHub shows the project's license as "Other" (not Apache-2.0) because the LICENSE file diverges from the SPDX template | MAJOR | v0.1.0a2 (canonical text restored) |
| `recall_at_k` denominator silently caps below 1.0 when `|relevant| > k` | HIGH | v0.1.0a2 (denominator switched to `min(k, |relevant|)` + regression test) |
| `MimiEncoder.pool` does not assert `max(token id) < vocab_size`; a future Mimi config drift would crash inside `bincount` | HIGH | v0.1.0a2 (explicit assertion with both numbers in the error message) |
| README opening reads "audio-native RAG" but the package ships no generator | MEDIUM | v0.1.0a2 (re-framed as "audio-native retriever, pair with any LLM generator") |
| README CLAIM table did not hedge the novelty claim against BEST-STD class prior art for speech-token retrieval | MEDIUM | v0.1.0a2 (scope-limited the "first" claim to the Mimi adaptation) |
| Axis table used "strong baseline" / "modality-pure" weasel phrasing | MINOR | v0.1.0a2 (neutral prose; explicit `baseline == text-only` callout) |
| In-flight Dependabot version-update PR contradicted the `limit: 0` policy | MINOR | v0.1.0a2 (closed PR #1 with explanatory comment) |

## Open Dependabot security alerts

### Closed in `v0.1.0a3`
- **pytest** (GHSA-6w46-j5rx-g56g / CVE-2025-71176) — tmpdir handling.
  Dev-only dependency; PR #2 (pytest 8.4.2 → 9.0.3) merged via
  `--admin` after CI matrix went 11/11 green. Alert auto-closed.

### Tolerable-risk, deferred to `v0.1.1` live-backend cycle
- **transformers** (GHSA-69w3-r845-3855 / CVE-2026-1839) — `Trainer`
  arbitrary code execution. The advisory is reachable only via
  `_load_rng_state()` inside the `Trainer` class. `mimirag` does NOT
  import `transformers.Trainer` — see `src/mimirag/encoders/mimi.py`,
  which only imports `MimiModel` and `AutoFeatureExtractor`. The
  default CI matrix runs `FakeMimiEncoder` and never imports
  `transformers` at all. `dependabot.yml` now lists `transformers`
  under `ignore:` so we don't auto-bump into the v5.x major API
  break before the live-backend bench cycle has a chance to validate
  it. The alert is intentionally left in the OPEN state on the
  GitHub Security tab as a transparent record of the deferred
  exposure rather than being dismissed.

## v0.1.0a3 post-/compact re-audit backlog (cosmetic, deferred)

Surfaced by an independent three-agent re-audit after the
`/compact` boundary. None were ship-blockers; all eight have
sub-MAJOR severity and are queued for the next refactor pass.

- **bench bootstrap vectorisation** — `bench.py:_bootstrap_ci` /
  `_paired_bootstrap_diff_ci` use a Python `for` loop. A 2-line
  `rng.integers(0, n, size=(n_boot, n))` rewrite is faster without
  changing the statistic.
- **latency-measurement duplication** — `pipeline.py:_query_text` /
  `_query_codec` both measure `time.perf_counter()` internally, but
  the `hybrid` axis discards those numbers and remeasures around the
  fused call. Move latency to the outer `query()` only.
- **CLI `cast(Axis, axis)` instead of `# type: ignore[arg-type]`** —
  `cli.py:96` uses `# type: ignore` to bridge `click.Choice → str` →
  `Axis Literal`. `cast(Axis, axis)` is the canonical fix.
- **encoders registry triplication** — `encoders/__init__.py` has
  three near-identical `get_encoder` / `get_text_encoder` / `get_asr`
  functions with the same lazy-import shape. Collapse into a single
  `dict`-backed registry helper.
- **`scripts/honest_marketing_check.py` "perfect" regex** — the
  word-boundary `\bperfect\b` regex flags neutral technical phrasing
  ("perfect ranking", `test_*_perfect_is_one`). Either narrow to
  phrase-level ("perfect score", "perfect retrieval") or drop.
- **`src/mimirag/data.py` vs `indexes/faiss_index.py` import style** —
  `data.py` does a bare `import soundfile as sf`; `faiss_index.py`
  wraps `import faiss` in a `try/except` with a helpful re-raise.
  Both are base deps; pick one style.
- **bench loop variable shadowing** — `bench.py:107` uses `for k in
  range(n_boot)` which shadows the `k=cutoff` parameter used
  throughout the file. Rename to `for b in range(n_boot)`.
- **RRF tie-break documentation** — `fusion.py:30` returns ties in
  `dict` insertion order (first source wins). The behaviour is
  correct and tested but should be 1-lined in the `RRFFuser`
  docstring so reviewers don't have to derive it.

## Architecture follow-ups (carried from earlier audit cycle)

- **Protocol-level `accepted_sample_rate` and `pooled_dim`
  contracts** (landed in `v0.1.0a3`). Both attributes are now
  Protocol-declared and pinned by `tests/test_protocols_sample_rate.py`.
- **`pool_token_stream` public-API trap** — the helper exposed in
  `models.py` derives `vocab` from `tokens.max()` and therefore drifts
  in output dimension across calls. `v0.1.0a3` removed it from
  `encoders/fake.py:__all__`; a fixed-vocab `pool_token_stream_fixed`
  helper that takes `(n_codebooks, vocab_size)` explicitly is queued
  for `v0.1.1`.
- **first-interaction workflow** — the bootstrap protocol mentions an
  actor-guard'd `first-interaction` workflow. We intentionally do not
  ship one at `v0.1.0a3`: the repo is in pre-alpha pre-contributor
  state and the workflow's only function (greet on first PR / issue)
  would only fire for Dependabot bots. Will be reconsidered when we
  enable Discussions or open up contributor PRs.
