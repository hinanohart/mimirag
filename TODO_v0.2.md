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
- **conformlock v0.2 PR (R17 demotion)**: the original `mcgate Bus`
  concept dissolved into the `conformlock`/`saegate`/`foldconsensus`
  trio at S0 self-overlap. Open a PR against `hinanohart/conformlock`
  to make its hard-gate API consume an arbitrary `Validator` Protocol
  (matching `mimirag.protocols.IndexBackend` style). Tracked here so
  the demotion is visible; the actual PR is filed in the `conformlock`
  repo, not here.

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

## Closed S8-critic findings (not in v0.1.0a1)
| finding | severity | landing |
|---|---|---|
| README claims Whisper as Apache-2.0 (it is MIT) | MAJOR | fixed in v0.1.0a1 patch chain (this commit) |
| `bench-dry-run` CI job referenced in README but absent | MAJOR | fixed in v0.1.0a1 patch chain (this commit) |
| `poc/s0_smoke.py` promised at S0 but missing | MAJOR | fixed in v0.1.0a1 patch chain (this commit) |
| `FakeMimiEncoder.pool` / `MimiEncoder.pool` lack 0-frame guard | MAJOR | fixed in v0.1.0a1 patch chain (this commit) |
| `.MIMIRAG-progress.json` / `.MIMIRAG-builder-lock.json` not at repo root | MAJOR | fixed (gitignored, but present for /compact recovery) |
| `tqdm` unused dependency | MINOR | fixed (removed) |
| diff-CI math drift between docstring and impl | MAJOR | fixed (paired bootstrap) |
