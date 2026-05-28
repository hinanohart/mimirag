# mimirag v0.1.0a1 post-release audit

This is the consolidated audit log for the pre-alpha `v0.1.0a1`
release. Three independent agents were spawned with read-only access:

- **Agent A** — code correctness + architecture adherence
- **Agent B** — GitHub state thoroughness (CI / branch protection / PRs / alerts)
- **Agent C** — completeness vs the bundled architecture + bootstrap protocol

A fourth `critic` agent merged the findings, deduped them, and ordered
the fix list. All seven actionable findings landed in `v0.1.0a2`.
Verdict: **CLOSED — no CRITICAL findings, all MAJOR / HIGH landed,
release pair `v0.1.0a1 + v0.1.0a2` both live (no yank)**.

## Audit verdict matrix

| Finding | Severity | Component | Landing |
|---|---|---|---|
| GitHub shows the project's license as "Other" rather than `apache-2.0` because the LICENSE file diverged from the SPDX template | MAJOR | release hygiene | v0.1.0a2 (canonical SPDX text restored) |
| `recall_at_k` denominator silently capped the metric below 1.0 when the corpus had more relevant docs than the cutoff | HIGH | `bench.py` | v0.1.0a2 (denominator switched to `min(k, |relevant|)` + regression test `tests/test_recall_denominator.py`) |
| `MimiEncoder.pool` did not assert `max(token id) < vocab_size`; a future Mimi config drift would crash inside `bincount` | HIGH | `encoders/mimi.py` | v0.1.0a2 (explicit assertion with both numbers in the error message + `tests/test_pool_guards.py`) |
| README opening read "audio-native RAG" but the package ships no generator | MEDIUM | `README.md` framing | v0.1.0a2 (re-framed as "audio-native retriever, pair with any LLM generator") |
| README CLAIM table did not hedge the novelty claim against BEST-STD class prior art for speech-token retrieval | MEDIUM | `README.md` CLAIM table | v0.1.0a2 (scope-limited the "first" claim to the Mimi codec adaptation) |
| Axis table used "strong baseline" / "modality-pure" weasel phrasing | MINOR | `README.md` axis table | v0.1.0a2 (neutral prose; explicit `baseline == text-only` callout) |
| The in-flight Dependabot version-update PR #1 contradicted the `limit: 0` policy | MINOR | `dependabot.yml` | v0.1.0a2 (PR #1 closed with explanatory comment) |

## Closed earlier (S8 internal critic, two rounds)

These nine findings closed in `v0.1.0a1` itself before the first push;
they are listed here for completeness because the same audit trail
covers them in `TODO_v0.2.md` (lines 50-61).

| Finding | Severity | Landing |
|---|---|---|
| README claims Whisper as Apache-2.0 (it is MIT) | MAJOR | `v0.1.0a1` patch chain |
| `bench-dry-run` CI job referenced in README but absent | MAJOR | `v0.1.0a1` patch chain |
| `poc/s0_smoke.py` promised at S0 but missing | MAJOR | `v0.1.0a1` patch chain |
| `FakeMimiEncoder.pool` / `MimiEncoder.pool` lacked 0-frame guard | MAJOR | `v0.1.0a1` patch chain |
| `.MIMIRAG-progress.json` / `.MIMIRAG-builder-lock.json` not at repo root | MAJOR | `v0.1.0a1` patch chain |
| `tqdm` unused dependency | MINOR | `v0.1.0a1` (removed) |
| diff-CI math drift between docstring and impl | MAJOR | `v0.1.0a1` (paired bootstrap) |
| pre-commit hook self-bombs the repo's own tracked content | MAJOR | `v0.1.0a1` (diff-only scan + allow-list) |
| `TODO_v0.2` phrasing implied the conformlock PR was already filed | MINOR | `v0.1.0a1` (rewritten) |

## Confidence

- All findings are reproducible from a clean clone at the listed commit
  refs.
- `v0.1.0a1` HEAD: tag `v0.1.0a1` (commit `f960152` was the `v0.1.0a2`
  release commit).
- `v0.1.0a2` audit cycle verified by an independent post-/compact
  re-audit (see `AUDIT_v0.1.0a3.md`).
- Coverage of the audit surface: `src/mimirag/` (15 files,
  ~1.2 kLOC), `tests/` (95 tests at the time of `v0.1.0a2`), CI
  matrix (11 required status checks), GitHub repo settings.

## Not in scope

- **Live Mimi GPU inference path.** Deferred to `v0.1.1` per
  architecture; the `mimi` extra works locally but the default CI
  matrix exercises `FakeMimiEncoder` only.
- **PyPI publication.** Deferred to `v0.1.1`.
- **Performance superiority claim.** Not claimed; `bench/RESULTS.md`
  auto-prints `undetermined` when the paired bootstrap 95 % CI of
  (hybrid minus baseline) contains 0.

## How to reproduce this audit

```bash
git clone https://github.com/hinanohart/mimirag
cd mimirag && git checkout v0.1.0a2
uv sync --extra dev
uv run pytest -q                              # 95 pass, 91.47% coverage
uv run mypy src                               # strict, no issues
uv run ruff check                             # no issues
uv run python scripts/honest_marketing_check.py
uv run pip-licenses --format=plain --with-license-file --fail-on='GPL;LGPL;AGPL'
```

GitHub-side spot checks:

```bash
gh repo view hinanohart/mimirag --json licenseInfo,repositoryTopics
gh api repos/hinanohart/mimirag/branches/main/protection --jq '.required_status_checks.contexts'
gh run list -R hinanohart/mimirag --workflow=ci.yml --limit 5
```
