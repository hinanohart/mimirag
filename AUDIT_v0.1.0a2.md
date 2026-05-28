# mimirag v0.1.0a3 post-/compact re-audit

This is the consolidated audit log for the second post-release cycle,
triggered after a Claude Code `/compact` boundary, to verify that the
`v0.1.0a2` ship survived the context summarisation intact.

Three independent agents and one critic were spawned with read-only
access:

- **Agent A** — `omc-code-reviewer`: code correctness + architecture
  adherence + refinement (Protocol invariants, math soundness,
  duplication, dead code, naming).
- **Agent B** — `architect`: GitHub state thoroughness (CI runs,
  branch protection, PRs, Dependabot alerts, check-suite noise,
  license API recognition).
- **Agent C** — `verifier`: completeness vs. the bundled
  architecture + bootstrap protocol (S0–S11 artefacts, S8 internal
  critic landings, S11 post-release audit landings, M1/M2/M3 monitor
  findings, intentional omissions).
- **Critic** — `critic`: merged findings, ordered the fix list,
  judged severity, picked the release vehicle.

Verdict: **CLOSED — CRITICAL 0, HIGH 2, MAJOR 2, MEDIUM 3, LOW 6,
NIT 3**. All HIGH / MAJOR / MEDIUM landed in `v0.1.0a3`. LOW / NIT
were deferred to a single dated backlog entry in
`TODO_v0.2.md` (with each finding listed) to keep the diff small and
the regression surface tight.

## Audit verdict matrix

| Finding | Severity | Component | Landing |
|---|---|---|---|
| `Encoder.encode` / `ASRBackend.transcribe` Protocols did not advertise an `accepted_sample_rate`, but `MimiEncoder` pins 24000 Hz and `WhisperASR` pins 16000 Hz. A pipeline could pass a mismatched waveform that the fake backends accept and the real backends silently reject at the wrong layer. | HIGH | `protocols.py`, `encoders/mimi.py`, `encoders/whisper.py`, `encoders/fake.py` | `v0.1.0a3` — `accepted_sample_rate: int | None` added to both Protocols (`None` = any positive Hz, `int` = pinned Hz) + pinned by `tests/test_protocols_sample_rate.py` + 1-paragraph README CAVEATS callout in the Quickstart |
| `AUDIT_v0.1.0a1.md` was missing from the repo even though the bootstrap protocol's S11 completion criterion required it. | HIGH (doc) | repo root | `v0.1.0a3` — `AUDIT_v0.1.0a1.md` committed with the original 7 findings + the earlier S8 9 findings as a single transcribed audit trail |
| Dependabot security-update workflow kept failing because `pyproject.toml` pinned `transformers<5.0` but the patched version is `5.0.0rc3` — `uv lock` could not resolve. Result: a permanent red `Dependabot Updates` workflow on every push. | MAJOR | `.github/dependabot.yml` | `v0.1.0a3` — `transformers` added to the `ignore:` list with an inline rationale; the corresponding alert is documented in `TODO_v0.2.md` as tolerable-risk pre-alpha exposure (Trainer not imported by mimirag) |
| PR #2 (Dependabot pytest 8.4.2 → 9.0.3, security GHSA-6w46-j5rx-g56g) was OPEN with 11/11 SUCCESS but blocked by the original `pyproject.toml` pin `pytest<9.0`. The same PR had been manually closed once as #1 and Dependabot kept re-opening it. | MAJOR | `pyproject.toml`, `uv.lock` | `v0.1.0a3` — PR #2 merged via `--admin` after Dependabot rewrote both `pyproject.toml` (`pytest>=9.0.3,<10.0`) and `uv.lock`; security alert auto-closed by GitHub |
| `cli.py:_build_pipeline` relied on `getattr(enc, "pooled_dim", None)` because the `Encoder` Protocol did not declare `pooled_dim`. A downstream-implemented backend without the property would still type-check but die at CLI runtime. | MEDIUM | `protocols.py`, `cli.py` | `v0.1.0a3` — `pooled_dim` added to the Protocol as `@property pooled_dim(self) -> int`; CLI helper simplified to `enc.pooled_dim` directly |
| `models.pool_token_stream` was exported from `encoders.fake.__all__` but derived its output dimension from `tokens.max()`, so two calls with different max-IDs would yield non-comparable vectors that silently mis-fit a FAISS index built with the first dim. | MEDIUM | `encoders/fake.py`, `models.py` | `v0.1.0a3` — removed from `__all__` (the helper stays inside `models.py` for back-compat with the existing tests); a future fixed-vocab `pool_token_stream_fixed(n_codebooks, vocab_size)` helper is queued in `TODO_v0.2.md` for `v0.1.1` |
| `encoders/mimi.py` carried a dead `if TYPE_CHECKING: pass` block left over from an earlier refactor. | LOW (refinement) | `encoders/mimi.py` | `v0.1.0a3` — block removed |

## Closed by the audit cycle but not requiring code changes

- **B-04 / branch-protection strength** (`enforce_admins=false`,
  required-review count `0`). Intentional pre-alpha pre-contributor
  state; documented as a DEFER-DECISION until the project opens up
  contributor PRs.
- **B-05 / Vercel + Claude check-suites stuck in `queued`**. These
  are external GitHub Apps installed at the org level; they do not
  participate in the required-status-checks set and do not block
  merges. Documented but not removed because removing an org-wide
  app from a single repo is a user-level operation the audit agent
  cannot perform.
- **C-03 / `first-interaction` workflow not shipped**. Explicit
  intentional omission documented in `TODO_v0.2.md`: the only
  current first-interactors are Dependabot bots, which we already
  handle without greeting.
- **C-04 / architecture doc says "8 CI jobs" but we have 11**. The
  architecture memory is immutable per the project's auto-memory
  convention; `project_mimirag_state.md` is the live source of
  truth and already records 11 required status checks.

## Deferred (cosmetic, listed in `TODO_v0.2.md` "v0.1.0a3 post-/compact re-audit backlog")

- bench bootstrap `for`-loop vectorisation (`bench.py:80-82`,
  `bench.py:107-109`).
- `pipeline.py` latency-measurement deduplication (the `hybrid`
  axis remeasures around the fused call).
- `cli.py:96` `# type: ignore[arg-type]` → `cast(Axis, axis)`.
- `encoders/__init__.py` registry triplication.
- `scripts/honest_marketing_check.py` `\bperfect\b` regex
  false-positive risk.
- `data.py` vs `faiss_index.py` import-style inconsistency.
- `bench.py:107` loop variable `k` shadowing the module-wide
  `k=cutoff` convention.
- `fusion.py:30` RRF tie-break behaviour documented in 1 line of
  `RRFFuser` docstring.

## Confidence

- The four-agent audit run is reproducible (all agents took read-only
  access; the Bash tools they invoked are listed in their reports).
- `v0.1.0a3` HEAD (the commit that lands these fixes) is tagged at
  ship time; the diff against `v0.1.0a2` is enclosed in two commits
  (one for the security PR auto-merge, one for the consolidated
  re-audit landing).
- The local mypy + ruff + pytest matrix is green on the prerelease
  commit (99 tests pass; 4 new tests for the Protocol sample-rate
  contract).

## How to reproduce

```bash
git clone https://github.com/hinanohart/mimirag
cd mimirag && git checkout v0.1.0a3
uv sync --extra dev
uv run pytest -q                              # 99 pass
uv run mypy src                               # strict, no issues
uv run ruff check                             # no issues
uv run python scripts/honest_marketing_check.py
uv run pip-licenses --format=plain --with-license-file --fail-on='GPL;LGPL;AGPL'

gh api repos/hinanohart/mimirag/dependabot/alerts \
  --jq '.[] | {n: .number, state: .state, pkg: .dependency.package.name}'
```
