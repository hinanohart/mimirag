# S0 Verification Gate — mimirag v0.1.0a1

**Executed**: 2026-05-29 (UTC heartbeat 2026-05-28T17:43Z)
**Verdict**: **PASS** — S1 着手可

## 1. Self-overlap check (hinanohart org, live `gh repo list --limit 300`)
Regex `mimi|codec|audio|neural-audio|rvq|asr|retriev|embed|rag` で 11 hit。各 hit を semantic 評価:

| repo | overlap class | 差別化 |
|---|---|---|
| lyric-video-generator | none (regex 偶然 hit) | audio→video pipeline、retrieval ではない |
| citelock | ADJACENT (text RAG gate) | mimirag は audio 直接 index、citelock は text claim NLI gate。modality 別 |
| driftset | none (conformal bio) | bio domain |
| wav2caption | ADJACENT (audio→text) | mimirag は audio→audio index で round-trip 撤廃が core、wav2caption は forward 変換のみ |
| yuragi | none (regex hit "frag**ility**") | LLM 摂動評価 |
| aufschreibesysteme-fm | ADJACENT (lossy codecs FM) | mimirag は **Mimi neural codec を index space に**、aufschreibesysteme は generative regime 化。目的層が違う |
| transduce | ADJACENT (audio adapters あり) | transduction probes、retrieval ではない |
| mirage | none (regex hit "mi**rag**e") | SAE agent monitor |
| omniprobe | ADJACENT (OMNI joint stream) | SAE interp、RAG ではない |
| differance | ADJACENT (text RAG invariants) | text retrieval invariants、modality 別 |
| hinata-lyric-video-experiments | none | lyric video |

**判定**: 真の semantic overlap = 0。mimirag の moat = 「Kyutai Mimi 12.5Hz semantic-token を vector として直接 index する RAG」で hinanohart org に前例なし。

## 2. Mimi weights license (HF API live fetch)
- `kyutai/mimi` cardData: `license: cc-by-4.0` ✓
- `library_name: transformers` ✓ (S3 で transformers Mimi class 使用、moshi pivot 不要)
- NOTICE/README/CITATION.cff に明記必須 (S1 で実施)

## 3. Omnilingual ASR license (v0.2 backlog)
v0.1 ブロッカー対象外。v0.2 reassess 時点で実調査。

## 4. Prior-art (arxiv + GitHub + PwC)
- arxiv `"mimi codec" AND retrieval`: 0 hit
- GitHub `mimi codec retrieval`: 0 hit
- GitHub `mimi rag`: 5 hit、全て無関係 (sara-ammourh/mimi-RAG = "mini-rag" typo, Shrini9797/Mimir engine, talhakam/Mimir Norse, NazarAbbas234/mimic-rag-flask, ema-da/mimi-rag2 = mini-rag)
- GitHub `neural codec retrieval`: 0 hit

**Claim 表記降格**: "world first" 禁止 ([[feedback_no-permanent-claim-2026-05-14]])。README は `to our knowledge, the first open-source RAG over Kyutai Mimi semantic tokens` で HONEST 化。性能優位は別 claim、要実測。

## 5. POC (S1 内で `poc/s0_smoke.py` として実装)
plan: 1s synthetic 24kHz waveform → Mimi encode → token tensor shape assert → faiss-cpu IndexFlatIP add/search 30 行。

## 6. PyPI name availability
- `mimirag`: HTTP 404 ✓ (available)
- `mimi-rag`: HTTP 404 ✓ (backup)
- **採用名**: `mimirag` (architecture と一致)

## 7. Transitive license (S1 で `uv sync` 後)
`pip-licenses --fail-on='GPL;LGPL;AGPL'` で 0 件確認、CI 強制。

## Sign-off
全 7 項目 PASS。S1 (skeleton) 移行。
