"""S0 verification gate POC.

Promised by S0_GATE.md § 5: a 30-line notebook (kept as a script for
reproducibility) that materialises a 1 s synthetic 24 kHz waveform,
runs the fake Mimi encoder, asserts the token tensor shape, and
exercises the FAISS-CPU add/search round-trip.

Run: `uv run python poc/s0_smoke.py`
"""

from __future__ import annotations

import numpy as np

from mimirag.encoders.fake import FakeMimiEncoder
from mimirag.indexes.faiss_index import FaissCpuIndex


def main() -> None:
    sr = 24_000
    rng = np.random.default_rng(seed=42)
    wave = (0.05 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    wave += 0.01 * rng.standard_normal(sr).astype(np.float32)

    enc = FakeMimiEncoder(n_codebooks=8, sample_hz=12.5, vocab_size=256)
    stream, tokens = enc.encode(wave, sample_rate=sr, source_id="poc-1")
    assert tokens.shape == (8, 13), f"unexpected token shape {tokens.shape}"
    assert tokens.dtype == np.int32
    assert stream.encoder == "fake-mimi"

    vec = enc.pool(stream=stream, tokens=tokens)
    assert vec.shape == (enc.pooled_dim,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    idx = FaissCpuIndex(dim=enc.pooled_dim)
    idx.add("poc-1", vec)
    hits = idx.search(vec, k=1)
    assert hits and hits[0][0] == "poc-1"

    print(
        f"S0 POC ok: tokens={tokens.shape}, pooled_dim={enc.pooled_dim}, "
        f"top_hit=({hits[0][0]}, {hits[0][1]:.4f})"
    )


if __name__ == "__main__":
    main()
