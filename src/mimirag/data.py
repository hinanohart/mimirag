"""Helpers for the bundled synthetic corpus + on-disk dataset loading."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from mimirag.models import AudioChunk


def synth_waveform(seed: int, duration_s: float = 1.0, sample_rate: int = 24000) -> np.ndarray:
    """Deterministic synthetic waveform (low-amplitude noise + a tone).

    Used by tests and by the bundled `tests/data/tiny` corpus generator.
    Not a substitute for real speech — just a stable byte sequence.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    freq = 100.0 + (seed % 50) * 5.0
    tone = 0.05 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    noise = 0.01 * rng.standard_normal(n).astype(np.float32)
    return tone + noise


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a wav file as mono float32 + sample rate."""
    waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform.astype(np.float32), int(sr)


def chunk_from_wav(path: str | Path, doc_id: str | None = None) -> AudioChunk:
    waveform, sr = load_wav(path)
    p = Path(path)
    if doc_id is None:
        doc_id = p.stem
    return AudioChunk(
        id=doc_id,
        path=str(p),
        sample_rate=sr,
        n_samples=int(waveform.size),
    )


def make_tiny_corpus(out_dir: str | Path, n_docs: int = 10, sample_rate: int = 24000) -> list[Path]:
    """Materialise a small synthetic corpus on disk for E2E tests."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n_docs):
        wav = synth_waveform(seed=42 + i, sample_rate=sample_rate)
        p = out / f"doc{i:03d}.wav"
        sf.write(str(p), wav, sample_rate, subtype="PCM_16")
        paths.append(p)
    return paths


def fingerprint(waveform: np.ndarray) -> str:
    return hashlib.blake2b(waveform.tobytes(), digest_size=8).hexdigest()
