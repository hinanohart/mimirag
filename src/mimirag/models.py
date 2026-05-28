"""Pydantic-v2 typed payloads for the mimirag pipeline."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

Axis = Literal["text-only", "codec-only", "hybrid", "baseline"]
"""The four retrieval axes exposed at query time."""


class AudioChunk(BaseModel):
    """A single utterance-level audio segment.

    `waveform` is held outside the model (NumPy is not JSON-serialisable);
    we keep `path`, `sample_rate`, `n_samples`, and `id` here so chunks
    can round-trip through JSON without losing identity.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=1)
    path: str = Field(...)
    sample_rate: int = Field(..., gt=0)
    n_samples: int = Field(..., gt=0)
    text: str | None = None  # ground-truth transcript, if any

    @property
    def duration_s(self) -> float:
        return self.n_samples / float(self.sample_rate)


class TokenStream(BaseModel):
    """The output of `Encoder.encode`: a (codebook, time)-shaped token grid.

    `tokens` is held outside the model (NumPy `int32` array); only metadata
    travels through pydantic.
    """

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(..., min_length=1)
    sample_hz: float = Field(..., gt=0)
    n_codebooks: int = Field(..., gt=0)
    n_frames: int = Field(..., gt=0)
    encoder: str = Field(..., min_length=1)
    meta: dict[str, str] = Field(default_factory=dict)


class Hit(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., min_length=1)
    score: float
    axis: Axis
    rank: int = Field(..., ge=0)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str = Field(..., min_length=1)
    axis: Axis
    hits: list[Hit] = Field(default_factory=list)
    latency_ms: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def _ranks_are_dense(self) -> RetrievalResult:
        for expected, hit in enumerate(self.hits):
            if hit.rank != expected:
                raise ValueError(
                    f"hits must have dense ranks 0..n-1; got rank={hit.rank} at position {expected}"
                )
            if hit.axis != self.axis:
                raise ValueError(f"hit axis '{hit.axis}' does not match result axis '{self.axis}'")
        return self


def pool_token_stream(tokens: np.ndarray, n_codebooks: int) -> np.ndarray:
    """Pool a (Q, T) token grid into a fixed-size float vector.

    Strategy: one-hot per codebook + mean-over-time + concatenate. This
    gives a (Q * V_eff)-dim vector where V_eff is the per-codebook
    codebook size observed in this stream (caller responsibility to fix V
    at index build time).
    """
    if tokens.ndim != 2:
        raise ValueError(f"expected (Q, T) tokens, got shape {tokens.shape}")
    if tokens.shape[0] != n_codebooks:
        raise ValueError(f"expected {n_codebooks} codebooks, got {tokens.shape[0]}")
    if tokens.shape[1] == 0:
        raise ValueError("token stream has 0 frames")

    vocab = int(tokens.max()) + 1
    q, t = tokens.shape
    out = np.zeros((q * vocab,), dtype=np.float32)
    for ci in range(q):
        counts = np.bincount(tokens[ci], minlength=vocab).astype(np.float32)
        out[ci * vocab : (ci + 1) * vocab] = counts / float(t)
    # L2 normalize so inner-product index ≈ cosine
    norm = float(np.linalg.norm(out))
    if norm > 0.0:
        out = out / norm
    return out
