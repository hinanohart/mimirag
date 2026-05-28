"""Unit tests for pydantic models in mimirag.models."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from mimirag.models import AudioChunk, Hit, RetrievalResult, TokenStream, pool_token_stream


def test_audiochunk_basics() -> None:
    c = AudioChunk(id="x", path="/tmp/x.wav", sample_rate=24000, n_samples=48000)
    assert c.duration_s == pytest.approx(2.0)


def test_audiochunk_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        AudioChunk(id="", path="/tmp/x.wav", sample_rate=24000, n_samples=10)


def test_audiochunk_rejects_zero_sr() -> None:
    with pytest.raises(ValidationError):
        AudioChunk(id="x", path="/tmp/x.wav", sample_rate=0, n_samples=10)


def test_tokenstream_validation() -> None:
    ts = TokenStream(source_id="x", sample_hz=12.5, n_codebooks=8, n_frames=12, encoder="fake-mimi")
    assert ts.n_codebooks == 8


def test_hit_axis_must_be_known() -> None:
    with pytest.raises(ValidationError):
        Hit(doc_id="d1", score=1.0, axis="invalid", rank=0)  # type: ignore[arg-type]


def test_retrievalresult_enforces_dense_ranks() -> None:
    hits = [
        Hit(doc_id="a", score=1.0, axis="hybrid", rank=0),
        Hit(doc_id="b", score=0.9, axis="hybrid", rank=2),  # gap
    ]
    with pytest.raises(ValidationError):
        RetrievalResult(query_id="q", axis="hybrid", hits=hits, latency_ms=1.0)


def test_retrievalresult_enforces_axis_match() -> None:
    hits = [Hit(doc_id="a", score=1.0, axis="codec-only", rank=0)]
    with pytest.raises(ValidationError):
        RetrievalResult(query_id="q", axis="hybrid", hits=hits, latency_ms=1.0)


def test_pool_token_stream_normalised() -> None:
    tokens = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32)
    vec = pool_token_stream(tokens, n_codebooks=2)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)


def test_pool_token_stream_rejects_wrong_shape() -> None:
    tokens = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(ValueError, match="expected"):
        pool_token_stream(tokens, n_codebooks=1)


def test_pool_token_stream_rejects_codebook_mismatch() -> None:
    tokens = np.array([[0, 1], [1, 0]], dtype=np.int32)
    with pytest.raises(ValueError, match="expected"):
        pool_token_stream(tokens, n_codebooks=3)


def test_pool_token_stream_rejects_zero_frames() -> None:
    tokens = np.zeros((2, 0), dtype=np.int32)
    with pytest.raises(ValueError, match="0 frames"):
        pool_token_stream(tokens, n_codebooks=2)
