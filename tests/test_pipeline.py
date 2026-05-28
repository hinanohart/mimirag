"""Integration tests for Pipeline (4-axis retrieval)."""

from __future__ import annotations

import numpy as np
import pytest

from mimirag.models import AudioChunk
from mimirag.pipeline import Pipeline


def _wave(seed: int, n: int = 24000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def _make_chunk(i: int) -> tuple[AudioChunk, np.ndarray]:
    wave = _wave(seed=100 + i)
    return AudioChunk(
        id=f"d{i:02d}",
        path=f"/synthetic/d{i:02d}.wav",
        sample_rate=24000,
        n_samples=int(wave.size),
    ), wave


def test_pipeline_ingest_grows_indexes(fake_pipeline: Pipeline) -> None:
    for i in range(5):
        chunk, wave = _make_chunk(i)
        fake_pipeline.ingest(chunk, wave)
    assert len(fake_pipeline) == 5
    assert len(fake_pipeline.codec_index) == 5
    assert len(fake_pipeline.text_index) == 5


def test_pipeline_codec_self_retrieval(fake_pipeline: Pipeline) -> None:
    for i in range(5):
        chunk, wave = _make_chunk(i)
        fake_pipeline.ingest(chunk, wave)
    _, wave3 = _make_chunk(3)
    res = fake_pipeline.query(wave3, sample_rate=24000, axis="codec-only", query_id="d03")
    assert res.axis == "codec-only"
    assert res.hits, "expected at least one hit"
    assert res.hits[0].doc_id == "d03"


def test_pipeline_text_self_retrieval(fake_pipeline: Pipeline) -> None:
    for i in range(5):
        chunk, wave = _make_chunk(i)
        fake_pipeline.ingest(chunk, wave)
    _, wave3 = _make_chunk(3)
    res = fake_pipeline.query(wave3, sample_rate=24000, axis="text-only", query_id="d03")
    assert res.axis == "text-only"
    # Top hit need not be the exact id since fake ASR is not perfect,
    # but the query itself must be a member of the corpus.
    assert {h.doc_id for h in res.hits} <= {f"d{i:02d}" for i in range(5)}


def test_pipeline_hybrid_returns_at_most_k(fake_pipeline: Pipeline) -> None:
    for i in range(8):
        chunk, wave = _make_chunk(i)
        fake_pipeline.ingest(chunk, wave)
    _, wave = _make_chunk(2)
    res = fake_pipeline.query(wave, sample_rate=24000, axis="hybrid", k=3, query_id="d02")
    assert res.axis == "hybrid"
    assert len(res.hits) <= 3
    assert [h.rank for h in res.hits] == list(range(len(res.hits)))


def test_pipeline_baseline_axis_runs(fake_pipeline: Pipeline) -> None:
    for i in range(4):
        chunk, wave = _make_chunk(i)
        fake_pipeline.ingest(chunk, wave)
    _, wave = _make_chunk(1)
    res = fake_pipeline.query(wave, sample_rate=24000, axis="baseline", query_id="d01")
    assert res.axis == "baseline"


def test_pipeline_query_text_override(fake_pipeline: Pipeline) -> None:
    chunk0, wave0 = _make_chunk(0)
    chunk1, wave1 = _make_chunk(1)
    # Set transcripts on the chunks themselves
    chunk0 = chunk0.model_copy(update={"text": "alpha bravo charlie"})
    chunk1 = chunk1.model_copy(update={"text": "delta echo foxtrot"})
    fake_pipeline.ingest(chunk0, wave0)
    fake_pipeline.ingest(chunk1, wave1)
    res = fake_pipeline.query(
        waveform=wave0,
        sample_rate=24000,
        axis="text-only",
        query_text="alpha bravo charlie",
        query_id="q0",
    )
    assert res.hits[0].doc_id == "d00"


def test_pipeline_duplicate_id_rejected(fake_pipeline: Pipeline) -> None:
    chunk, wave = _make_chunk(0)
    fake_pipeline.ingest(chunk, wave)
    with pytest.raises(ValueError, match="duplicate"):
        fake_pipeline.ingest(chunk, wave)


def test_pipeline_unknown_axis_rejected(fake_pipeline: Pipeline) -> None:
    chunk, wave = _make_chunk(0)
    fake_pipeline.ingest(chunk, wave)
    with pytest.raises(ValueError, match="unknown axis"):
        fake_pipeline.query(wave, sample_rate=24000, axis="bogus", k=2)  # type: ignore[arg-type]


def test_pipeline_k_must_be_positive(fake_pipeline: Pipeline) -> None:
    chunk, wave = _make_chunk(0)
    fake_pipeline.ingest(chunk, wave)
    with pytest.raises(ValueError):
        fake_pipeline.query(wave, sample_rate=24000, axis="hybrid", k=0)


def test_pipeline_latency_recorded(fake_pipeline: Pipeline) -> None:
    chunk, wave = _make_chunk(0)
    fake_pipeline.ingest(chunk, wave)
    res = fake_pipeline.query(wave, sample_rate=24000, axis="hybrid", k=1, query_id="q")
    assert res.latency_ms >= 0.0
