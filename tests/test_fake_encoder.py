"""Tests for the deterministic fake encoders."""

from __future__ import annotations

import numpy as np
import pytest

from mimirag.encoders.fake import FakeASR, FakeMimiEncoder, FakeTextEncoder


def _signal(seed: int, n: int = 24000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def test_fake_mimi_encode_shape() -> None:
    enc = FakeMimiEncoder(n_codebooks=8, sample_hz=12.5, vocab_size=256)
    wave = _signal(seed=1, n=24000)
    stream, tokens = enc.encode(wave, sample_rate=24000, source_id="u1")
    assert stream.n_codebooks == 8
    assert stream.encoder == "fake-mimi"
    assert tokens.shape == (8, 13)
    assert tokens.dtype == np.int32
    assert tokens.min() >= 0 and tokens.max() < 256


def test_fake_mimi_deterministic() -> None:
    enc = FakeMimiEncoder(vocab_size=128)
    wave = _signal(seed=2)
    s1, t1 = enc.encode(wave, sample_rate=24000, source_id="x")
    s2, t2 = enc.encode(wave, sample_rate=24000, source_id="x")
    assert np.array_equal(t1, t2)
    assert s1.n_frames == s2.n_frames


def test_fake_mimi_source_id_changes_tokens() -> None:
    enc = FakeMimiEncoder(vocab_size=128)
    wave = _signal(seed=3)
    _, t_a = enc.encode(wave, sample_rate=24000, source_id="A")
    _, t_b = enc.encode(wave, sample_rate=24000, source_id="B")
    assert not np.array_equal(t_a, t_b)


def test_fake_mimi_rejects_2d() -> None:
    enc = FakeMimiEncoder()
    wave = np.zeros((2, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        enc.encode(wave, sample_rate=24000, source_id="x")


def test_fake_mimi_rejects_empty() -> None:
    enc = FakeMimiEncoder()
    with pytest.raises(ValueError, match="empty"):
        enc.encode(np.zeros(0, dtype=np.float32), sample_rate=24000, source_id="x")


def test_fake_mimi_pool_normalised() -> None:
    enc = FakeMimiEncoder(vocab_size=128)
    wave = _signal(seed=4)
    stream, tokens = enc.encode(wave, sample_rate=24000, source_id="x")
    vec = enc.pool(stream=stream, tokens=tokens)
    assert vec.shape == (enc.pooled_dim,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_fake_text_encoder_deterministic() -> None:
    txt = FakeTextEncoder(dim=64)
    a = txt.encode_text("hello world")
    b = txt.encode_text("hello world")
    c = txt.encode_text("totally different sentence")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_fake_text_encoder_shape() -> None:
    txt = FakeTextEncoder(dim=128)
    v = txt.encode_text("anything here")
    assert v.shape == (128,)
    assert v.dtype == np.float32


def test_fake_text_encoder_normalised_when_non_empty() -> None:
    txt = FakeTextEncoder(dim=64)
    v = txt.encode_text("hello world")
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_fake_text_encoder_empty_string() -> None:
    txt = FakeTextEncoder(dim=32)
    v = txt.encode_text("")
    assert v.shape == (32,)
    assert np.linalg.norm(v) == 0.0


def test_fake_text_encoder_rejects_non_str() -> None:
    txt = FakeTextEncoder()
    with pytest.raises(TypeError):
        txt.encode_text(123)  # type: ignore[arg-type]


def test_fake_asr_deterministic() -> None:
    asr = FakeASR()
    wave = _signal(seed=5)
    a = asr.transcribe(wave, sample_rate=24000)
    b = asr.transcribe(wave, sample_rate=24000)
    assert a == b
    assert len(a.split()) == 8


def test_fake_asr_empty_waveform() -> None:
    asr = FakeASR()
    assert asr.transcribe(np.zeros(0, dtype=np.float32), sample_rate=24000) == ""
