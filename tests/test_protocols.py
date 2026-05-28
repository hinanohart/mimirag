"""Tests confirming the fake backends satisfy the typing.Protocols."""

from __future__ import annotations

from mimirag.encoders import FakeASR, FakeMimiEncoder, FakeTextEncoder
from mimirag.protocols import ASRBackend, Encoder, TextEncoder


def test_fake_mimi_satisfies_encoder() -> None:
    enc = FakeMimiEncoder()
    assert isinstance(enc, Encoder)
    assert enc.name == "fake-mimi"
    assert enc.sample_hz == 12.5
    assert enc.n_codebooks == 8


def test_fake_text_satisfies_text_encoder() -> None:
    txt = FakeTextEncoder(dim=64)
    assert isinstance(txt, TextEncoder)
    assert txt.dim == 64


def test_fake_asr_satisfies_asr_backend() -> None:
    asr = FakeASR()
    assert isinstance(asr, ASRBackend)
    assert asr.name == "fake-asr"
