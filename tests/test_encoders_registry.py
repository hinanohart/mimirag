"""Tests for the encoder/text-encoder/ASR registry in mimirag.encoders.

The `mimi`/`bge`/`whisper` branches require the corresponding extras
and download model weights at runtime; in the default CI matrix we
only exercise the `fake` branches and the unknown-name error paths.
"""

from __future__ import annotations

import pytest

from mimirag.encoders import (
    FakeASR,
    FakeMimiEncoder,
    FakeTextEncoder,
    get_asr,
    get_encoder,
    get_text_encoder,
)


def test_get_encoder_fake_returns_fake_mimi() -> None:
    enc = get_encoder("fake")
    assert isinstance(enc, FakeMimiEncoder)


def test_get_encoder_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown audio encoder"):
        get_encoder("doesnotexist")


def test_get_text_encoder_fake_returns_fake_text() -> None:
    txt = get_text_encoder("fake")
    assert isinstance(txt, FakeTextEncoder)


def test_get_text_encoder_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown text encoder"):
        get_text_encoder("doesnotexist")


def test_get_asr_fake_returns_fake_asr() -> None:
    asr = get_asr("fake")
    assert isinstance(asr, FakeASR)


def test_get_asr_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown ASR"):
        get_asr("doesnotexist")


def test_get_encoder_fake_passes_kwargs() -> None:
    enc = get_encoder("fake", n_codebooks=4, vocab_size=128)
    assert isinstance(enc, FakeMimiEncoder)
    assert enc.n_codebooks == 4
    assert enc.vocab_size == 128


def test_get_text_encoder_fake_passes_kwargs() -> None:
    txt = get_text_encoder("fake", dim=512)
    assert isinstance(txt, FakeTextEncoder)
    assert txt.dim == 512
