"""Sample-rate contract regression tests.

Background: the Encoder / ASRBackend Protocols document
`accepted_sample_rate` as part of the contract (None = any positive
Hz, int = a fixed Hz the backend will accept). The real backends pin
specific values (Mimi = self._fe.sampling_rate, Whisper = 16000);
the fake backends accept any positive Hz so the CI matrix can run
without resampling.

This test pins the attribute on the fake backends (the real backends
are exercised by @pytest.mark.live in v0.1.1).
"""

from __future__ import annotations

from mimirag.encoders.fake import FakeASR, FakeMimiEncoder, FakeTextEncoder
from mimirag.protocols import ASRBackend, Encoder


def test_fake_mimi_encoder_advertises_sample_rate_attribute() -> None:
    enc = FakeMimiEncoder()
    assert hasattr(enc, "accepted_sample_rate")
    assert enc.accepted_sample_rate is None  # any-Hz contract
    # Protocol runtime check
    assert isinstance(enc, Encoder)


def test_fake_asr_advertises_sample_rate_attribute() -> None:
    asr = FakeASR()
    assert hasattr(asr, "accepted_sample_rate")
    assert asr.accepted_sample_rate is None
    assert isinstance(asr, ASRBackend)


def test_fake_mimi_encoder_advertises_pooled_dim() -> None:
    enc = FakeMimiEncoder(n_codebooks=4, vocab_size=128)
    assert enc.pooled_dim == 4 * 128
    assert isinstance(enc, Encoder)


def test_fake_text_encoder_advertises_dim_via_protocol() -> None:
    # TextEncoder Protocol requires `dim`; this just pins the existing
    # behaviour so a refactor does not silently drop it.
    txt = FakeTextEncoder(dim=64)
    assert txt.dim == 64
