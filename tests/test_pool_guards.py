"""Tests for the `pool` zero-frame and shape guards.

Round-1 critic-gate finding: both `FakeMimiEncoder.pool` and the
`MimiEncoder.pool` implementation divided by `tokens.shape[1]` without
asserting the shape; a `(Q, 0)` array would crash on division. These
guards now exist; pin them with a regression test so the next
refactor cannot quietly remove them.
"""

from __future__ import annotations

import numpy as np
import pytest

from mimirag.encoders.fake import FakeMimiEncoder
from mimirag.models import TokenStream


def _stream(n_codebooks: int, n_frames: int) -> TokenStream:
    return TokenStream(
        source_id="x",
        sample_hz=12.5,
        n_codebooks=n_codebooks,
        n_frames=max(1, n_frames),
        encoder="fake-mimi",
    )


def test_pool_rejects_zero_frames() -> None:
    enc = FakeMimiEncoder(n_codebooks=8, vocab_size=64)
    empty = np.zeros((8, 0), dtype=np.int32)
    with pytest.raises(ValueError, match="0 frames"):
        enc.pool(stream=_stream(8, 1), tokens=empty)


def test_pool_rejects_wrong_codebook_count() -> None:
    enc = FakeMimiEncoder(n_codebooks=8, vocab_size=64)
    wrong = np.zeros((4, 10), dtype=np.int32)
    with pytest.raises(ValueError, match=r"\(8, T\)"):
        enc.pool(stream=_stream(8, 10), tokens=wrong)


def test_pool_rejects_1d_tokens() -> None:
    enc = FakeMimiEncoder(n_codebooks=8, vocab_size=64)
    wrong = np.zeros((10,), dtype=np.int32)
    with pytest.raises(ValueError, match=r"\(8, T\)"):
        enc.pool(stream=_stream(8, 10), tokens=wrong)
