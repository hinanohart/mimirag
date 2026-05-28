"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from mimirag.data import make_tiny_corpus
from mimirag.encoders import get_asr, get_encoder, get_text_encoder
from mimirag.pipeline import Pipeline


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> list[Path]:
    return make_tiny_corpus(tmp_path / "tiny", n_docs=10, sample_rate=24000)


@pytest.fixture
def fake_pipeline() -> Iterator[Pipeline]:
    enc = get_encoder("fake")
    txt = get_text_encoder("fake")
    asr = get_asr("fake")
    pipe = Pipeline(
        encoder=enc,
        text_encoder=txt,
        asr=asr,
        codec_dim=int(enc.pooled_dim),  # type: ignore[attr-defined]
    )
    yield pipe
