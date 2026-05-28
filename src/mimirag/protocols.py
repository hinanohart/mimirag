"""typing.Protocol surfaces for swap-in backends.

Kept in the same repo at v0.1; only extract to `mimirag-core` once we
have a second consumer (YAGNI; see `project_mimirag_architecture_2026-05-29.md`).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from mimirag.models import Hit, TokenStream


@runtime_checkable
class Encoder(Protocol):
    """Audio → TokenStream + pooled vector.

    Implementations: `encoders.fake.FakeMimiEncoder`,
    `encoders.mimi.MimiEncoder`. The pooled vector is what gets stored in
    the FAISS index; the TokenStream metadata is what gets persisted next
    to it for provenance.

    `accepted_sample_rate` is part of the contract so callers can refuse
    to dispatch a mismatched waveform before the per-backend `encode`
    raises. ``None`` advertises "any positive sample rate". A concrete
    int advertises a single Hz the backend will accept; pipelines must
    resample upstream.
    """

    name: str
    sample_hz: float
    n_codebooks: int
    accepted_sample_rate: int | None

    @property
    def pooled_dim(self) -> int: ...

    def encode(
        self, waveform: np.ndarray, sample_rate: int, source_id: str
    ) -> tuple[TokenStream, np.ndarray]: ...

    def pool(self, stream: TokenStream, tokens: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class TextEncoder(Protocol):
    """Text string → dense float vector.

    Implementations: `encoders.fake.FakeTextEncoder`, `encoders.bge.BGEEncoder`.
    """

    name: str
    dim: int

    def encode_text(self, text: str) -> np.ndarray: ...


@runtime_checkable
class ASRBackend(Protocol):
    """Waveform → transcript.

    Implementations: `encoders.fake.FakeASR`, `encoders.whisper.WhisperASR`.
    See `Encoder.accepted_sample_rate` docstring for the same contract.
    """

    name: str
    accepted_sample_rate: int | None

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> str: ...


@runtime_checkable
class IndexBackend(Protocol):
    """Vector index. FAISS-CPU is the default backing store."""

    dim: int

    def add(self, key: str, vec: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def __len__(self) -> int: ...


@runtime_checkable
class Fuser(Protocol):
    """Combines per-source hit lists into a single ranked list."""

    name: str

    def fuse(self, hits_per_source: list[list[Hit]], k: int) -> list[Hit]: ...
