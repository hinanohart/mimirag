"""Deterministic CPU-only fake encoders.

These mirror the *interface* of the real Mimi / Whisper / BGE-M3
backends so the CI matrix can exercise the full pipeline without
downloading model weights. They are NOT meant to be informative — they
are meant to be (a) deterministic for a given waveform, (b) cheap on
CPU, (c) shaped like the real outputs.

Real numerical claims in `bench/RESULTS.md` MUST be produced with the
real backends and tagged `[MEASURED YYYY-MM-DD]`; the honest-marketing
CI grep enforces this.
"""

from __future__ import annotations

import hashlib

import numpy as np

from mimirag.models import TokenStream, pool_token_stream


def _seed_from_waveform(waveform: np.ndarray, salt: str = "") -> int:
    """Hash a waveform (first 4096 samples) + salt → a 32-bit seed."""
    head = waveform[:4096].tobytes()
    digest = hashlib.blake2b(head + salt.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % (2**32)


class FakeMimiEncoder:
    """Deterministic Mimi-shaped tokenizer.

    Output shape: (Q=`n_codebooks`, T=`ceil(duration_s * sample_hz)`).
    Tokens are deterministic per waveform (via blake2b seed of the first
    4096 samples). Vocab size per codebook is `vocab_size`.
    """

    def __init__(
        self,
        n_codebooks: int = 8,
        sample_hz: float = 12.5,
        vocab_size: int = 2048,
        name: str = "fake-mimi",
    ) -> None:
        if n_codebooks <= 0:
            raise ValueError("n_codebooks must be > 0")
        if sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")
        if vocab_size <= 1:
            raise ValueError("vocab_size must be > 1")
        self.n_codebooks = n_codebooks
        self.sample_hz = sample_hz
        self.vocab_size = vocab_size
        self.name = name

    def encode(
        self, waveform: np.ndarray, sample_rate: int, source_id: str
    ) -> tuple[TokenStream, np.ndarray]:
        if waveform.ndim != 1:
            raise ValueError(f"expected 1-D waveform, got shape {waveform.shape}")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if waveform.size == 0:
            raise ValueError("waveform is empty")
        duration_s = waveform.size / float(sample_rate)
        n_frames = max(1, int(np.ceil(duration_s * self.sample_hz)))

        seed = _seed_from_waveform(waveform, salt=source_id)
        rng = np.random.default_rng(seed)
        tokens = rng.integers(
            low=0,
            high=self.vocab_size,
            size=(self.n_codebooks, n_frames),
            dtype=np.int32,
        )
        stream = TokenStream(
            source_id=source_id,
            sample_hz=self.sample_hz,
            n_codebooks=self.n_codebooks,
            n_frames=n_frames,
            encoder=self.name,
            meta={"backend": "fake", "vocab_size": str(self.vocab_size)},
        )
        return stream, tokens

    def pool(self, stream: TokenStream, tokens: np.ndarray) -> np.ndarray:
        if tokens.ndim != 2 or tokens.shape[0] != self.n_codebooks:
            raise ValueError(f"expected ({self.n_codebooks}, T) tokens, got shape {tokens.shape}")
        if tokens.shape[1] == 0:
            raise ValueError("token stream has 0 frames")
        # Stable pooled dim = n_codebooks * vocab_size, independent of input
        out = np.zeros((self.n_codebooks * self.vocab_size,), dtype=np.float32)
        for ci in range(self.n_codebooks):
            counts = np.bincount(tokens[ci], minlength=self.vocab_size).astype(np.float32)
            out[ci * self.vocab_size : (ci + 1) * self.vocab_size] = counts / float(tokens.shape[1])
        norm = float(np.linalg.norm(out))
        if norm > 0.0:
            out = out / norm
        return out

    @property
    def pooled_dim(self) -> int:
        return self.n_codebooks * self.vocab_size


class FakeTextEncoder:
    """Deterministic hash-based text → fixed-dim float vector."""

    def __init__(self, dim: int = 256, name: str = "fake-text") -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim
        self.name = name

    def encode_text(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError(f"expected str, got {type(text).__name__}")
        # Map each whitespace-split token to a hash bucket and accumulate.
        out = np.zeros((self.dim,), dtype=np.float32)
        for tok in text.lower().split():
            h = int.from_bytes(
                hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(),
                "little",
                signed=False,
            )
            idx = h % self.dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            out[idx] += sign
        norm = float(np.linalg.norm(out))
        if norm > 0.0:
            out = out / norm
        return out


class FakeASR:
    """Deterministic waveform → 'fake utterance N' transcript.

    Same waveform → same transcript. Tests can pin known relationships.
    """

    def __init__(self, name: str = "fake-asr") -> None:
        self.name = name

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> str:
        if waveform.ndim != 1:
            raise ValueError(f"expected 1-D waveform, got shape {waveform.shape}")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if waveform.size == 0:
            return ""
        seed = _seed_from_waveform(waveform, salt="asr")
        rng = np.random.default_rng(seed)
        # 8-token deterministic "transcript" with a stable vocabulary so
        # adjacent waveforms can still share tokens (and therefore match
        # under text retrieval).
        vocab = [
            "alpha",
            "bravo",
            "charlie",
            "delta",
            "echo",
            "foxtrot",
            "golf",
            "hotel",
            "india",
            "juliet",
            "kilo",
            "lima",
            "mike",
            "november",
            "oscar",
            "papa",
            "quebec",
            "romeo",
            "sierra",
            "tango",
        ]
        n_words = 8
        idxs = rng.integers(low=0, high=len(vocab), size=(n_words,), dtype=np.int32)
        return " ".join(vocab[int(i)] for i in idxs)


__all__ = ["FakeASR", "FakeMimiEncoder", "FakeTextEncoder", "pool_token_stream"]
