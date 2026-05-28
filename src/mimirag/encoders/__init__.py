"""Encoder backends.

The default is the deterministic `fake` backend, which lets the CI
matrix run on CPU without downloading multi-GB Mimi weights. Real
backends are lazy-imported only when explicitly requested via
`get_encoder("mimi")` / `get_text_encoder("bge")` / `get_asr("whisper")`,
so the `extras` (`mimirag[mimi]`, `[text]`, `[asr]`) are only required
on the path that uses them.
"""

from __future__ import annotations

from mimirag.encoders.fake import FakeASR, FakeMimiEncoder, FakeTextEncoder
from mimirag.protocols import ASRBackend, Encoder, TextEncoder


def get_encoder(name: str, **kwargs: object) -> Encoder:
    if name == "fake":
        return FakeMimiEncoder(**kwargs)  # type: ignore[arg-type]
    if name == "mimi":
        from mimirag.encoders.mimi import MimiEncoder  # noqa: PLC0415

        return MimiEncoder(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown audio encoder backend: {name!r}")


def get_text_encoder(name: str, **kwargs: object) -> TextEncoder:
    if name == "fake":
        return FakeTextEncoder(**kwargs)  # type: ignore[arg-type]
    if name == "bge":
        from mimirag.encoders.bge import BGEEncoder  # noqa: PLC0415

        return BGEEncoder(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown text encoder backend: {name!r}")


def get_asr(name: str, **kwargs: object) -> ASRBackend:
    if name == "fake":
        return FakeASR(**kwargs)  # type: ignore[arg-type]
    if name == "whisper":
        from mimirag.encoders.whisper import WhisperASR  # noqa: PLC0415

        return WhisperASR(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown ASR backend: {name!r}")


__all__ = [
    "FakeASR",
    "FakeMimiEncoder",
    "FakeTextEncoder",
    "get_asr",
    "get_encoder",
    "get_text_encoder",
]
