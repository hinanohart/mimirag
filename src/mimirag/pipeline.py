"""End-to-end orchestration: ingest → index → query → retrieve.

`Pipeline` holds two indexes (codec + text) plus an ASR backend. It
exposes a single `query(audio, axis)` that selects which path(s) run,
including the RRF hybrid axis. Encoders/indexes/ASR/text-encoder are
injected so tests can plug in `FakeMimiEncoder` etc. without touching
the orchestration logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from mimirag.fusion import RRFFuser
from mimirag.indexes.faiss_index import FaissCpuIndex
from mimirag.models import AudioChunk, Axis, Hit, RetrievalResult
from mimirag.protocols import ASRBackend, Encoder, TextEncoder


@dataclass
class IngestedDoc:
    chunk: AudioChunk
    transcript: str


class Pipeline:
    def __init__(
        self,
        encoder: Encoder,
        text_encoder: TextEncoder,
        asr: ASRBackend,
        codec_dim: int,
    ) -> None:
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.asr = asr
        self.codec_index = FaissCpuIndex(dim=codec_dim)
        self.text_index = FaissCpuIndex(dim=text_encoder.dim)
        self._docs: dict[str, IngestedDoc] = {}
        self._fuser = RRFFuser()

    def __len__(self) -> int:
        return len(self._docs)

    def ingest(self, chunk: AudioChunk, waveform: np.ndarray) -> IngestedDoc:
        if chunk.id in self._docs:
            raise ValueError(f"duplicate chunk id: {chunk.id!r}")
        stream, tokens = self.encoder.encode(waveform, chunk.sample_rate, chunk.id)
        pooled = self.encoder.pool(stream=stream, tokens=tokens)
        self.codec_index.add(chunk.id, pooled)
        transcript = (
            chunk.text
            if chunk.text is not None
            else self.asr.transcribe(waveform, chunk.sample_rate)
        )
        text_vec = self.text_encoder.encode_text(transcript)
        self.text_index.add(chunk.id, text_vec)
        doc = IngestedDoc(chunk=chunk, transcript=transcript)
        self._docs[chunk.id] = doc
        return doc

    def get_transcript(self, doc_id: str) -> str:
        return self._docs[doc_id].transcript

    # ------------------------------------------------------------------
    # Query paths

    def _query_codec(
        self, waveform: np.ndarray, sample_rate: int, query_id: str, k: int
    ) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        stream, tokens = self.encoder.encode(waveform, sample_rate, query_id)
        pooled = self.encoder.pool(stream=stream, tokens=tokens)
        raw = self.codec_index.search(pooled, k)
        dt = (time.perf_counter() - t0) * 1000.0
        return [
            Hit(doc_id=doc_id, score=score, axis="codec-only", rank=rank)
            for rank, (doc_id, score) in enumerate(raw)
        ], dt

    def _query_text(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        k: int,
        axis: Axis,
        query_text: str | None,
    ) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        text = query_text if query_text is not None else self.asr.transcribe(waveform, sample_rate)
        vec = self.text_encoder.encode_text(text)
        raw = self.text_index.search(vec, k)
        dt = (time.perf_counter() - t0) * 1000.0
        return [
            Hit(doc_id=doc_id, score=score, axis=axis, rank=rank)
            for rank, (doc_id, score) in enumerate(raw)
        ], dt

    def query(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        axis: Axis = "hybrid",
        k: int = 5,
        query_id: str = "q",
        query_text: str | None = None,
    ) -> RetrievalResult:
        """Run one of the four retrieval axes.

        - `text-only`   : Whisper transcript → BGE → text index
        - `codec-only`  : Mimi pooled → codec index
        - `hybrid`      : RRF of text-only and codec-only (default)
        - `baseline`    : Whisper round-trip text index (identical to
                          text-only in this implementation; kept as a
                          distinct *axis label* in RESULTS for clarity)
        """
        if k <= 0:
            raise ValueError("k must be > 0")
        if axis == "codec-only":
            hits, dt = self._query_codec(waveform, sample_rate, query_id, k)
            return RetrievalResult(query_id=query_id, axis=axis, hits=hits, latency_ms=dt)
        if axis in ("text-only", "baseline"):
            hits, dt = self._query_text(waveform, sample_rate, k, axis, query_text)
            return RetrievalResult(query_id=query_id, axis=axis, hits=hits, latency_ms=dt)
        if axis == "hybrid":
            t0 = time.perf_counter()
            text_hits, _ = self._query_text(waveform, sample_rate, k, "text-only", query_text)
            codec_hits, _ = self._query_codec(waveform, sample_rate, query_id, k)
            fused = self._fuser.fuse([text_hits, codec_hits], k)
            dt = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult(query_id=query_id, axis="hybrid", hits=fused, latency_ms=dt)
        raise ValueError(f"unknown axis: {axis!r}")
