"""click CLI: `mimirag ingest|query|bench|verify`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from mimirag import __version__
from mimirag.bench import QueryItem, render_results_md, run_bench
from mimirag.data import chunk_from_wav, load_wav, make_tiny_corpus
from mimirag.encoders import get_asr, get_encoder, get_text_encoder
from mimirag.pipeline import Pipeline


def _build_pipeline(backend: str) -> Pipeline:
    """Build a pipeline. `backend` selects the audio-encoder side.

    For CI/CPU we use `fake` everywhere; users with the `[all]` extra
    can pass `--backend mimi`.
    """
    if backend == "fake":
        enc = get_encoder("fake")
        txt = get_text_encoder("fake")
        asr = get_asr("fake")
    elif backend == "mimi":
        enc = get_encoder("mimi")
        txt = get_text_encoder("bge")
        asr = get_asr("whisper")
    else:
        raise click.BadParameter(f"unknown backend: {backend!r}")
    return Pipeline(encoder=enc, text_encoder=txt, asr=asr, codec_dim=int(enc.pooled_dim))


@click.group()
@click.version_option(__version__, prog_name="mimirag")
def main() -> None:
    """mimirag: audio-native RAG over Mimi semantic tokens."""


@main.command()
@click.option(
    "--corpus", "corpus_dir", type=click.Path(exists=True, file_okay=False), required=True
)
@click.option("--out", "out_dir", type=click.Path(), required=True)
@click.option("--backend", default="fake", show_default=True)
def ingest(corpus_dir: str, out_dir: str, backend: str) -> None:
    """Ingest a folder of .wav files into the two indexes."""
    paths = sorted(Path(corpus_dir).glob("*.wav"))
    if not paths:
        raise click.ClickException(f"no .wav under {corpus_dir!r}")
    pipe = _build_pipeline(backend)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for p in paths:
        chunk = chunk_from_wav(p)
        waveform, _ = load_wav(p)
        pipe.ingest(chunk, waveform)
        manifest.append(
            {"id": chunk.id, "path": str(p), "transcript": pipe.get_transcript(chunk.id)}
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    click.echo(f"ingested {len(manifest)} documents → {out}/manifest.json")


@main.command()
@click.option("--audio", "audio_path", type=click.Path(exists=True), required=True)
@click.option(
    "--corpus", "corpus_dir", type=click.Path(exists=True, file_okay=False), required=True
)
@click.option(
    "--axis",
    type=click.Choice(["text-only", "codec-only", "hybrid", "baseline"]),
    default="hybrid",
    show_default=True,
)
@click.option("--k", type=int, default=5, show_default=True)
@click.option("--backend", default="fake", show_default=True)
def query(audio_path: str, corpus_dir: str, axis: str, k: int, backend: str) -> None:
    """One-shot query: ingest corpus, then run one query against it."""
    pipe = _build_pipeline(backend)
    for p in sorted(Path(corpus_dir).glob("*.wav")):
        chunk = chunk_from_wav(p)
        waveform, _ = load_wav(p)
        pipe.ingest(chunk, waveform)
    qwave, qsr = load_wav(audio_path)
    result = pipe.query(qwave, qsr, axis=axis, k=k)  # type: ignore[arg-type]
    click.echo(result.model_dump_json(indent=2))


@main.command()
@click.option(
    "--corpus", "corpus_dir", type=click.Path(exists=True, file_okay=False), required=True
)
@click.option("--out", "out_path", type=click.Path(), required=True)
@click.option("--backend", default="fake", show_default=True)
@click.option("--k", type=int, default=5, show_default=True)
@click.option("--n-boot", type=int, default=1000, show_default=True)
def bench(corpus_dir: str, out_path: str, backend: str, k: int, n_boot: int) -> None:
    """4-axis benchmark. Each ingested doc is also used as its own query."""
    pipe = _build_pipeline(backend)
    queries: list[QueryItem] = []
    for p in sorted(Path(corpus_dir).glob("*.wav")):
        chunk = chunk_from_wav(p)
        waveform, _ = load_wav(p)
        pipe.ingest(chunk, waveform)
        queries.append(
            QueryItem(
                query_id=f"q-{chunk.id}",
                waveform=waveform,
                sample_rate=chunk.sample_rate,
                relevant_doc_ids=frozenset({chunk.id}),
            )
        )
    stats = run_bench(pipe, queries, k=k, n_boot=n_boot)
    md = render_results_md(
        stats,
        corpus_desc=f"{corpus_dir} ({len(queries)} docs, self-retrieval)",
        backend_desc=backend,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(md)
    click.echo(f"wrote {out_path} with {len(stats)} axes x {len(queries)} queries")


@main.command()
@click.option("--out-dir", type=click.Path(), default="tests/data/tiny", show_default=True)
@click.option("--n", type=int, default=10, show_default=True)
def make_tiny(out_dir: str, n: int) -> None:
    """Materialise the bundled synthetic test corpus on disk."""
    paths = make_tiny_corpus(out_dir, n_docs=n)
    click.echo(f"wrote {len(paths)} wav files under {out_dir}")


@main.command()
def verify() -> None:
    """Smoke-check: import the package, build a fake pipeline, ping it."""
    pipe = _build_pipeline("fake")
    click.echo(f"mimirag {__version__} OK; codec_dim={pipe.codec_index.dim}")
    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
