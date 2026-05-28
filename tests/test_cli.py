"""CLI smoke tests via click's CliRunner."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from mimirag.cli import main


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "mimirag" in result.output


def test_cli_verify_fake_backend() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["verify"])
    assert result.exit_code == 0
    assert "OK" in result.output


def test_cli_make_tiny_writes_wavs(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "corpus"
    result = runner.invoke(main, ["make-tiny", "--out-dir", str(out), "--n", "3"])
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("*.wav"))) == 3


def test_cli_ingest_then_query(tmp_path: Path) -> None:
    runner = CliRunner()
    corpus = tmp_path / "corpus"
    runner.invoke(main, ["make-tiny", "--out-dir", str(corpus), "--n", "4"])
    out = tmp_path / "index"
    r = runner.invoke(
        main, ["ingest", "--corpus", str(corpus), "--out", str(out), "--backend", "fake"]
    )
    assert r.exit_code == 0, r.output
    assert (out / "manifest.json").exists()
    one_wav = next(corpus.glob("*.wav"))
    r2 = runner.invoke(
        main,
        [
            "query",
            "--audio",
            str(one_wav),
            "--corpus",
            str(corpus),
            "--axis",
            "codec-only",
            "--k",
            "2",
            "--backend",
            "fake",
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert '"axis": "codec-only"' in r2.output


def test_cli_bench_writes_results(tmp_path: Path) -> None:
    runner = CliRunner()
    corpus = tmp_path / "corpus"
    runner.invoke(main, ["make-tiny", "--out-dir", str(corpus), "--n", "5"])
    out = tmp_path / "RESULTS.md"
    r = runner.invoke(
        main,
        [
            "bench",
            "--corpus",
            str(corpus),
            "--out",
            str(out),
            "--backend",
            "fake",
            "--n-boot",
            "100",
        ],
    )
    assert r.exit_code == 0, r.output
    text = out.read_text()
    assert "[MEASURED" in text
    assert "hybrid" in text
