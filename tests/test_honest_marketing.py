"""Tests for the honest-marketing CI grep.

These import the script directly rather than spawning a subprocess so
coverage applies and the test is fast.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "honest_marketing_check.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("honest_marketing_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["honest_marketing_check"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_banned_phrase_detected(tmp_path: Path) -> None:
    mod = _load()
    f = tmp_path / "bad.md"
    f.write_text("This is the world first thing.\n")
    violations = mod.scan_file(f)  # type: ignore[attr-defined]
    assert any("world first" in v.lower() for v in violations)


def test_exempt_marker_skips_line(tmp_path: Path) -> None:
    mod = _load()
    f = tmp_path / "ok.md"
    f.write_text("We do NOT claim 'world first'. # HONEST-CLAIM: counter-example\n")
    assert mod.scan_file(f) == []  # type: ignore[attr-defined]


def test_clean_file_no_violations(tmp_path: Path) -> None:
    mod = _load()
    f = tmp_path / "clean.md"
    f.write_text("Just an ordinary sentence.\n")
    assert mod.scan_file(f) == []  # type: ignore[attr-defined]


def test_sota_detected(tmp_path: Path) -> None:
    mod = _load()
    f = tmp_path / "x.md"
    f.write_text("We achieved SOTA on bench.\n")
    v = mod.scan_file(f)  # type: ignore[attr-defined]
    assert v and "SOTA" in v[0]


def test_japanese_banned_phrase(tmp_path: Path) -> None:
    mod = _load()
    f = tmp_path / "ja.md"
    f.write_text("完全自動で永続的に動きます。\n")
    v = mod.scan_file(f)  # type: ignore[attr-defined]
    assert len(v) >= 2


@pytest.mark.parametrize(
    "phrase",
    ["state of the art", "state-of-the-art", "world-leading", "guaranteed", "forever"],
)
def test_various_banned_phrases(tmp_path: Path, phrase: str) -> None:
    mod = _load()
    f = tmp_path / "x.md"
    f.write_text(f"This is {phrase} stuff.\n")
    v = mod.scan_file(f)  # type: ignore[attr-defined]
    assert v, f"expected violation for {phrase!r}"
