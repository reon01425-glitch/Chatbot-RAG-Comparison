"""Unit tests for the deterministic metrics in evaluate_benchmark.py (no LLM needed).

Run:  python -m pytest tests/test_benchmark_metrics.py -q
"""
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate_benchmark as eb  # noqa: E402

BENCH = eb.load_benchmark(ROOT / "benchmark" / "sop_benchmark_v1.json")
ITEMS = {it["id"]: it for it in BENCH["items"]}
ACTORS = BENCH["actors"]


def _numbered(ref: str) -> str:
    """Turn '1) ... 2) ...' reference prose into one numbered line per step."""
    parts = re.split(r"\s(?=\d+\)\s)", ref)
    return "\n".join(p.strip() for p in parts)


def _score(answer: str, proc: str):
    al = eb.align_steps(answer, BENCH["procedures"][proc]["steps"], ACTORS)
    return eb.procedural_scores(al), al


# ---------------------------------------------------------------- pattern semantics
def test_whole_word_vs_prefix():
    assert eb.match_any("ke bak fakultas", ["bak"])
    assert not eb.match_any("bakal diproses", ["bak"])
    assert eb.match_any("melampirkan berkas", ["melampir*"])
    assert not eb.match_any("melampirkan berkas", ["lampir"])
    assert eb.match_any("ketua   program\nstudi", ["ketua program studi"])


def test_dekan_not_confused_with_wakil_dekan_or_document_name():
    named, _ = eb.actors_in("wakil dekan sumber daya menandatangani surat izin dekan", ACTORS)
    assert "Dekan" not in named
    assert "Wakil Dekan Sumber Daya" in named
    named, _ = eb.actors_in("dekan menandatangani surat", ACTORS)
    assert "Dekan" in named


def test_generic_alias_is_not_a_mismatch():
    named, generic = eb.actors_in("form anda diperiksa kelengkapannya", ACTORS)
    assert not named and "Mahasiswa" in generic


# ---------------------------------------------------------------- reference answers are the upper bound
@pytest.mark.parametrize("qid", ["CUTI-P1", "LEG-P1", "PROP-P1", "BEA-P1", "IRS-P1", "AKT-P1", "UKT-P1"])
def test_reference_answer_scores_perfectly(qid):
    it = ITEMS[qid]
    s, al = _score(_numbered(it["reference"]), it["procedure"])
    missing = [a["n"] for a in al if not a["detected"]]
    assert s["step_coverage"] == 1.0, f"undetected steps {missing}"
    assert s["tir"] == 0.0
    assert (s["amr"] == 0.0) or math.isnan(s["amr"]), [a for a in al if a["actor_status"] == "mismatch"]
    assert eb.key_fact_recall(it["reference"], it["must_include"]) == 1.0


@pytest.mark.parametrize("qid", [i["id"] for i in BENCH["items"] if i.get("answerable")])
def test_every_reference_contains_its_required_facts(qid):
    it = ITEMS[qid]
    assert eb.key_fact_recall(it["reference"], it["must_include"]) == 1.0


# ---------------------------------------------------------------- failure modes are detected
def test_reversed_order_gives_high_tir():
    lines = _numbered(ITEMS["CUTI-P1"]["reference"]).splitlines()
    s, _ = _score("\n".join(reversed(lines)), "CUTI")
    assert s["tir"] > 0.8


def test_actor_swap_is_a_mismatch():
    ans = ("1. Mahasiswa mengunduh dan mengisi Form Cuti Akademik serta melampirkan persyaratan dan mengisi di SIAP.\n"
           "2. Mahasiswa meminta persetujuan dan tanda tangan Dosen Wali.\n"   # SOP: Ketua Program Studi
           "3. Form diserahkan ke Dekan lalu didisposisikan ke Subbag Akademik.")
    s, al = _score(ans, "CUTI")
    assert al[1]["actor_status"] == "mismatch"
    assert s["amr"] > 0


def test_missing_steps_lower_coverage():
    ans = "1. Isi form cuti dan lampirkan persyaratan, isi juga di SIAP.\n2. Ambil surat di loket akademik."
    s, _ = _score(ans, "CUTI")
    assert s["step_coverage"] <= 0.25


def test_refusal_detection():
    pats = BENCH["refusal_patterns"]
    assert eb.is_refusal("Maaf, informasi tersebut tidak ditemukan dalam SOP resmi.", pats)
    assert not eb.is_refusal("Biaya legalisir adalah Rp5.000 per lembar.", pats)


def test_holm_and_bootstrap():
    adj = eb.holm([0.01, 0.04, float("nan"), 0.03])
    assert adj[0] == pytest.approx(0.03) and math.isnan(adj[2])
    mean, lo, hi, n = eb.bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1])
    assert n == 8 and lo <= mean <= hi
