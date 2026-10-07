"""
Verification sheet for the benchmark (Goal A, Tugas 3): generation, auto_flags and applying it.
All writes go to temporary copies; the real benchmark is never touched.
"""
import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import apply_verification as av  # noqa: E402
import make_verification_sheet as mvs  # noqa: E402

BENCH_PATH = ROOT / "benchmark" / "sop_benchmark_v1.json"


@pytest.fixture(scope="module")
def bench():
    return json.loads(BENCH_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows(bench):
    return mvs.build_rows(bench)


def test_one_row_per_item_with_empty_human_columns(bench, rows):
    assert [r["id"] for r in rows] == [it["id"] for it in bench["items"]]
    for r in rows:
        assert list(r) == mvs.COLUMNS
        assert all(r[c] == "" for c in mvs.HUMAN_COLUMNS)
        assert r["kutipan_pdf"], r["id"]


def test_procedural_rows_list_gold_and_pdf_steps(bench, rows):
    for r in rows:
        it = next(i for i in bench["items"] if i["id"] == r["id"])
        if it.get("procedure"):
            n = len(bench["procedures"][it["procedure"]]["steps"])
            assert len(r["langkah_emas"].splitlines()) == n
            assert len(r["langkah_pdf"].splitlines()) == n
            assert "subjek:" in r["langkah_pdf"]


def test_known_actor_differences_are_flagged(rows):
    by_id = {r["id"]: r for r in rows}
    assert "AKTOR_EMAS_BEDA_SUBJEK: langkah 2 emas=Ketua Program Studi" in by_id["CUTI-P1"]["auto_flags"]
    assert "AKTOR_EMAS_BEDA_SUBJEK: langkah 4 emas=Dosen Wali" in by_id["IRS-P1"]["auto_flags"]
    assert "AKTOR_EMAS_BEDA_SUBJEK: langkah 2 emas=Dosen Wali/Ketua Program Studi" in by_id["UKT-P1"]["auto_flags"]


def test_flags_detect_fact_count_and_number_problems(bench):
    from src.graph_sop.workflow_graph import SOPWorkflowGraph
    b = copy.deepcopy(bench)
    it = next(i for i in b["items"] if i["id"] == "CUTI-P1")
    it["must_include"] = it["must_include"] + [["helikopter"]]
    it["reference"] += " Selesai dalam 7 hari."
    b["procedures"]["CUTI"]["steps"] = b["procedures"]["CUTI"]["steps"][:7]
    flags = "\n".join(mvs.flags_for(it, b, SOPWorkflowGraph()))
    assert "FAKTA_TIDAK_DI_PDF: [helikopter]" in flags
    assert "ANGKA_TIDAK_DI_PDF: '7 hari'" in flags
    assert "JUMLAH_LANGKAH: emas 7 vs PDF 8" in flags


def test_sheet_roundtrip_utf8_bom(tmp_path, rows):
    out = tmp_path / "sheet.csv"
    mvs.write_sheet(out, rows)
    assert out.read_bytes().startswith(b"\xef\xbb\xbf")
    back = mvs.read_sheet(out)
    assert [r["id"] for r in back] == [r["id"] for r in rows]
    assert back[0]["question"] == rows[0]["question"]


def test_sheet_with_semicolons_is_read(tmp_path, rows):
    import csv
    out = tmp_path / "sheet_semicolon.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mvs.COLUMNS, delimiter=";")
        w.writeheader()
        w.writerows(rows[:2])
    assert [r["id"] for r in mvs.read_sheet(out)] == [r["id"] for r in rows[:2]]


def _fill(rows, qid, **kw):
    out = [dict(r) for r in rows]
    for r in out:
        if r["id"] == qid:
            r.update(kw)
    return out


def _setup(tmp_path, rows):
    import re
    b = tmp_path / "bench.json"
    text = BENCH_PATH.read_text(encoding="utf-8")
    text = re.sub(r'"verified_by": "[^"]*", ', "", text)
    text = re.sub(r'"verified_at": "[^"]*", ', "", text)
    text = re.sub(r'"verified": true', '"verified": false', text)
    b.write_text(text, encoding="utf-8")
    s = tmp_path / "sheet.csv"
    mvs.write_sheet(s, rows)
    return b, s


def test_apply_marks_only_clean_yes_rows(tmp_path, rows):
    filled = _fill(rows, "CUTI-A1", **{"verified (y/n)": "y", "verifikator": "Penulis 1"})
    filled = _fill(filled, "LEG-D1", **{"verified (y/n)": "n", "perlu_koreksi": "", "verifikator": "Penulis 1"})
    b, s = _setup(tmp_path, filled)
    before = b.read_text(encoding="utf-8")
    assert av.main(["--sheet", str(s), "--benchmark", str(b), "--date", "2026-10-08"]) == 0
    after_raw = b.read_text(encoding="utf-8")
    after = json.loads(after_raw)
    by_id = {it["id"]: it for it in after["items"]}
    assert by_id["CUTI-A1"]["verified"] is True
    assert by_id["CUTI-A1"]["verified_by"] == "Penulis 1"
    assert by_id["CUTI-A1"]["verified_at"] == "2026-10-08"
    assert sum(1 for it in after["items"] if it.get("verified")) == 1
    assert "verified_by" not in by_id["LEG-D1"]
    # hand-made layout preserved: exactly one line differs
    diff = [(a, c) for a, c in zip(before.splitlines(), after_raw.splitlines()) if a != c]
    assert len(before.splitlines()) == len(after_raw.splitlines()) and len(diff) == 1
    assert '"id": "CUTI-A1"' in diff[0][1]


def test_apply_refuses_yes_with_correction(tmp_path, rows):
    filled = _fill(rows, "CUTI-A1", **{"verified (y/n)": "y", "verifikator": "X"})
    filled = _fill(filled, "CUTI-D1", **{"verified (y/n)": "y", "perlu_koreksi": "referensi kurang 'dokumen pendukung lain'",
                                         "verifikator": "X"})
    b, s = _setup(tmp_path, filled)
    before = b.read_bytes()
    assert av.main(["--sheet", str(s), "--benchmark", str(b)]) == 2
    assert b.read_bytes() == before          # nothing written, not even the clean row


@pytest.mark.parametrize("value", ["-", "tidak", "n", "  "])
def test_correction_placeholders_count_as_empty(tmp_path, rows, value):
    filled = _fill(rows, "CUTI-A1", **{"verified (y/n)": "Ya", "perlu_koreksi": value, "verifikator": "X"})
    b, s = _setup(tmp_path, filled)
    assert av.main(["--sheet", str(s), "--benchmark", str(b)]) == 0


def test_apply_refuses_missing_verifier_bad_value_and_stale_item(tmp_path, rows, bench):
    for bad in (_fill(rows, "CUTI-A1", **{"verified (y/n)": "y"}),
                _fill(rows, "CUTI-A1", **{"verified (y/n)": "mungkin", "verifikator": "X"}),
                _fill(rows, "CUTI-A1", **{"verified (y/n)": "y", "verifikator": "X", "item_sha256": "0" * 16})):
        b, s = _setup(tmp_path, bad)
        before = b.read_bytes()
        assert av.main(["--sheet", str(s), "--benchmark", str(b)]) == 2
        assert b.read_bytes() == before


def test_dry_run_writes_nothing(tmp_path, rows):
    filled = _fill(rows, "CUTI-A1", **{"verified (y/n)": "y", "verifikator": "X"})
    b, s = _setup(tmp_path, filled)
    before = b.read_bytes()
    assert av.main(["--sheet", str(s), "--benchmark", str(b), "--dry-run"]) == 0
    assert b.read_bytes() == before


def test_merge_keeps_human_input_and_resets_changed_items(tmp_path, rows, bench):
    out = tmp_path / "sheet.csv"
    mvs.write_sheet(out, _fill(rows, "CUTI-A1", **{"verified (y/n)": "y", "catatan": "ok", "verifikator": "X"}))
    with pytest.raises(SystemExit):           # refuses to overwrite human input without --merge/--force
        mvs.main(["--out", str(out)])
    b = copy.deepcopy(bench)
    next(i for i in b["items"] if i["id"] == "CUTI-A1")["reference"] += " (dikoreksi)"
    bp = tmp_path / "bench.json"
    bp.write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
    assert mvs.main(["--out", str(out), "--benchmark", str(bp), "--merge"]) == 0
    r = next(x for x in mvs.read_sheet(out) if x["id"] == "CUTI-A1")
    assert r["verified (y/n)"] == "" and r["verifikator"] == "X" and "item berubah" in r["catatan"]
