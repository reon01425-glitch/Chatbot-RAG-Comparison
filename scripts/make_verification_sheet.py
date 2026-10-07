#!/usr/bin/env python3
"""
Build the human verification sheet for the benchmark (Goal A, Tugas 3).

    python scripts/make_verification_sheet.py                 # writes benchmark/verification_sheet_v1.csv
    python scripts/make_verification_sheet.py --merge         # regenerate, keep the human columns already filled in

One row per benchmark item with the item text, the gold steps (procedural items), the relevant PDF
text found automatically, and `auto_flags` - mechanical hints of possible benchmark errors:

  FAKTA_TIDAK_DI_PDF        a must_include fact (no alternative) matches the text of the item's SOP PDF(s)
  ANGKA_TIDAK_DI_PDF        a number + unit (menit/hari/jam) in the reference answer is not in the PDF(s)
  JUMLAH_LANGKAH            number of gold steps != number of numbered steps in the PDF
  AKTOR_EMAS_BEDA_SUBJEK    gold actor of a step != grammatical subject of that PDF step
                            (subject = actor in the PDF-reconciled workflow graph, see docs/GRAPH_PDF_RECONCILIATION.md)
  KW_TIDAK_DI_LANGKAH_PDF   a keyword group of a gold step matches nothing in the text of that PDF step

Flags are hints for the human verifier, not verdicts. The script never edits the benchmark.
The CSV is UTF-8 with BOM (Excel/Numbers open it directly); the human columns are
`verified (y/n)`, `perlu_koreksi`, `catatan`, `verifikator`. Apply them with scripts/apply_verification.py.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate_benchmark import actors_in, load_benchmark, match_any  # noqa: E402
from src.graph_sop.workflow_graph import SOPWorkflowGraph  # noqa: E402

DEFAULT_BENCHMARK = ROOT / "benchmark" / "sop_benchmark_v1.json"
DEFAULT_SHEET = ROOT / "benchmark" / "verification_sheet_v1.csv"
DATA_DIR = ROOT / "data"

HUMAN_COLUMNS = ["verified (y/n)", "perlu_koreksi", "catatan", "verifikator"]
COLUMNS = ["id", "category", "sop", "answerable", "question", "reference", "must_include", "langkah_emas",
           "langkah_pdf", "kutipan_pdf", "auto_flags", "item_sha256"] + HUMAN_COLUMNS
VERIFICATION_FIELDS = ("verified", "verified_by", "verified_at")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=None)
def pdf_text(pdf_name: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(DATA_DIR / pdf_name))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


@lru_cache(maxsize=None)
def pdf_segments(pdf_name: str) -> Tuple[Tuple[str, str], ...]:
    """('pembuka', intro) followed by ('langkah n', text of numbered step n incl. its Waktu/Output lines)."""
    text = pdf_text(pdf_name)
    starts = []
    expect = 1
    for m in re.finditer(r"(?m)^\s*(\d{1,2})\.\s", text):
        if int(m.group(1)) == expect:
            starts.append((expect, m.start()))
            expect += 1
    segs = [("pembuka", normalize_ws(text[: starts[0][1]] if starts else text))]
    for i, (n, pos) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        segs.append((f"langkah {n}", normalize_ws(text[pos:end])))
    return tuple(segs)


def item_sha256(item: Dict[str, Any], bench: Dict[str, Any]) -> str:
    """Fingerprint of everything a verifier checks (item content + its gold procedure), minus verification fields."""
    content = {k: v for k, v in item.items() if k not in VERIFICATION_FIELDS}
    if item.get("procedure"):
        content["_procedure"] = bench["procedures"][item["procedure"]]
    blob = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def item_pdfs(item: Dict[str, Any], bench: Dict[str, Any]) -> List[str]:
    codes = [c for c in item.get("sop", "").split("+") if c in bench["procedures"]]
    return [bench["procedures"][c]["sop"] for c in codes]


def fmt_must_include(groups: List[List[str]]) -> str:
    return " ; ".join("[" + " | ".join(g) + "]" for g in groups)


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 3}


def find_quotes(item: Dict[str, Any], pdfs: List[str], max_quotes: int = 3) -> List[str]:
    """
    Rank PDF segments by 2 x (must_include facts matched) + (word overlap with question + reference),
    where words of the question itself count double.
    Items that span several SOPs get the best segment of each of their PDFs first.
    Items without an SOP (e.g. unanswerable '-') are searched in all PDFs and get the 2 best segments.
    """
    groups = item.get("must_include", [])
    q_words = _tokens(item["question"])
    want = q_words | _tokens(item["reference"])
    scored = []
    for pdf in pdfs or sorted(p.name for p in DATA_DIR.glob("*.pdf")):
        for label, seg in pdf_segments(pdf):
            facts = sum(1 for g in groups if match_any(seg.lower(), g))
            words = _tokens(seg)
            score = 2 * facts + len(want & words) + len(q_words & words)
            scored.append((score, pdf, f"[{Path(pdf).stem} | {label}] {seg}"))
    scored.sort(key=lambda x: -x[0])
    if not pdfs:
        return [q for _, _, q in scored[:2]]
    max_quotes = max(max_quotes, len(pdfs) + 1)
    picked = []
    for pdf in pdfs:                       # one per SOP first
        best = next((q for s, p, q in scored if p == pdf and s > 0), None)
        if best:
            picked.append(best)
    for s, _, q in scored:                 # then the best remaining ones
        if len(picked) >= max_quotes:
            break
        if s > 0 and q not in picked:
            picked.append(q)
    return picked


def pdf_steps_with_subject(pdf: str, wg: SOPWorkflowGraph) -> List[Dict[str, Any]]:
    sop = next(s for s in wg.sop_metadata.values() if s["source_pdf"] == pdf)
    return sop["steps"]


def flags_for(item: Dict[str, Any], bench: Dict[str, Any], wg: SOPWorkflowGraph) -> List[str]:
    flags = []
    pdfs = item_pdfs(item, bench)
    corpus = " ".join(normalize_ws(pdf_text(p)) for p in pdfs).lower()
    if pdfs:
        for g in item.get("must_include", []):
            if not match_any(corpus, g):
                flags.append(f"FAKTA_TIDAK_DI_PDF: [{' | '.join(g)}]")
        for n, unit in re.findall(r"(\d+)\s*(menit|hari|jam)", item["reference"].lower()):
            if not re.search(rf"\b{n}\s*(\([a-z]+\)\s*)?,?\s*{unit}", corpus):
                flags.append(f"ANGKA_TIDAK_DI_PDF: '{n} {unit}'")
    proc = item.get("procedure")
    if proc:
        gold = bench["procedures"][proc]
        pdf = gold["sop"]
        steps = pdf_steps_with_subject(pdf, wg)
        segs = dict(pdf_segments(pdf))
        if len(gold["steps"]) != len(steps):
            flags.append(f"JUMLAH_LANGKAH: emas {len(gold['steps'])} vs PDF {len(steps)}")
        for st in gold["steps"]:
            if st["n"] > len(steps):
                continue
            subject = steps[st["n"] - 1]["actor"]
            named, _ = actors_in(subject.lower(), bench["actors"])
            if named != set(st["actor"]):
                flags.append(f"AKTOR_EMAS_BEDA_SUBJEK: langkah {st['n']} emas={'/'.join(st['actor'])}"
                             f" (also_ok={'/'.join(st['also_ok']) or '-'}) vs subjek PDF={subject}")
            seg = segs.get(f"langkah {st['n']}", "").lower()
            for g in st["kw"]:
                if not match_any(seg, g):
                    flags.append(f"KW_TIDAK_DI_LANGKAH_PDF: langkah {st['n']} [{' | '.join(g)}]")
    return flags


def build_rows(bench: Dict[str, Any]) -> List[Dict[str, str]]:
    wg = SOPWorkflowGraph()
    rows = []
    for it in bench["items"]:
        pdfs = item_pdfs(it, bench)
        gold_txt, pdf_txt = "", ""
        proc = it.get("procedure")
        if proc:
            gold = bench["procedures"][proc]
            gold_txt = "\n".join(
                f"{st['n']}. {' / '.join(st['actor'])}" + (f" (also_ok: {', '.join(st['also_ok'])})" if st["also_ok"] else "")
                for st in gold["steps"])
            pdf_txt = "\n".join(f"{s['step_num']}. subjek: {s['actor']}" for s in pdf_steps_with_subject(gold["sop"], wg))
            quotes = [f"[{Path(gold['sop']).stem} | {label}] {seg}" for label, seg in pdf_segments(gold["sop"])]
        else:
            quotes = find_quotes(it, pdfs)
        rows.append({
            "id": it["id"], "category": it["category"], "sop": it.get("sop", ""),
            "answerable": "ya" if it.get("answerable", True) else "tidak",
            "question": it["question"], "reference": it["reference"],
            "must_include": fmt_must_include(it.get("must_include", [])),
            "langkah_emas": gold_txt, "langkah_pdf": pdf_txt, "kutipan_pdf": "\n".join(quotes),
            "auto_flags": "\n".join(flags_for(it, bench, wg)),
            "item_sha256": item_sha256(it, bench),
            **{c: "" for c in HUMAN_COLUMNS},
        })
    return rows


def read_sheet(path: Path) -> List[Dict[str, str]]:
    raw = path.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(raw.split("\n", 1)[0], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return list(csv.DictReader(raw.splitlines(keepends=True), dialect=dialect))


def write_sheet(path: Path, rows: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    ap.add_argument("--out", default=str(DEFAULT_SHEET))
    ap.add_argument("--merge", action="store_true",
                    help="keep human columns from an existing sheet; items whose content changed get their "
                         "'verified (y/n)' cleared and a note in 'catatan'")
    ap.add_argument("--force", action="store_true", help="overwrite an existing sheet that already has human input")
    args = ap.parse_args(argv)

    bench = load_benchmark(Path(args.benchmark))
    rows = build_rows(bench)
    out = Path(args.out)
    if out.exists():
        old = {r["id"]: r for r in read_sheet(out)}
        has_input = any((r.get(c) or "").strip() for r in old.values() for c in HUMAN_COLUMNS)
        if args.merge:
            for r in rows:
                o = old.get(r["id"])
                if not o:
                    continue
                for c in HUMAN_COLUMNS:
                    r[c] = o.get(c, "") or ""
                if o.get("item_sha256") != r["item_sha256"] and r["verified (y/n)"].strip():
                    r["verified (y/n)"] = ""
                    r["catatan"] = (r["catatan"] + " " if r["catatan"] else "") + \
                        "[item berubah sejak lembar sebelumnya; periksa ulang]"
        elif has_input and not args.force:
            raise SystemExit(f"{out} already contains human input. Use --merge to keep it or --force to overwrite.")
    write_sheet(out, rows)
    n_flag = sum(1 for r in rows if r["auto_flags"])
    print(f"Wrote {out} ({len(rows)} items, {n_flag} with auto_flags).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
