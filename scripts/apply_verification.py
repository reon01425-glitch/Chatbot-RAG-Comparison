#!/usr/bin/env python3
"""
Apply a filled-in verification sheet to the benchmark (Goal A, Tugas 3).

    python scripts/apply_verification.py --dry-run     # show what would change
    python scripts/apply_verification.py               # write benchmark/sop_benchmark_v1.json

Rules
  * An item gets "verified": true (plus "verified_by" and "verified_at") only when its row has
    `verified (y/n)` = y AND `perlu_koreksi` is empty.
    Accepted values: y / ya / yes  and  n / tidak / no / (empty).
    `perlu_koreksi` counts as empty when blank or one of: -, n, no, tidak, t.
  * Refuses to write anything (exit code 2) if
      - a row is marked y and also has something in `perlu_koreksi`;
      - a y row has no `verifikator`;
      - a y row's `item_sha256` differs from the current item (the item was edited after the sheet
        was made: regenerate it with `make_verification_sheet.py --merge` and check the row again);
      - a `verified (y/n)` value is not recognised, an id is unknown, or an id appears twice.
  * Only the fields verified / verified_by / verified_at are touched. Question, reference, must_include
    and procedures are never changed; corrections are made by a human directly in the JSON.
    Items marked n are left as they are (the script never sets verified back to false).
"""
import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_verification_sheet import (  # noqa: E402
    DEFAULT_BENCHMARK, DEFAULT_SHEET, VERIFICATION_FIELDS, item_sha256, read_sheet)

YES = {"y", "ya", "yes"}
NO = {"n", "tidak", "no", ""}
EMPTY_CORRECTION = {"", "-", "n", "no", "tidak", "t"}


class SheetError(Exception):
    pass


def plan_updates(bench: Dict[str, Any], rows: List[Dict[str, str]], today: str) -> Dict[str, Dict[str, Any]]:
    """Validate the sheet and return {item_id: {verified, verified_by, verified_at}} for the items to verify."""
    items = {it["id"]: it for it in bench["items"]}
    errors, seen, updates = [], set(), {}
    for i, r in enumerate(rows, start=2):  # row 1 is the header
        qid = (r.get("id") or "").strip()
        if not qid:
            continue
        if qid in seen:
            errors.append(f"row {i}: duplicate id {qid}")
            continue
        seen.add(qid)
        if qid not in items:
            errors.append(f"row {i}: unknown id {qid}")
            continue
        v = (r.get("verified (y/n)") or "").strip().lower()
        corr = (r.get("perlu_koreksi") or "").strip()
        who = (r.get("verifikator") or "").strip()
        if v not in YES | NO:
            errors.append(f"row {i} ({qid}): 'verified (y/n)' = {r.get('verified (y/n)')!r} is not y/n")
            continue
        if v not in YES:
            continue
        if corr.lower() not in EMPTY_CORRECTION:
            errors.append(f"row {i} ({qid}): marked verified = y but perlu_koreksi = {corr!r}")
            continue
        if not who:
            errors.append(f"row {i} ({qid}): marked verified = y but 'verifikator' is empty")
            continue
        current = item_sha256(items[qid], bench)
        if (r.get("item_sha256") or "").strip() != current:
            errors.append(f"row {i} ({qid}): item changed since the sheet was made "
                          f"(sheet {r.get('item_sha256')!r}, now {current!r}); regenerate with --merge and re-check")
            continue
        updates[qid] = {"verified": True, "verified_by": who, "verified_at": today}
    if errors:
        raise SheetError("\n".join(errors))
    return updates


def _render_fields(f: Dict[str, Any]) -> str:
    return (f'"verified": true, "verified_by": {json.dumps(f["verified_by"], ensure_ascii=False)}, '
            f'"verified_at": {json.dumps(f["verified_at"])}')


def apply_to_text(raw: str, updates: Dict[str, Dict[str, Any]]) -> str:
    """
    Edit only the verification fields on each item's line, keeping the hand-made layout of the file.
    Falls back to a full re-serialisation if an item line cannot be found unambiguously.
    """
    text = raw
    for qid, f in updates.items():
        pat = re.compile(
            r'(\{"id": ' + re.escape(json.dumps(qid)) + r',[^\n]*?)"verified": (?:true|false)'
            r'(?:, "verified_by": "(?:[^"\\]|\\.)*")?(?:, "verified_at": "[^"]*")?')
        text, n = pat.subn(lambda m: m.group(1) + _render_fields(f), text)
        if n != 1:
            data = json.loads(raw)
            for it in data["items"]:
                if it["id"] in updates:
                    it.update(updates[it["id"]])
            print("NOTE: benchmark layout not recognised; re-serialising the whole file (indent=2).")
            return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    return text


def check_only_verification_changed(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    strip = lambda it: {k: v for k, v in it.items() if k not in VERIFICATION_FIELDS}  # noqa: E731
    b_rest = {k: v for k, v in before.items() if k != "items"}
    a_rest = {k: v for k, v in after.items() if k != "items"}
    if b_rest != a_rest or [strip(i) for i in before["items"]] != [strip(i) for i in after["items"]]:
        raise SheetError("internal error: benchmark content other than verification fields would change")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", default=str(DEFAULT_SHEET))
    ap.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    ap.add_argument("--date", default=_dt.date.today().isoformat(), help="value for verified_at (default: today)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    bench_path = Path(args.benchmark)
    raw = bench_path.read_text(encoding="utf-8")
    bench = json.loads(raw)
    rows = read_sheet(Path(args.sheet))
    try:
        updates = plan_updates(bench, rows, args.date)
    except SheetError as e:
        print("Refusing to apply the verification sheet:\n" + str(e), file=sys.stderr)
        return 2

    new_raw = apply_to_text(raw, updates)
    after = json.loads(new_raw)
    check_only_verification_changed(bench, after)
    marked = {r["id"].strip() for r in rows if (r.get("verified (y/n)") or "").strip().lower() in NO and r.get("id")}
    stale = sorted(it["id"] for it in after["items"] if it.get("verified") and it["id"] in marked)
    n_ver = sum(1 for it in after["items"] if it.get("verified"))
    print(f"{len(updates)} item(s) to mark verified; benchmark would then have {n_ver}/{len(after['items'])} verified.")
    if stale:
        print(f"WARNING: already verified in the JSON but not 'y' in the sheet (left unchanged): {', '.join(stale)}")
    if args.dry_run:
        for qid, f in updates.items():
            print(f"  {qid}: verified_by={f['verified_by']!r} verified_at={f['verified_at']}")
        return 0
    if new_raw != raw:
        bench_path.write_text(new_raw, encoding="utf-8")
        print(f"Wrote {bench_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
