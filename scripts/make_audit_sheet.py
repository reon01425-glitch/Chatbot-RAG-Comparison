#!/usr/bin/env python3
"""
Build the manual TIR/AMR audit sheet (Goal B, Tugas 5) from a run's step_alignment_audit.csv.

    python scripts/make_audit_sheet.py --run full_v3_20261008

Sample: all verified procedural questions, for the three main systems (naive, graph, workflow).
If that is more than 60 step rows, take a stratified random sample (by system) with a fixed seed.
Writes results/<run>/audit_sheet.csv (empty human columns) and
results/<run>/audit_sheet_answers.txt (full answer text per system/qid, for context).
"""
import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN_SYSTEMS = ["naive", "graph", "workflow"]
MAX_ROWS = 60
SEED = 42

HUMAN_COLUMNS = ["human_detected (y/n)", "human_actor_status (correct/mismatch/unattributed)", "catatan"]
COLUMNS = ["system", "qid", "step_n", "gold_actor", "gold_also_ok", "gold_keywords",
           "matched_unit", "detected", "actor_status", "actors_found"] + HUMAN_COLUMNS


def load_benchmark():
    bench = json.load(open(ROOT / "benchmark" / "sop_benchmark_v1.json", encoding="utf-8"))
    items = {it["id"]: it for it in bench["items"]}
    return bench, items


def gold_step_lookup(bench, items):
    lookup = {}
    for it in items.values():
        proc = it.get("procedure")
        if not proc:
            continue
        for step in bench["procedures"][proc]["steps"]:
            lookup[(it["id"], step["n"])] = step
    return lookup


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    run_dir = ROOT / "results" / args.run
    bench, items = load_benchmark()
    gold = gold_step_lookup(bench, items)

    rows = list(csv.DictReader(open(run_dir / "step_alignment_audit.csv", encoding="utf-8")))
    rows = [r for r in rows if r["system"] in MAIN_SYSTEMS]

    if len(rows) > args.max_rows:
        rng = random.Random(args.seed)
        by_system = {}
        for r in rows:
            by_system.setdefault(r["system"], []).append(r)
        per_system = args.max_rows // len(MAIN_SYSTEMS)
        sampled = []
        for sysid in MAIN_SYSTEMS:
            pool = by_system.get(sysid, [])
            sampled.extend(rng.sample(pool, min(per_system, len(pool))))
        rows = sampled

    rows.sort(key=lambda r: (r["qid"], r["system"], int(r["n"])))

    gens = {}
    for line in open(run_dir / "generations.jsonl", encoding="utf-8"):
        g = json.loads(line)
        gens[(g["system"], g["qid"])] = g

    out_rows = []
    for r in rows:
        step = gold.get((r["qid"], int(r["n"])))
        gold_actor = ";".join(step["actor"]) if step else ""
        gold_also_ok = ";".join(step.get("also_ok", [])) if step else ""
        gold_kw = " | ".join(",".join(g) for g in step.get("kw", [])) if step else ""
        out_rows.append({
            "system": r["system"], "qid": r["qid"], "step_n": r["n"],
            "gold_actor": gold_actor, "gold_also_ok": gold_also_ok, "gold_keywords": gold_kw,
            "matched_unit": r["unit"], "detected": r["detected"], "actor_status": r["actor_status"],
            "actors_found": r["actors_found"],
            **{c: "" for c in HUMAN_COLUMNS},
        })

    out_path = run_dir / "audit_sheet.csv"
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(out_rows)

    answers_path = run_dir / "audit_sheet_answers.txt"
    with open(answers_path, "w", encoding="utf-8") as f:
        seen = set()
        for r in out_rows:
            key = (r["system"], r["qid"])
            if key in seen:
                continue
            seen.add(key)
            g = gens.get(key)
            f.write(f"=== {r['system']} / {r['qid']} ===\n")
            f.write((g["answer"] if g else "(not found)") + "\n\n")

    print(f"Wrote {len(out_rows)} rows to {out_path}")
    print(f"Wrote full answers for {len(seen)} (system, qid) pairs to {answers_path}")


if __name__ == "__main__":
    main()
