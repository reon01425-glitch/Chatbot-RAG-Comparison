#!/usr/bin/env python3
"""
Compute agreement between the automatic step-alignment audit and a filled-in human
audit sheet (Goal B, Tugas 5; Goal C uses this after the sheet is labelled).

    python scripts/audit_agreement.py --run full_v3_20261008

Reads results/<run>/audit_sheet.csv once its human columns are filled in
(`human_detected (y/n)` and `human_actor_status (correct/mismatch/unattributed)`).
Writes results/<run>/audit_agreement.json and results/<run>/paper_tables/audit.tex.

Agreement/kappa are computed only over rows with a non-empty human label; rows left
blank are reported as "not yet labelled" rather than silently dropped.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DETECTED_YES = {"y", "ya", "yes", "true", "1"}
DETECTED_NO = {"n", "tidak", "no", "false", "0"}


def _norm_bool(x):
    x = (x or "").strip().lower()
    if x in DETECTED_YES:
        return True
    if x in DETECTED_NO:
        return False
    return None


def cohens_kappa(pairs):
    """pairs: list of (auto_label, human_label), both hashable, same label space."""
    labels = sorted(set(a for a, _ in pairs) | set(h for _, h in pairs))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(pairs)
    if n == 0:
        return None
    confusion = [[0] * len(labels) for _ in labels]
    for a, h in pairs:
        confusion[idx[a]][idx[h]] += 1
    agree = sum(confusion[i][i] for i in range(len(labels))) / n
    row_marg = [sum(confusion[i]) / n for i in range(len(labels))]
    col_marg = [sum(confusion[i][j] for i in range(len(labels))) / n for j in range(len(labels))]
    expected = sum(row_marg[i] * col_marg[i] for i in range(len(labels)))
    if expected == 1.0:
        return 1.0
    return (agree - expected) / (1 - expected)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    run_dir = ROOT / "results" / args.run
    sheet_path = run_dir / "audit_sheet.csv"
    rows = list(csv.DictReader(open(sheet_path, encoding="utf-8-sig")))

    detected_pairs, actor_pairs = [], []
    n_labelled_detected, n_labelled_actor = 0, 0
    for r in rows:
        auto_detected = _norm_bool(r["detected"])
        human_detected = _norm_bool(r["human_detected (y/n)"])
        if human_detected is not None:
            n_labelled_detected += 1
            detected_pairs.append((auto_detected, human_detected))

        human_actor = (r["human_actor_status (correct/mismatch/unattributed)"] or "").strip().lower()
        if human_actor in ("correct", "mismatch", "unattributed"):
            n_labelled_actor += 1
            actor_pairs.append((r["actor_status"].strip().lower(), human_actor))

    n_total = len(rows)
    detected_agree = (sum(1 for a, h in detected_pairs if a == h) / len(detected_pairs)
                       if detected_pairs else None)
    actor_agree = (sum(1 for a, h in actor_pairs if a == h) / len(actor_pairs)
                    if actor_pairs else None)
    detected_kappa = cohens_kappa(detected_pairs) if detected_pairs else None
    actor_kappa = cohens_kappa(actor_pairs) if actor_pairs else None

    # TIR/AMR recomputed from human labels, per (system, qid), for comparison with the
    # automatic scorer. TIR needs step order, which this flat sheet does not carry per
    # question when sub-sampled, so we report AMR (actor) and the detected rate (coverage
    # proxy) from human labels; a full TIR recomputation needs the unsampled per-question
    # ordering and is left to Goal C if the sheet is extended to the full 225 rows.
    human_amr = None
    if actor_pairs:
        mismatches = sum(1 for _, h in actor_pairs if h == "mismatch")
        attributable = sum(1 for _, h in actor_pairs if h in ("correct", "mismatch"))
        human_amr = mismatches / attributable if attributable else None

    result = {
        "run": args.run,
        "n_rows_total": n_total,
        "n_rows_labelled_detected": n_labelled_detected,
        "n_rows_labelled_actor": n_labelled_actor,
        "detected_agreement": detected_agree,
        "detected_kappa": detected_kappa,
        "actor_status_agreement": actor_agree,
        "actor_status_kappa": actor_kappa,
        "amr_from_human_labels": human_amr,
        "fully_labelled": n_labelled_detected == n_total and n_labelled_actor == n_total,
    }
    json.dump(result, open(run_dir / "audit_agreement.json", "w", encoding="utf-8"), indent=2)

    out_dir = run_dir / "paper_tables"
    out_dir.mkdir(exist_ok=True)
    if n_labelled_detected == 0 and n_labelled_actor == 0:
        tex = "% audit_sheet.csv has no human labels yet -- run after Goal C labelling\n"
    else:
        def _fmt(x):
            return "--" if x is None else f"{x:.3f}"
        tex = (
            f"Step detected (agreement / $\\kappa$) & {_fmt(detected_agree)} / {_fmt(detected_kappa)} "
            f"& {n_labelled_detected}/{n_total} rows labelled \\\\\n"
            f"Actor status (agreement / $\\kappa$) & {_fmt(actor_agree)} / {_fmt(actor_kappa)} "
            f"& {n_labelled_actor}/{n_total} rows labelled \\\\\n"
        )
    (out_dir / "audit.tex").write_text(tex, encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Wrote {run_dir / 'audit_agreement.json'} and {out_dir / 'audit.tex'}")


if __name__ == "__main__":
    main()
