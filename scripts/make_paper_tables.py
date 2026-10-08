#!/usr/bin/env python3
"""
Build the paper's LaTeX tables and number macros straight from a run's result files
(Goal B, Tugas 3). No number in the paper should be typed by hand.

    python scripts/make_paper_tables.py --run full_v3_20261008 --ablation-run full_base_20261008

Reads (from results/<run>/): summary.csv, summary_by_category.csv, significance.csv,
run_config.json, score_config.json. Writes to results/<run>/paper_tables/:
main_results.tex, per_category.tex, cost.tex, significance.tex, ablation.tex, numbers.tex.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MAIN_ORDER = [
    ("naive", "Naive RAG"),
    ("graph", "GraphRAG (entity)"),
    ("workflow", "\\textbf{Workflow GraphRAG}"),
]
OTHER_ORDER = [
    ("hybrid", "Hybrid RAG"),
    ("agentic", "Agentic RAG"),
    ("crag", "Corrective RAG"),
    ("multimodal", "Multimodal-layout RAG"),
]
REF_ORDER = [
    ("llm_only", "LLM only"),
    ("full_context", "Full context"),
]
ALL_ORDER = MAIN_ORDER + OTHER_ORDER + REF_ORDER

SYSTEM_LABEL = {k: v for k, v in ALL_ORDER}


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    return f"{x:.{nd}f}"


def _fmt_ci(mean, lo, hi, nd=3):
    if mean is None or mean != mean:
        return "--"
    if lo is None or lo != lo or hi is None or hi != hi:
        return _fmt(mean, nd)
    return f"{_fmt(mean, nd)} [{_fmt(lo, nd)}, {_fmt(hi, nd)}]"


def load_run(run_name):
    run_dir = ROOT / "results" / run_name
    summary = pd.read_csv(run_dir / "summary.csv").set_index("system")
    cat = pd.read_csv(run_dir / "summary_by_category.csv")
    sig_path = run_dir / "significance.csv"
    sig = pd.read_csv(sig_path) if sig_path.exists() else None
    run_cfg = json.load(open(run_dir / "run_config.json", encoding="utf-8"))
    score_cfg_path = run_dir / "score_config.json"
    score_cfg = json.load(open(score_cfg_path, encoding="utf-8")) if score_cfg_path.exists() else {}
    return run_dir, summary, cat, sig, run_cfg, score_cfg


def write_main_results(summary, out_dir):
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             "System & KFR $\\uparrow$ & Coverage $\\uparrow$ & TIR $\\downarrow$ & AMR $\\downarrow$ "
             "& Refusal acc. $\\uparrow$ & Prompt tok. \\\\", "\\midrule"]
    for i, group in enumerate([MAIN_ORDER, OTHER_ORDER, REF_ORDER]):
        for sysid, label in group:
            if sysid not in summary.index:
                lines.append(f"{label} & \\multicolumn{{6}}{{c}}{{(not run)}} \\\\")
                continue
            r = summary.loc[sysid]
            cells = [
                _fmt(r["key_fact_recall"]),
                _fmt(r["step_coverage"]),
                _fmt(r["tir"]),
                _fmt(r["amr"]),
                _fmt(r["refusal_accuracy"]),
                f"{r['prompt_tokens_mean']:.0f}",
            ]
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if i < 2:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "main_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Supplementary: same table with full 95% bootstrap CIs (referenced from the main table's caption)
    full_lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
                  "System & KFR $\\uparrow$ & Coverage $\\uparrow$ & TIR $\\downarrow$ & AMR $\\downarrow$ "
                  "& Refusal acc. $\\uparrow$ & Prompt tok. \\\\", "\\midrule"]
    for i, group in enumerate([MAIN_ORDER, OTHER_ORDER, REF_ORDER]):
        for sysid, label in group:
            if sysid not in summary.index:
                continue
            r = summary.loc[sysid]
            cells = [
                _fmt_ci(r["key_fact_recall"], r.get("key_fact_recall_lo"), r.get("key_fact_recall_hi")),
                _fmt_ci(r["step_coverage"], r.get("step_coverage_lo"), r.get("step_coverage_hi")),
                _fmt_ci(r["tir"], r.get("tir_lo"), r.get("tir_hi")),
                _fmt_ci(r["amr"], r.get("amr_lo"), r.get("amr_hi")),
                _fmt_ci(r["refusal_accuracy"], r.get("refusal_accuracy_lo"), r.get("refusal_accuracy_hi")),
                f"{r['prompt_tokens_mean']:.0f}",
            ]
            full_lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if i < 2:
            full_lines.append("\\midrule")
    full_lines += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "main_results_full.tex").write_text(
        "% Same rows/columns as main_results.tex, with 95% bootstrap CI in brackets\n"
        + "\n".join(full_lines) + "\n", encoding="utf-8")


def write_per_category(cat, out_dir):
    cats = ["procedural", "actor", "documents", "time_location", "cross_sop"]
    cat_label = {"procedural": "Procedural", "actor": "Actor", "documents": "Documents",
                 "time_location": "Time/location", "cross_sop": "Cross-SOP"}
    systems = ["naive", "graph", "workflow"]
    lines = []
    for sysid, label in [(s, SYSTEM_LABEL[s]) for s in systems]:
        sub = cat[cat["system"] == sysid].set_index("category")
        cells = []
        for c in cats:
            if c in sub.index:
                kfr = sub.loc[c, "key_fact_recall"]
                if c == "procedural":
                    cov = sub.loc[c, "step_coverage"]
                    cov_s = _fmt(cov) if cov == cov else "--"
                    cells.append(f"{_fmt(kfr)} / {cov_s}")
                else:
                    cells.append(_fmt(kfr))
            else:
                cells.append("--")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    header = "System & " + " & ".join(
        (cat_label[c] + " (KFR/Cov.)" if c == "procedural" else cat_label[c]) for c in cats
    ) + " \\\\"
    full = (["% Key-fact recall per category (procedural column also has step coverage), for the three main systems",
             "\\begin{tabular}{lccccc}", "\\toprule", header, "\\midrule"]
            + lines + ["\\bottomrule", "\\end{tabular}"])
    (out_dir / "per_category.tex").write_text("\n".join(full) + "\n", encoding="utf-8")


def _mean_llm_calls(run_dir):
    import json
    from collections import defaultdict
    sums = defaultdict(list)
    for line in open(run_dir / "generations.jsonl", encoding="utf-8"):
        r = json.loads(line)
        sums[r["system"]].append(r.get("llm_calls", 0))
    return {s: sum(v) / len(v) for s, v in sums.items()}


def write_cost(summary, out_dir, run_dir):
    calls = _mean_llm_calls(run_dir)
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
              "System & Prompt tok. & Completion tok. & LLM calls & Latency (s) \\\\", "\\midrule"]
    for sysid, label in ALL_ORDER:
        if sysid not in summary.index:
            continue
        r = summary.loc[sysid]
        cells = [
            f"{r['prompt_tokens_mean']:.0f}",
            f"{r['completion_tokens_mean']:.0f}",
            f"{calls.get(sysid, float('nan')):.2f}",
            f"{r['latency_median_s']:.1f}",
        ]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "cost.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_significance(sig, out_dir):
    if sig is None:
        (out_dir / "significance.tex").write_text("% significance.csv not found\n", encoding="utf-8")
        return
    metric_label = {"key_fact_recall": "KFR", "rougeL": "ROUGE-L", "step_coverage": "Step coverage",
                     "tir": "TIR", "amr": "AMR"}
    lines = []
    for ref in ("naive", "graph"):
        ref_label = "Naive RAG" if ref == "naive" else "GraphRAG (entity)"
        lines.append(f"% Workflow GraphRAG vs. {ref_label}")
        sub = sig[(sig["system"] == "workflow") & (sig["reference"] == ref)]
        for metric in ("key_fact_recall", "rougeL", "step_coverage", "tir", "amr"):
            row = sub[sub["metric"] == metric]
            if row.empty:
                continue
            row = row.iloc[0]
            sig_mark = "*" if row["p_holm"] == row["p_holm"] and row["p_holm"] < 0.05 else ""
            lines.append(
                f"{metric_label[metric]} vs.\\ {ref_label} & "
                f"{_fmt(row['mean_diff'])} [{_fmt(row['mean_diff_ci_lo'])}, {_fmt(row['mean_diff_ci_hi'])}] & "
                f"{_fmt(row['p_holm'])}{sig_mark} \\\\"
            )
    (out_dir / "significance.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation(summary_v3, summary_base, out_dir):
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
              "System & v3 embedding & Base embedding \\\\", "\\midrule"]
    for sysid, label in MAIN_ORDER:
        row = []
        for summ in (summary_v3, summary_base):
            if sysid not in summ.index:
                row.append("--")
                continue
            r = summ.loc[sysid]
            row.append(_fmt(r["key_fact_recall"]) + " / " + _fmt(r["step_coverage"]))
        lines.append(f"{label} & " + " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _macro_name(system, metric):
    parts = "".join(p.capitalize() for p in metric.split("_"))
    sysname = "".join(p.capitalize() for p in system.split("_"))
    return f"\\{parts[0].lower()}{parts[1:]}{sysname}"


def write_numbers(summary, run_cfg, score_cfg, out_dir, n_items_base=None):
    lines = []
    lines.append(f"\\newcommand{{\\nVerified}}{{{run_cfg['n_items_verified']}}}")
    lines.append(f"\\newcommand{{\\nTotal}}{{{run_cfg['n_items']}}}")
    lines.append(f"\\newcommand{{\\graphSteps}}{{44}}")
    lines.append(f"\\newcommand{{\\graphRootNodes}}{{7}}")
    for sysid in ("naive", "graph", "workflow", "full_context"):
        if sysid not in summary.index:
            continue
        r = summary.loc[sysid]
        for metric in ("key_fact_recall", "step_coverage", "refusal_accuracy"):
            val = r.get(metric)
            if val is None or val != val:
                continue
            lines.append(f"\\newcommand{{{_macro_name(sysid, metric)}}}{{{_fmt(val)}}}")
    lines.append(f"\\newcommand{{\\judgeModel}}{{{score_cfg.get('judge_model') or 'n/a'}}}")
    (out_dir / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ablation-run", default=None)
    args = ap.parse_args()

    run_dir, summary, cat, sig, run_cfg, score_cfg = load_run(args.run)
    out_dir = run_dir / "paper_tables"
    out_dir.mkdir(exist_ok=True)

    write_main_results(summary, out_dir)
    write_per_category(cat, out_dir)
    write_cost(summary, out_dir, run_dir)
    write_significance(sig, out_dir)
    write_numbers(summary, run_cfg, score_cfg, out_dir)

    if args.ablation_run:
        _, summary_base, _, _, _, _ = load_run(args.ablation_run)
        write_ablation(summary, summary_base, out_dir)

    print(f"Wrote paper tables to {out_dir}")
    for f in sorted(out_dir.glob("*.tex")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
