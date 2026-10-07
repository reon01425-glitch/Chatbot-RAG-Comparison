"""
Paired significance against more than one reference system (Goal A, Tugas 2).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate_benchmark as eb  # noqa: E402

ORDER = ["naive", "hybrid", "graph", "workflow"]
METRICS = ["key_fact_recall", "rougeL", "step_coverage", "tir", "amr"]
NEW_COLUMNS = ["mean_diff_ci_lo", "mean_diff_ci_hi"]


def synthetic_scores(seed: int = 7, n_q: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s_i, s in enumerate(ORDER):
        for q in range(n_q):
            row = {"system": s, "qid": f"Q{q:02d}"}
            for m in METRICS:
                v = float(np.clip(rng.normal(0.5 + 0.05 * s_i, 0.2), 0, 1))
                if m in ("step_coverage", "tir", "amr") and q % 3 == 0:
                    v = float("nan")          # non-procedural items have no step metrics
                row[m] = v
            rows.append(row)
    return pd.DataFrame(rows)


def legacy_significance(df: pd.DataFrame, order, ref: str):
    """Verbatim copy of the significance block in cmd_score at commit d421b93 (single reference)."""
    sig_rows = []
    if ref in set(df["system"]):
        from scipy.stats import wilcoxon
        test_metrics = [m for m in ["key_fact_recall", "rougeL", "bertscore_f1", "faithfulness", "answer_relevancy",
                                    "step_coverage", "tir", "amr"] if m in df.columns]
        for m in test_metrics:
            base = df[df["system"] == ref].set_index("qid")[m]
            block = []
            for s in order:
                if s == ref:
                    continue
                other = df[df["system"] == s].set_index("qid")[m]
                pair = pd.concat([base, other], axis=1, keys=["ref", "sys"]).dropna()
                n = len(pair)
                diff = (pair["sys"] - pair["ref"])
                p = float("nan")
                if n >= 6 and (diff != 0).any():
                    try:
                        p = float(wilcoxon(pair["sys"], pair["ref"], zero_method="zsplit").pvalue)
                    except ValueError:
                        p = float("nan")
                block.append({"metric": m, "system": s, "reference": ref, "n_pairs": n,
                              "mean_diff": float(diff.mean()) if n else float("nan"),
                              "median_diff": float(diff.median()) if n else float("nan"), "p_wilcoxon": p})
            adj = eb.holm([b["p_wilcoxon"] for b in block])
            for b, a in zip(block, adj):
                b["p_holm"] = a
            sig_rows.extend(block)
    return sig_rows


def test_parse_references():
    assert eb.parse_references("naive") == ["naive"]
    assert eb.parse_references("naive, graph,naive") == ["naive", "graph"]
    with pytest.raises(SystemExit):
        eb.parse_references("naive,nope")


def test_default_reference_reproduces_legacy_file_content():
    df = synthetic_scores()
    old_csv = pd.DataFrame(legacy_significance(df, ORDER, "naive")).to_csv(index=False)
    new = pd.DataFrame(eb.paired_significance(df, ORDER, eb.parse_references("naive")))
    # the only change for the default is two CI columns appended at the end
    assert list(new.columns[-2:]) == NEW_COLUMNS
    assert new.drop(columns=NEW_COLUMNS).to_csv(index=False) == old_csv


def test_two_references_give_two_independent_blocks():
    df = synthetic_scores()
    rows = eb.paired_significance(df, ORDER, ["naive", "graph"])
    out = pd.DataFrame(rows)
    assert list(dict.fromkeys(out["reference"])) == ["naive", "graph"]
    naive_block = out[out["reference"] == "naive"]
    graph_block = out[out["reference"] == "graph"]
    assert set(naive_block["system"]) == {"hybrid", "graph", "workflow"}
    assert set(graph_block["system"]) == {"naive", "hybrid", "workflow"}
    # adding a second reference leaves the first block untouched
    alone = pd.DataFrame(eb.paired_significance(df, ORDER, ["naive"]))
    pd.testing.assert_frame_equal(naive_block.reset_index(drop=True), alone.reset_index(drop=True))
    # graph-vs-naive is the sign-flipped naive-vs-graph pair
    a = naive_block[(naive_block.metric == "rougeL") & (naive_block.system == "graph")].iloc[0]
    b = graph_block[(graph_block.metric == "rougeL") & (graph_block.system == "naive")].iloc[0]
    assert a.mean_diff == pytest.approx(-b.mean_diff)
    assert a.p_wilcoxon == pytest.approx(b.p_wilcoxon)


def test_missing_reference_is_skipped():
    df = synthetic_scores()
    rows = eb.paired_significance(df, ORDER, ["crag", "naive"])
    assert {r["reference"] for r in rows} == {"naive"}


def test_holm_hand_computed():
    # sorted 0.01, 0.03, 0.04 -> 3*0.01=0.03, 2*0.03=0.06, max(0.06, 1*0.04)=0.06
    assert eb.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_is_applied_per_reference_block(monkeypatch):
    """Fixed raw p-values; Holm must be computed within each (reference, metric) block of 3, not over all 6."""
    import scipy.stats

    df = synthetic_scores()[["system", "qid", "rougeL"]]
    raw = iter([0.01, 0.04, 0.03,     # vs naive: hybrid, graph, workflow
                0.02, 0.50, 0.01])    # vs graph: naive, hybrid, workflow

    class _Res:
        def __init__(self, p):
            self.pvalue = p

    monkeypatch.setattr(scipy.stats, "wilcoxon", lambda *a, **k: _Res(next(raw)))
    rows = eb.paired_significance(df, ORDER, ["naive", "graph"])
    got = {(r["reference"], r["system"]): r["p_holm"] for r in rows}
    # block naive: sorted 0.01 (hybrid), 0.03 (workflow), 0.04 (graph) -> 0.03, 0.06, 0.06
    assert got[("naive", "hybrid")] == pytest.approx(0.03)
    assert got[("naive", "workflow")] == pytest.approx(0.06)
    assert got[("naive", "graph")] == pytest.approx(0.06)
    # block graph: sorted 0.01 (workflow), 0.02 (naive), 0.50 (hybrid) -> 0.03, 0.04, 0.50
    assert got[("graph", "workflow")] == pytest.approx(0.03)
    assert got[("graph", "naive")] == pytest.approx(0.04)
    assert got[("graph", "hybrid")] == pytest.approx(0.50)
    # pooling all six would have given 6 * 0.01 = 0.06 to the smallest p-value instead
    assert got[("graph", "workflow")] != pytest.approx(0.06)


def test_paired_mean_difference_ci_is_deterministic_and_brackets_mean():
    df = synthetic_scores()
    r1 = eb.paired_significance(df, ORDER, ["naive", "graph"], seed=42)
    r2 = eb.paired_significance(df, ORDER, ["naive", "graph"], seed=42)
    assert pd.DataFrame(r1).equals(pd.DataFrame(r2))
    for r in r1:
        if r["n_pairs"] >= 2:
            assert r["mean_diff_ci_lo"] <= r["mean_diff"] <= r["mean_diff_ci_hi"]
    # matches a direct 10,000-resample bootstrap of the per-question differences
    base = df[df.system == "naive"].set_index("qid")["rougeL"]
    other = df[df.system == "workflow"].set_index("qid")["rougeL"]
    _, lo, hi, _ = eb.bootstrap_ci((other - base).dropna().tolist(), n_boot=10000, seed=42)
    row = next(r for r in r1 if r["reference"] == "naive" and r["system"] == "workflow" and r["metric"] == "rougeL")
    assert (row["mean_diff_ci_lo"], row["mean_diff_ci_hi"]) == pytest.approx((lo, hi))


def test_significance_table_marks_each_reference_separately(monkeypatch):
    import scipy.stats

    df = synthetic_scores()[["system", "qid", "rougeL"]]
    raw = iter([0.001, 0.9, 0.9, 0.9, 0.9, 0.001])

    class _Res:
        def __init__(self, p):
            self.pvalue = p

    monkeypatch.setattr(scipy.stats, "wilcoxon", lambda *a, **k: _Res(next(raw)))
    rows = eb.paired_significance(df, ORDER, ["naive", "graph"])
    tex = eb.significance_table_tex(rows, ["naive", "graph"])
    assert "vs.\\ Naive RAG (Baseline)" in tex and "vs.\\ GraphRAG (Entity Expansion)" in tex
    naive_part, graph_part = tex.split("vs.\\ GraphRAG (Entity Expansion)")
    hybrid_line = next(l for l in naive_part.splitlines() if l.startswith("Hybrid RAG"))
    workflow_line = next(l for l in graph_part.splitlines() if l.startswith("Workflow GraphRAG"))
    assert "$^{*}$" in hybrid_line          # hybrid vs naive: p_holm = 0.003
    assert "$^{*}$" in workflow_line        # workflow vs graph: p_holm = 0.003
    assert tex.count("$^{*}$") == 2 + 1     # two marked cells + the caption legend
