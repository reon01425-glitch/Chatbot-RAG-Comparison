#!/usr/bin/env python3
"""Diagnosis script for H-Tree parent SOP selection algorithms on dev set."""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from src.manifest import resolve_embedding_model, resolve_chroma_path


def select_parent_sop(scored_results, variant: str = "top1") -> str:
    """Select parent SOP from scored leaf chunks based on variant."""
    if not scored_results:
        return ""
    
    if variant == "top1":
        return os.path.basename(scored_results[0][0].metadata.get("source", ""))
    
    elif variant == "vote":
        votes = defaultdict(int)
        best_score_per_sop = defaultdict(float)
        for doc, score in scored_results:
            sop = os.path.basename(doc.metadata.get("source", ""))
            votes[sop] += 1
            if score > best_score_per_sop[sop]:
                best_score_per_sop[sop] = score
        
        # Sort by vote count desc, then by best individual score desc
        sorted_sops = sorted(votes.keys(), key=lambda s: (votes[s], best_score_per_sop[s]), reverse=True)
        return sorted_sops[0]
        
    elif variant == "sum":
        sums = defaultdict(float)
        best_score_per_sop = defaultdict(float)
        for doc, score in scored_results:
            sop = os.path.basename(doc.metadata.get("source", ""))
            sums[sop] += score
            if score > best_score_per_sop[sop]:
                best_score_per_sop[sop] = score
                
        sorted_sops = sorted(sums.keys(), key=lambda s: (sums[s], best_score_per_sop[s]), reverse=True)
        return sorted_sops[0]
        
    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_diagnosis(model_alias: str, k: int = 3, dev_path: str = "benchmark/dev_htree_v1.json"):
    model_name = resolve_embedding_model(model_alias)
    chroma_dir = resolve_chroma_path(model_alias)
    
    print(f"\n=======================================================")
    print(f"H-TREE DIAGNOSIS ON DEV SET: Model = {model_alias} ({model_name})")
    print(f"Chroma Directory: {chroma_dir} | k = {k}")
    print(f"=======================================================")

    dev_data = json.load(open(ROOT / dev_path, encoding="utf-8"))
    items = dev_data["items"]

    emb_fn = HuggingFaceEmbeddings(model_name=model_name)
    htree_db = Chroma(collection_name="htree_leaves", persist_directory=str(ROOT / chroma_dir), embedding_function=emb_fn)
    main_db = Chroma(collection_name="langchain", persist_directory=str(ROOT / chroma_dir), embedding_function=emb_fn)

    results = []
    correct_counts = {"top1": 0, "vote": 0, "sum": 0, "v0": 0}

    for it in items:
        qid = it["id"]
        q = it["question"]
        expected = it["expected_sop"]

        # H-Tree Leaf Retrieval
        leaf_docs = htree_db.similarity_search(q, k=k)
        q_emb = emb_fn.embed_query(q)
        
        scored_leaves = []
        for d in leaf_docs:
            d_emb = emb_fn.embed_query(d.page_content)
            sim = float(cosine_similarity([q_emb], [d_emb])[0][0])
            scored_leaves.append((d, sim))
        scored_leaves.sort(key=lambda x: x[1], reverse=True)

        # H-Tree v0 (Document-level chunk retrieval)
        v0_docs = main_db.similarity_search(q, k=k)
        v0_scored = []
        for d in v0_docs:
            d_emb = emb_fn.embed_query(d.page_content)
            sim = float(cosine_similarity([q_emb], [d_emb])[0][0])
            v0_scored.append((d, sim))
        v0_scored.sort(key=lambda x: x[1], reverse=True)
        v0_choice = os.path.basename(v0_scored[0][0].metadata.get("source", "")) if v0_scored else ""

        pred_top1 = select_parent_sop(scored_leaves, "top1")
        pred_vote = select_parent_sop(scored_leaves, "vote")
        pred_sum = select_parent_sop(scored_leaves, "sum")

        ok_top1 = (pred_top1 == expected)
        ok_vote = (pred_vote == expected)
        ok_sum = (pred_sum == expected)
        ok_v0 = (v0_choice == expected)

        if ok_top1: correct_counts["top1"] += 1
        if ok_vote: correct_counts["vote"] += 1
        if ok_sum: correct_counts["sum"] += 1
        if ok_v0: correct_counts["v0"] += 1

        top_leaf_info = [
            f"{os.path.basename(d.metadata.get('source',''))}:{d.metadata.get('step_num', 'ov')} ({s:.3f})"
            for d, s in scored_leaves
        ]

        results.append({
            "id": qid,
            "expected": expected,
            "top1": pred_top1,
            "vote": pred_vote,
            "sum": pred_sum,
            "v0": v0_choice,
            "ok_top1": ok_top1,
            "ok_vote": ok_vote,
            "ok_sum": ok_sum,
            "ok_v0": ok_v0,
            "leaves": top_leaf_info,
        })

        status_str = f"top1:{'✓' if ok_top1 else '✗'} vote:{'✓' if ok_vote else '✗'} sum:{'✓' if ok_sum else '✗'} v0:{'✓' if ok_v0 else '✗'}"
        print(f"[{qid}] {status_str} | exp: {expected[:15]}... | top1: {pred_top1[:15]}... | vote: {pred_vote[:15]}... | sum: {pred_sum[:15]}...")
        print(f"       Leaves: {', '.join(top_leaf_info)}")

    n = len(items)
    print(f"\n--- ACCURACY SUMMARY ({model_alias}, N={n}) ---")
    print(f"  htree (top1):    {correct_counts['top1']}/{n} ({correct_counts['top1']/n*100:.1f}%)")
    print(f"  htree (vote):    {correct_counts['vote']}/{n} ({correct_counts['vote']/n*100:.1f}%)")
    print(f"  htree (sum):     {correct_counts['sum']}/{n} ({correct_counts['sum']/n*100:.1f}%)")
    print(f"  htree_v0:        {correct_counts['v0']}/{n} ({correct_counts['v0']/n*100:.1f}%)")

    return {
        "model": model_alias,
        "n": n,
        "accuracies": {k: v / n for k, v in correct_counts.items()},
        "counts": correct_counts,
        "details": results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="base,v2", help="Comma-separated list of model aliases to test")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    all_summaries = {}
    for m in args.models.split(","):
        res = run_diagnosis(m.strip(), k=args.k)
        all_summaries[m.strip()] = res

    # Write summary report
    out_file = ROOT / "benchmark" / "htree_dev_diagnosis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nFull diagnosis results saved to {out_file}")


if __name__ == "__main__":
    main()
