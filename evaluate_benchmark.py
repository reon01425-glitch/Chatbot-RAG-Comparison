#!/usr/bin/env python3
"""
Unified, reproducible benchmark for the FSM UNDIP SOP chatbot.

Every system is evaluated on the SAME held-out question set, with the SAME
generator settings, and scored with the SAME metrics.

    # 0) sanity check: benchmark questions must not overlap the fine-tuning data
    python evaluate_benchmark.py check-leakage

    # 1) generate answers (needs Ollama running; resumable, re-run to continue)
    python evaluate_benchmark.py generate --run-name gemma4_v1

    # 2) score (deterministic metrics always; Ragas / BERTScore optional)
    python evaluate_benchmark.py score --run-name gemma4_v1 --ragas --judge-model llama3.1:8b --bertscore

    # quick smoke test on 3 questions and 2 systems
    python evaluate_benchmark.py generate --run-name smoke --systems naive,workflow --limit 3
    python evaluate_benchmark.py score --run-name smoke

Design rules (why this script exists):
  * No silent fallback. If the LLM call fails or Ollama is down, the answer is
    recorded as an error, never replaced by an extractive copy of the context.
  * Faithfulness is judged against the context the generator ACTUALLY received
    (parsed from the prompt), so Workflow GraphRAG's DAG context counts.
  * The engine's built-in heuristic "metrics" are ignored; they are not Ragas.
  * Two reference baselines are added: LLM-only (no retrieval) and Full-context
    (all 7 SOPs in the prompt) - the honest comparison for the cost argument.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = ROOT / "benchmark" / "sop_benchmark_v1.json"
RESULTS_ROOT = ROOT / "results"

# short key -> display / engine name (engine names must match RAGEngine.query_architecture)
SYSTEMS: Dict[str, str] = {
    "naive": "Naive RAG (Baseline)",
    "hybrid": "Hybrid RAG (Dense + BM25)",
    "graph": "GraphRAG (Entity Expansion)",
    "agentic": "Agentic RAG (Tools Agent)",
    "crag": "Corrective RAG (CRAG)",
    "multimodal": "Multimodal RAG (Layout RAG)",
    "htree": "Hierarchical Tree RAG",
    "htree_v0": "Hierarchical Tree RAG v0 (Doc Chunk Baseline)",
    "workflow": "Workflow GraphRAG (Process DAG)",
    "llm_only": "LLM-only (no retrieval)",
    "full_context": "Full-context LLM (all 7 SOPs in prompt)",
}
ENGINE_SYSTEMS = ["naive", "hybrid", "graph", "agentic", "crag", "multimodal", "htree", "htree_v0", "workflow"]
DIRECT_SYSTEMS = ["llm_only", "full_context"]

NO_CONTEXT_TEMPLATE = """
Anda adalah asisten layanan mahasiswa Fakultas Sains dan Matematika Universitas Diponegoro.
Jawablah pertanyaan berikut dengan bahasa Indonesia yang jelas, runut, akurat, dan sopan.
Jika Anda tidak mengetahui jawabannya dengan pasti, katakan secara jujur bahwa informasi tersebut tidak tersedia.

Pertanyaan: {question}

Jawaban sebagai asisten layanan mahasiswa:
"""

PROMPT_CONTEXT_RE = re.compile(r"Konteks:\s*\n(.*?)\n\s*---\s*\n\s*Pertanyaan:", re.S)


# --------------------------------------------------------------------------------------
# Pattern matching (benchmark syntax: whole word/phrase; trailing '*' = prefix)
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _compile(alt: str) -> "re.Pattern[str]":
    alt = alt.strip().lower()
    prefix = alt.endswith("*")
    if prefix:
        alt = alt[:-1]
    body = r"\s+".join(re.escape(p) for p in alt.split())
    return re.compile(r"(?<!\w)" + body + ("" if prefix else r"(?!\w)"))


def match_any(text_lower: str, alts: Sequence[str]) -> bool:
    return any(_compile(a).search(text_lower) for a in alts)


def _clean(text: str) -> str:
    text = re.sub(r"[*_`#>]+", " ", text or "")
    return re.sub(r"[ \t]+", " ", text)


def segment_answer(answer: str) -> List[str]:
    """Split an answer into units: lines, then sentences within long lines."""
    units: List[str] = []
    for line in (answer or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\-•])", line)
        for part in parts:
            # clause boundaries that usually separate procedural steps in Indonesian prose
            sub = re.split(r";\s*|,\s*(?=(?:lalu|kemudian|setelah itu|selanjutnya)\b)", part, flags=re.I)
            units.extend(u.strip(" ,") for u in sub if u and u.strip(" ,"))
    return units


def actors_in(text_lower: str, actor_lex: Dict[str, Dict[str, Any]]) -> Tuple[set, set]:
    """Return (named actors, actors mentioned only via generic aliases like 'anda')."""
    named, generic = set(), set()
    for name, spec in actor_lex.items():
        blocked = set(spec.get("not_preceded_by", []))
        for alt in spec.get("alts", []):
            hit = False
            for m in _compile(alt).finditer(text_lower):
                prev = text_lower[: m.start()].split()
                if blocked and prev and prev[-1].strip(".,;:()/\"'") in blocked:
                    continue
                hit = True
                break
            if hit:
                named.add(name)
                break
        if name not in named and match_any(text_lower, spec.get("generic", [])):
            generic.add(name)
    return named, generic


def is_refusal(answer: str, patterns: Sequence[str]) -> bool:
    return match_any(_clean(answer).lower(), patterns)


def key_fact_recall(answer: str, must_include: List[List[str]]) -> float:
    if not must_include:
        return float("nan")
    low = _clean(answer).lower()
    return sum(1 for grp in must_include if match_any(low, grp)) / len(must_include)


def align_steps(answer: str, steps: List[Dict[str, Any]], actor_lex: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Align gold steps to answer units and judge the actor of each aligned step."""
    units = segment_answer(_clean(answer))
    low = [u.lower() for u in units]
    rows = []
    for st in steps:
        groups = st["kw"]
        need = st.get("min_hits", min(2, len(groups)))
        scores = [sum(1 for g in groups if match_any(u, g)) for u in low]
        best = max(scores) if scores else 0
        row = {"n": st["n"], "detected": False, "pos": None, "hits": best, "actor_status": "not_detected",
               "actors_found": "", "unit": ""}
        if best >= need:
            pos = scores.index(best)
            window = low[pos]
            if pos > 0 and units[pos - 1].rstrip().endswith(":"):
                window = low[pos - 1] + " " + window
            named, generic = actors_in(window, actor_lex)
            gold = set(st["actor"])
            if gold & (named | generic):
                status = "correct"
            elif named - gold - set(st.get("also_ok", [])):
                status = "mismatch"
            else:
                status = "unattributed"
            row.update(detected=True, pos=pos, actor_status=status,
                       actors_found=";".join(sorted(named | {g + "(generic)" for g in generic})),
                       unit=units[pos][:200])
        rows.append(row)
    return rows


def procedural_scores(alignment: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(alignment)
    det = [r for r in alignment if r["detected"]]
    coverage = len(det) / total if total else float("nan")
    disc = comp = 0
    for i in range(len(det)):
        for j in range(i + 1, len(det)):
            a, b = det[i], det[j]          # a precedes b in the gold order
            if a["pos"] == b["pos"]:
                continue
            comp += 1
            if a["pos"] > b["pos"]:
                disc += 1
    tir = disc / comp if comp else float("nan")
    correct = sum(r["actor_status"] == "correct" for r in det)
    mismatch = sum(r["actor_status"] == "mismatch" for r in det)
    unattr = sum(r["actor_status"] == "unattributed" for r in det)
    return {
        "step_coverage": coverage,
        "tir": tir,
        "amr": mismatch / (correct + mismatch) if (correct + mismatch) else float("nan"),
        "actor_accuracy": correct / len(det) if det else float("nan"),
        "actor_unattributed_rate": unattr / len(det) if det else float("nan"),
        "steps_detected": len(det),
        "steps_total": total,
    }


# --------------------------------------------------------------------------------------
# Benchmark I/O
# --------------------------------------------------------------------------------------
def load_benchmark(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        bench = json.load(f)
    ids = [it["id"] for it in bench["items"]]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f"Duplicate item ids in benchmark: {sorted(dup)}")
    for it in bench["items"]:
        if it.get("procedure") and it["procedure"] not in bench["procedures"]:
            raise SystemExit(f"Item {it['id']} references unknown procedure {it['procedure']}")
    return bench


def select_items(bench: Dict[str, Any], args) -> List[Dict[str, Any]]:
    items = bench["items"]
    if getattr(args, "require_verified", False):
        items = [it for it in items if it.get("verified")]
    if getattr(args, "items", None):
        wanted = {s.strip() for s in args.items.split(",")}
        items = [it for it in items if it["id"] in wanted]
    if getattr(args, "limit", None):
        items = items[: args.limit]
    return items


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_systems(spec: str) -> List[str]:
    if spec in ("all", ""):
        return list(SYSTEMS)
    keys = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [k for k in keys if k not in SYSTEMS]
    if bad:
        raise SystemExit(f"Unknown system keys {bad}. Valid: {', '.join(SYSTEMS)}")
    return keys


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "not installed"


# --------------------------------------------------------------------------------------
# Leakage check
# --------------------------------------------------------------------------------------
def cmd_check_leakage(args) -> int:
    from rouge_score import rouge_scorer
    bench = load_benchmark(Path(args.benchmark))
    train_q = []
    for p in sorted((ROOT / "datasets").glob("train_*.json")):
        for it in json.load(open(p, encoding="utf-8")):
            qa = it.get("qa", "")
            if "Q:" in qa and "A:" in qa:
                train_q.append((p.name, qa.split("Q:")[1].split("A:")[0].strip()))
    if not train_q:
        print("No fine-tuning questions found in datasets/train_*.json")
        return 0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    flagged = 0
    print(f"Comparing {len(bench['items'])} benchmark questions with {len(train_q)} fine-tuning questions (ROUGE-L F1)")
    for it in bench["items"]:
        best = max(((scorer.score(tq, it["question"])["rougeL"].fmeasure, src, tq) for src, tq in train_q))
        mark = "  <-- REVIEW" if best[0] >= args.threshold else ""
        flagged += bool(mark)
        print(f"{it['id']:8s} max={best[0]:.2f}  ({best[1]}){mark}")
    print(f"\n{flagged} item(s) at or above threshold {args.threshold}.")
    return 0


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
class _Recorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.errors: List[str] = []
        self.prompts: List[str] = []


def _ollama_up(host: str = "127.0.0.1", port: int = 11434) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _ollama_version() -> str:
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2) as r:
            return json.loads(r.read().decode()).get("version", "unknown")
    except Exception:
        return "unknown"


def _make_llm(model: str, args):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=0.0, seed=args.seed, num_predict=args.max_tokens,
                      num_ctx=args.num_ctx, client_kwargs={"timeout": args.timeout})


def _invoke(llm, prompt: str, rec: _Recorder) -> str:
    rec.calls += 1
    rec.prompts.append(prompt)
    try:
        resp = llm.invoke(prompt)
        md = getattr(resp, "response_metadata", None) or {}
        rec.prompt_tokens += int(md.get("prompt_eval_count") or 0)
        rec.completion_tokens += int(md.get("eval_count") or 0)
        text = (resp.content or "").strip()
        if not text:
            rec.errors.append("empty_response")
        return text
    except Exception as e:  # recorded, never replaced by an extractive fallback
        rec.errors.append(f"{type(e).__name__}: {e}")
        return ""


def _prompt_context(prompts: List[str]) -> Optional[str]:
    for p in reversed(prompts):
        m = PROMPT_CONTEXT_RE.search(p)
        if m:
            return m.group(1).strip()
    return None


def _engine_prompt_template() -> str:
    """Read PROMPT_TEMPLATE from src/engine.py without importing the heavy engine stack."""
    import ast
    tree = ast.parse((ROOT / "src" / "engine.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PROMPT_TEMPLATE" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("PROMPT_TEMPLATE not found in src/engine.py")


def _load_corpus_text() -> str:
    import pypdf
    parts = []
    for pdf in sorted((ROOT / "data").glob("*.pdf")):
        reader = pypdf.PdfReader(str(pdf))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        parts.append(f"### {pdf.stem}\n{text.strip()}")
    return "\n\n".join(parts)


def cmd_generate(args) -> int:
    bench_path = Path(args.benchmark)
    bench = load_benchmark(bench_path)
    items = select_items(bench, args)
    systems = parse_systems(args.systems)
    run_dir = RESULTS_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    gen_path = run_dir / "generations.jsonl"

    model = args.model or os.getenv("OLLAMA_MODEL", "gemma4:e2b")
    if not _ollama_up():
        raise SystemExit("Ollama is not reachable at 127.0.0.1:11434. Start it (`ollama serve`) and pull the "
                         f"model (`ollama pull {model}`). Refusing to run: the engine would silently fall back "
                         "to extractive answers, which would invalidate the benchmark.")
    os.environ["OLLAMA_MODEL"] = model

    unverified = sum(1 for it in items if not it.get("verified"))
    if unverified:
        print(f"WARNING: {unverified}/{len(items)} benchmark items are not yet verified by a human.")

    emb_model = getattr(args, "embedding_model", None) or os.getenv("EMBEDDING_MODEL_PATH", "./indo_finetuned_embedding")
    if emb_model.lower() == "base":
        emb_model = "LazarusNLP/all-indo-e5-small-v4"

    chroma_dir = getattr(args, "chroma_path", None) or os.getenv("CHROMA_PATH", None)
    if not chroma_dir:
        if "all-indo-e5-small-v4" in emb_model:
            chroma_dir = "chroma_base" if (ROOT / "chroma_base").exists() else "chroma"
        elif "v2" in emb_model:
            chroma_dir = "chroma_v2" if (ROOT / "chroma_v2").exists() else "chroma"
        else:
            chroma_dir = "chroma"

    os.environ["EMBEDDING_MODEL_PATH"] = emb_model
    os.environ["CHROMA_PATH"] = str(chroma_dir)

    cfg_path = run_dir / "run_config.json"
    cfg = {
        "run_name": args.run_name,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "benchmark_file": str(bench_path.relative_to(ROOT) if bench_path.is_relative_to(ROOT) else bench_path),
        "benchmark_sha256": sha256_file(bench_path),
        "benchmark_version": bench.get("version"),
        "n_items": len(items),
        "n_items_verified": len(items) - unverified,
        "systems": {k: SYSTEMS[k] for k in systems},
        "generator_model": model,
        "embedding_model": emb_model,
        "chroma_path": str(chroma_dir),
        "ollama_version": _ollama_version(),
        "temperature": 0.0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "num_ctx": args.num_ctx,
        "llm_timeout_s": args.timeout,
        "retrieval_k": args.k,
        "retrieval_threshold": args.threshold,
        "agentic_max_steps": getattr(args, "agentic_max_steps", 3),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _pkg_version(p) for p in
                     ["langchain", "langchain-ollama", "langchain-chroma", "sentence-transformers", "ragas",
                      "rouge-score", "bert-score", "pypdf"]},
        "notes": ("Token counts are read from Ollama for the answer-synthesis call(s). Latency is wall-clock per "
                  "query after a warm-up query. The engine's built-in heuristic metrics are not used."),
    }
    if cfg_path.exists():
        old = json.load(open(cfg_path, encoding="utf-8"))
        for key in ("generator_model", "benchmark_sha256", "seed", "retrieval_k", "retrieval_threshold", "num_ctx", "embedding_model"):
            if old.get(key) != cfg.get(key):
                raise SystemExit(f"run_config mismatch on '{key}' ({old.get(key)} vs {cfg.get(key)}). "
                                 "Use a new --run-name instead of mixing settings in one run.")
        cfg["created"] = old.get("created", cfg["created"])
        cfg["resumed"] = _dt.datetime.now().isoformat(timespec="seconds")
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in read_jsonl(gen_path):
        latest[(r["system"], r["qid"])] = r
    done = {k for k, r in latest.items()
            if not (args.retry_errors and r.get("status") in ("llm_error", "exception"))}
    todo = [(s, it) for s in systems for it in items if (s, it["id"]) not in done]
    print(f"Run '{args.run_name}': {len(done)} done, {len(todo)} to generate "
          f"({len(systems)} systems x {len(items)} items, model {model}).")
    if not todo:
        return 0

    rec = _Recorder()
    llm = _make_llm(model, args)

    sys.path.insert(0, str(ROOT))
    engine = None
    if any(s in ENGINE_SYSTEMS for s, _ in todo):
        from src.engine import RAGCore, RAGEngine  # noqa: E402

        def _patched_call_llm(self, prompt: str) -> str:  # intercepts all LLM calls (synthesis, agent, rewrite)
            return _invoke(llm, prompt, rec)

        RAGCore.call_llm = _patched_call_llm
        RAGCore.disable_extractive_fallback = True
        os.environ["RAG_DISABLE_EXTRACTIVE_FALLBACK"] = "1"
        engine = RAGEngine()
        engine.agentic_max_steps = getattr(args, "agentic_max_steps", 3)

    PROMPT_TEMPLATE = _engine_prompt_template()  # same prompt for the direct baselines
    corpus_text = _load_corpus_text() if "full_context" in systems else ""

    def run_one(system: str, question: str) -> Dict[str, Any]:
        rec.reset()
        t0 = time.perf_counter()
        matched_sop = None
        if system in ENGINE_SYSTEMS:
            res = engine.query_architecture(SYSTEMS[system], question, k=args.k, threshold=args.threshold)
            answer, ctx = res.get("answer", ""), list(res.get("contexts") or [])
            matched_sop = res.get("matched_sop")
        elif system == "full_context":
            answer = _invoke(llm, PROMPT_TEMPLATE.format(context=corpus_text, question=question), rec)
            ctx = [corpus_text]
        else:  # llm_only
            answer = _invoke(llm, NO_CONTEXT_TEMPLATE.format(question=question), rec)
            ctx = []
        latency = time.perf_counter() - t0
        pctx = _prompt_context(rec.prompts)
        if rec.calls == 0:
            status = "no_llm_call"          # e.g. retrieval threshold refusal, answered by a template
        elif rec.errors and not answer:
            status = "llm_error"
        else:
            status = "ok"
        return {"answer": answer, "contexts": ctx, "prompt_context": pctx, "latency_s": round(latency, 3),
                "llm_calls": rec.calls, "prompt_tokens": rec.prompt_tokens,
                "completion_tokens": rec.completion_tokens, "errors": rec.errors, "status": status,
                "matched_sop": matched_sop}

    # warm-up (loads models into memory so the first timed query is not penalised)
    print("Warm-up ...")
    _invoke(llm, "Jawab singkat: halo", rec)
    if engine is not None:
        try:
            engine.query_architecture(SYSTEMS["naive"], "Bagaimana cara mengisi IRS?", k=args.k, threshold=args.threshold)
        except Exception as e:
            print(f"Warm-up engine query failed: {e}")

    for i, (system, it) in enumerate(todo, 1):
        try:
            out = run_one(system, it["question"])
        except Exception as e:
            out = {"answer": "", "contexts": [], "prompt_context": None, "latency_s": None, "llm_calls": rec.calls,
                   "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
                   "errors": [f"{type(e).__name__}: {e}"], "status": "exception", "matched_sop": None}
        row = {"run": args.run_name, "system": system, "system_name": SYSTEMS[system], "qid": it["id"],
               "category": it["category"], "question": it["question"], **out,
               "timestamp": _dt.datetime.now().isoformat(timespec="seconds")}
        append_jsonl(gen_path, row)
        flag = "" if out["status"] in ("ok", "no_llm_call") else f"  [{out['status']}] {out['errors'][:1]}"
        print(f"[{i}/{len(todo)}] {system:12s} {it['id']:8s} {out['latency_s']}s "
              f"tok={out['prompt_tokens']}+{out['completion_tokens']}{flag}")
    print(f"\nSaved generations to {gen_path}")
    return 0


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def _nanmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values, n_boot: int = 10000, seed: int = 42, alpha: float = 0.05):
    import numpy as np
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if v.size == 1:
        return float(v[0]), float("nan"), float("nan"), 1
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi), int(v.size)


def holm(pvals: List[float]) -> List[float]:
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvals[i])
    adj = [float("nan")] * len(pvals)
    running = 0.0
    m = len(order)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def _rouge(refs: List[str], cands: List[str]) -> List[Dict[str, float]]:
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    out = []
    for r, c in zip(refs, cands):
        s = sc.score(r, c or "")
        out.append({"rouge1": s["rouge1"].fmeasure, "rouge2": s["rouge2"].fmeasure, "rougeL": s["rougeL"].fmeasure})
    return out


def _bertscore(refs, cands, args) -> List[float]:
    from bert_score import score as bs
    kwargs = {"lang": "id", "verbose": False, "batch_size": 8}
    if args.bertscore_model:
        kwargs = {"model_type": args.bertscore_model, "num_layers": args.bertscore_layers, "verbose": False,
                  "batch_size": 8}
    _, _, f1 = bs([c or " " for c in cands], refs, **kwargs)
    return [float(x) for x in f1]


def _run_ragas(rows: List[Dict[str, Any]], args, cache_path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Ragas faithfulness / answer relevancy / context recall with one judge for all systems. Cached."""
    cache = {(r["system"], r["qid"]): r for r in read_jsonl(cache_path) if r.get("judge") == args.judge_model}
    todo = [r for r in rows if (r["system"], r["qid"]) not in cache]
    if todo:
        if not _ollama_up():
            raise SystemExit("Ollama is not reachable; it is needed for the Ragas judge model.")
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_ollama import ChatOllama
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
        from ragas.run_config import RunConfig

        judge = LangchainLLMWrapper(ChatOllama(model=args.judge_model, temperature=0.0, seed=args.seed,
                                               client_kwargs={"timeout": args.judge_timeout}))
        emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=args.judge_embedding))
        by_system: Dict[str, List[Dict[str, Any]]] = {}
        for r in todo:
            by_system.setdefault(r["system"], []).append(r)
        for system, srows in by_system.items():
            print(f"Ragas judge {args.judge_model}: {system} ({len(srows)} answers) ...")
            for r in srows:
                ctx = [r["prompt_context"]] if r.get("prompt_context") else [c for c in r.get("contexts", []) if c]
                metrics = [ResponseRelevancy()]
                if ctx:
                    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextRecall()]
                sample = SingleTurnSample(user_input=r["question"], response=r["answer"] or " ",
                                          retrieved_contexts=ctx or [" "], reference=r["reference"])
                try:
                    res = evaluate(EvaluationDataset(samples=[sample]), metrics=metrics, llm=judge, embeddings=emb,
                                   run_config=RunConfig(timeout=args.judge_timeout, max_workers=1, max_retries=2),
                                   show_progress=False, raise_exceptions=False)
                    df = res.to_pandas()
                    vals = {k: (float(df[k].iloc[0]) if k in df.columns and df[k].iloc[0] == df[k].iloc[0]
                                else float("nan"))
                            for k in ("faithfulness", "answer_relevancy", "context_recall")}
                except Exception as e:
                    print(f"  Ragas failed on {system}/{r['qid']}: {e}")
                    vals = {"faithfulness": float("nan"), "answer_relevancy": float("nan"),
                            "context_recall": float("nan")}
                entry = {"system": system, "qid": r["qid"], "judge": args.judge_model,
                         "judge_embedding": args.judge_embedding, **vals}
                append_jsonl(cache_path, entry)
                cache[(system, r["qid"])] = entry
    return {k: {m: v.get(m, float("nan")) for m in ("faithfulness", "answer_relevancy", "context_recall")}
            for k, v in cache.items()}


def _fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def cmd_score(args) -> int:
    import pandas as pd

    bench = load_benchmark(Path(args.benchmark))
    item_by_id = {it["id"]: it for it in bench["items"]}
    run_dir = RESULTS_ROOT / args.run_name
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for g in read_jsonl(run_dir / "generations.jsonl"):   # the last row per (system, item) wins
        latest[(g["system"], g["qid"])] = g
    gens = list(latest.values())
    if not gens:
        raise SystemExit(f"No generations found in {run_dir}. Run `generate` first.")
    if args.require_verified:
        gens = [g for g in gens if item_by_id.get(g["qid"], {}).get("verified")]
    gens = [g for g in gens if g["qid"] in item_by_id]
    if args.systems != "all":
        keep = set(parse_systems(args.systems))
        gens = [g for g in gens if g["system"] in keep]

    actor_lex = bench["actors"]
    refusal_pats = bench["refusal_patterns"]
    for g in gens:
        g["reference"] = item_by_id[g["qid"]]["reference"]

    rouge = _rouge([g["reference"] for g in gens], [g["answer"] for g in gens])
    align_rows, rows = [], []
    for g, rg in zip(gens, rouge):
        it = item_by_id[g["qid"]]
        refused = is_refusal(g["answer"], refusal_pats)
        kfr = key_fact_recall(g["answer"], it.get("must_include", []))
        row = {"system": g["system"], "system_name": g["system_name"], "qid": g["qid"], "category": it["category"],
               "answerable": bool(it.get("answerable", True)), "verified": bool(it.get("verified")),
               "status": g.get("status"), "latency_s": g.get("latency_s"), "prompt_tokens": g.get("prompt_tokens"),
               "completion_tokens": g.get("completion_tokens"), "llm_calls": g.get("llm_calls"),
               "answer_chars": len(g.get("answer") or ""), **rg, "key_fact_recall": kfr, "refusal": refused}
        if row["answerable"]:
            row["false_refusal"] = float(refused and (kfr == 0 or math.isnan(kfr)))
            row["refusal_correct"] = float("nan")
        else:
            row["false_refusal"] = float("nan")
            row["refusal_correct"] = float(refused)
        proc = it.get("procedure")
        if proc:
            al = align_steps(g["answer"], bench["procedures"][proc]["steps"], actor_lex)
            row.update(procedural_scores(al))
            for a in al:
                align_rows.append({"system": g["system"], "qid": g["qid"], **a})
        rows.append(row)

    if args.bertscore:
        print("Computing BERTScore ...")
        f1 = _bertscore([g["reference"] for g in gens], [g["answer"] for g in gens], args)
        for row, v in zip(rows, f1):
            row["bertscore_f1"] = v

    if args.ragas:
        rag = _run_ragas(gens, args, run_dir / "ragas_cache.jsonl")
        for row in rows:
            row.update(rag.get((row["system"], row["qid"]), {}))
        for row in rows:  # LLM-only has no context: faithfulness/context recall are undefined
            if row["system"] == "llm_only":
                row["faithfulness"] = float("nan")
                row["context_recall"] = float("nan")

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "scores_per_query.csv", index=False)
    err_mask = df["status"].isin(["llm_error", "exception"])
    if err_mask.any():
        print(f"WARNING: {int(err_mask.sum())} answer(s) ended in an LLM error and are EXCLUDED from quality metrics "
              "(see n_errors). Re-run `generate --retry-errors` before reporting.")
    df_all, df = df, df[~err_mask].copy()
    if align_rows:
        pd.DataFrame(align_rows).to_csv(run_dir / "step_alignment_audit.csv", index=False)

    # ---- summary with bootstrap CIs
    answerable_metrics = ["key_fact_recall", "rouge1", "rouge2", "rougeL", "bertscore_f1", "faithfulness",
                          "answer_relevancy", "context_recall", "false_refusal"]
    procedural_metrics = ["step_coverage", "tir", "amr", "actor_accuracy", "actor_unattributed_rate"]
    summary = []
    order = [s for s in SYSTEMS if s in set(df_all["system"])]
    for s in order:
        d = df[df["system"] == s]
        rec = {"system": s, "system_name": SYSTEMS[s], "n_items": int((df_all["system"] == s).sum()),
               "n_errors": int(((df_all["system"] == s) & err_mask).sum())}
        da = d[d["answerable"]]
        for m in answerable_metrics:
            if m in d.columns:
                mean, lo, hi, n = bootstrap_ci(da[m].tolist(), seed=args.seed)
                rec.update({m: mean, f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_n": n})
        dp = d[d["step_coverage"].notna()] if "step_coverage" in d.columns else d.iloc[0:0]
        for m in procedural_metrics:
            if m in d.columns:
                mean, lo, hi, n = bootstrap_ci(dp[m].tolist(), seed=args.seed)
                rec.update({m: mean, f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_n": n})
        du = d[~d["answerable"]]
        mean, lo, hi, n = bootstrap_ci(du["refusal_correct"].tolist(), seed=args.seed)
        rec.update({"refusal_accuracy": mean, "refusal_accuracy_lo": lo, "refusal_accuracy_hi": hi,
                    "refusal_accuracy_n": n})
        lat = [x for x in d["latency_s"].tolist() if x is not None and x == x]
        rec["latency_mean_s"] = _nanmean(lat)
        rec["latency_median_s"] = float(pd.Series(lat).median()) if lat else float("nan")
        rec["prompt_tokens_mean"] = _nanmean(d["prompt_tokens"].tolist())
        rec["completion_tokens_mean"] = _nanmean(d["completion_tokens"].tolist())
        summary.append(rec)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(run_dir / "summary.csv", index=False)

    # ---- per-category breakdown
    cat_metrics = [m for m in ["key_fact_recall", "rougeL", "faithfulness", "answer_relevancy", "step_coverage",
                               "tir", "amr", "refusal_correct"] if m in df.columns]
    df.groupby(["system", "category"])[cat_metrics].mean(numeric_only=True).reset_index() \
        .to_csv(run_dir / "summary_by_category.csv", index=False)

    # ---- paired significance vs the reference system (Wilcoxon signed-rank + Holm)
    sig_rows = []
    ref = args.reference
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
            adj = holm([b["p_wilcoxon"] for b in block])
            for b, a in zip(block, adj):
                b["p_holm"] = a
            sig_rows.extend(block)
        pd.DataFrame(sig_rows).to_csv(run_dir / "significance.csv", index=False)

    # ---- LaTeX table (mean with 95% bootstrap CI)
    cols = [("key_fact_recall", "KFR"), ("rougeL", "R-L"), ("faithfulness", "Faith."),
            ("answer_relevancy", "Relev."), ("step_coverage", "StepCov"), ("tir", "TIR$\\downarrow$"),
            ("amr", "AMR$\\downarrow$"), ("refusal_accuracy", "Refusal"), ("latency_median_s", "Lat.(s)"),
            ("prompt_tokens_mean", "Tok$_{in}$")]
    cols = [(k, h) for k, h in cols if k in sdf.columns and sdf[k].notna().any()]
    n_items = df["qid"].nunique()
    n_ver = df[df["verified"]]["qid"].nunique()
    lines = [
        "% Auto-generated by evaluate_benchmark.py -- do not edit numbers by hand",
        "\\begin{table*}[!t]", "\\centering",
        f"\\caption{{Unified benchmark on {n_items} held-out questions ({n_ver} human-verified), generator "
        f"\\texttt{{{json.load(open(run_dir / 'run_config.json'))['generator_model'] if (run_dir / 'run_config.json').exists() else '?'}}}"
        + (f", judge \\texttt{{{args.judge_model}}}" if args.ragas else "")
        + ". Mean with 95\\% bootstrap CI.}",
        "\\label{tab:unified_benchmark}", "\\scriptsize",
        "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
        "\\textbf{System} & " + " & ".join(f"\\textbf{{{h}}}" for _, h in cols) + " \\\\", "\\midrule"]
    for _, r in sdf.iterrows():
        cells = []
        for k, _ in cols:
            if k in ("latency_median_s", "prompt_tokens_mean"):
                cells.append(_fmt(r[k], 1 if k == "latency_median_s" else 0))
            elif f"{k}_lo" in r and r[f"{k}_lo"] == r[f"{k}_lo"]:
                cells.append(f"{_fmt(r[k])} [{_fmt(r[k + '_lo'], 2)}, {_fmt(r[k + '_hi'], 2)}]")
            else:
                cells.append(_fmt(r[k]))
        lines.append(r["system_name"].replace("&", "\\&") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (run_dir / "summary_table.tex").write_text("\n".join(lines), encoding="utf-8")

    score_cfg = {"scored": _dt.datetime.now().isoformat(timespec="seconds"), "ragas": args.ragas,
                 "judge_model": args.judge_model if args.ragas else None,
                 "judge_embedding": args.judge_embedding if args.ragas else None,
                 "bertscore": args.bertscore, "bertscore_model": args.bertscore_model or "lang=id default",
                 "reference_system": ref, "bootstrap_resamples": 10000, "seed": args.seed,
                 "require_verified": args.require_verified, "n_items": n_items, "n_items_verified": n_ver}
    json.dump(score_cfg, open(run_dir / "score_config.json", "w", encoding="utf-8"), indent=2)

    # ---- console summary
    show = ["key_fact_recall", "rougeL", "faithfulness", "answer_relevancy", "step_coverage", "tir", "amr",
            "refusal_accuracy", "latency_median_s", "prompt_tokens_mean"]
    show = [c for c in show if c in sdf.columns and sdf[c].notna().any()]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(sdf[["system"] + show].round(3).to_string(index=False))
    if n_ver < n_items:
        print(f"\nNOTE: only {n_ver}/{n_items} items are human-verified. Verify the benchmark before reporting.")
    print(f"\nWrote: {run_dir}/summary.csv, summary_by_category.csv, significance.csv, scores_per_query.csv, "
          f"step_alignment_audit.csv, summary_table.tex")
    return 0


# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    sub = p.add_subparsers(dest="cmd", required=True)

    lk = sub.add_parser("check-leakage", help="compare benchmark questions with fine-tuning questions")
    lk.add_argument("--threshold", type=float, default=0.5)

    g = sub.add_parser("generate", help="generate answers for all systems (resumable)")
    g.add_argument("--run-name", required=True)
    g.add_argument("--systems", default="all", help=f"comma list of: {', '.join(SYSTEMS)}")
    g.add_argument("--model", default=None, help="generator model (default: $OLLAMA_MODEL or gemma4:e2b)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--max-tokens", type=int, default=1024)
    g.add_argument("--num-ctx", type=int, default=8192)
    g.add_argument("--timeout", type=float, default=300.0, help="LLM call timeout in seconds")
    g.add_argument("--k", type=int, default=3)
    g.add_argument("--threshold", type=float, default=0.3)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--items", default=None, help="comma list of item ids")
    g.add_argument("--require-verified", action="store_true")
    g.add_argument("--retry-errors", action="store_true", help="re-generate rows that ended in an LLM error")
    g.add_argument("--agentic-max-steps", type=int, default=3, help="Maximum ReAct steps for Agentic RAG (default: 3)")
    g.add_argument("--embedding-model", default=None,
                   help="Embedding model path/ID (e.g. 'base', 'LazarusNLP/all-indo-e5-small-v4', './indo_finetuned_embedding_v2')")
    g.add_argument("--chroma-path", default=None,
                   help="Chroma persist directory (e.g. 'chroma', 'chroma_base', 'chroma_v2')")

    s = sub.add_parser("score", help="score generations")
    s.add_argument("--run-name", required=True)
    s.add_argument("--systems", default="all")
    s.add_argument("--reference", default="naive", help="system used as the baseline in significance tests")
    s.add_argument("--ragas", action="store_true", help="run Ragas faithfulness/relevancy/context recall")
    s.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "llama3.1:8b"),
                   help="Ragas judge (use a DIFFERENT model from the generator)")
    s.add_argument("--judge-embedding", default="LazarusNLP/all-indo-e5-small-v4",
                   help="embedding model for answer relevancy (base model, not the fine-tuned one)")
    s.add_argument("--judge-timeout", type=float, default=600.0)
    s.add_argument("--bertscore", action="store_true")
    s.add_argument("--bertscore-model", default=None, help="e.g. cahya/bert-base-indonesian-522M")
    s.add_argument("--bertscore-layers", type=int, default=9)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--require-verified", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "check-leakage":
        return cmd_check_leakage(args)
    if args.cmd == "generate":
        return cmd_generate(args)
    if args.cmd == "score":
        return cmd_score(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
