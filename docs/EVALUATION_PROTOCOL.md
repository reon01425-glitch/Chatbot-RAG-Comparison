# Unified Evaluation Protocol

This protocol defines the experimental controls, metric definitions, and reproducibility requirements for evaluating RAG architectures on the SOP Layanan Akademik FSM Universitas Diponegoro benchmark.

## 1. What is Compared

| Key | System | Notes |
|---|---|---|
| `naive` … `multimodal` | The 6 baselines in `src/engine.py` | via `RAGEngine.query_architecture` |
| `htree` | Hierarchical Tree RAG (proposed) | leaf chunk retrieval (`htree_leaves`) + parent tree expansion |
| `htree_v0` | Hierarchical Tree RAG v0 (ablation baseline) | document-level chunk retrieval + parent tree expansion |
| `workflow` | Workflow GraphRAG / Process DAG (proposed) | same engine |
| `llm_only` | Generator with no retrieval | lower bound; shows what pure-LLM answers look like |
| `full_context` | All 7 SOPs pasted into the prompt | **the cost comparison**: the whole corpus is ~1,100 words, so this is cheap here. Must be in the paper if the paper argues RAG is cheaper than "pure LLM". |

### Experimental Controls
- Generator: Ollama `gemma4:e2b`, temperature 0, seed 42.
- Retrieval: top-$k = 3$, similarity threshold 0.30 across all retrieval-based systems.
- CRAG thresholds: $\ge 0.55$ CORRECT, $0.35 - 0.55$ AMBIGUOUS, $< 0.35$ INCORRECT.
- LLM Call Tracking: All generation calls across all systems (final synthesis, ReAct multi-step agent actions, CRAG query rewrites) route through a single choke point: `RAGCore.call_llm(prompt) -> str`.
- Strict Error Handling: `RAGCore.disable_extractive_fallback = True` is enforced during evaluation. If an LLM call fails, the query is recorded as an error and excluded (never replaced silently by extractive context).

---

## 2. Mandatory Evaluation Flags & Configuration

`evaluate_benchmark.py generate` enforces strict reproducibility guards:

1. **`--embedding-model {base, v1, v2, v3, <path>}` (MANDATORY)**:
   - Evaluator will **refuse to run** if this flag (or `EMBEDDING_MODEL_PATH` env) is omitted.
   - Built-in aliases:
     - `base`: `LazarusNLP/all-indo-e5-small-v4`
     - `v1`: `./indo_finetuned_embedding`
     - `v2`: `./indo_finetuned_embedding_v2`
     - `v3`: `./indo_finetuned_embedding_v3`
2. **Index Manifest Validation**:
   - Every Chroma directory must contain a valid `index_manifest.json` matching the active embedding model's weight SHA256.
   - In evaluation mode (`EVALUATION_MODE=1`), mismatched or missing manifests cause an immediate fatal abort.
3. **`--htree-variant {top1, vote, sum}` (Optional, default: `top1`)**:
   - Controls how Hierarchical Tree RAG selects the parent SOP document from top-$k$ retrieved leaf chunks.
   - Selection is empirically guided by diagnosis on the development set (`benchmark/dev_htree_v1.json`).
4. **`--require-verified` (MANDATORY for Final Published Runs)**:
   - Rejects execution if any item in the benchmark has `"verified": false`.
5. **Git Cleanliness**:
   - `run_config.json` records `git_dirty`. Official benchmark runs must have `git_dirty: false`.

---

## 3. Building Indexes with Manifests

To build or rebuild Chroma vector stores with cryptographic manifests:

```bash
# Build index for base model (chroma_base)
python build_indexes.py --embedding-model base

# Build index for fine-tuned v2 model (chroma_v2)
python build_indexes.py --embedding-model v2

# Build index for audited v3 model (chroma_v3)
python build_indexes.py --embedding-model v3
```

This builds both the `langchain` document collection and the fine-grained `htree_leaves` collection (51 leaves) and generates `index_manifest.json` with model weights SHA256, PDF corpus SHA256, and chunk metadata.

---

## 4. Development Set & Strict Prohibition of Benchmark Tuning

To preserve experimental integrity and prevent test-set contamination:
- **`benchmark/sop_benchmark_v1.json` is STRICTLY a test set.** It must NEVER be used for prompt tuning, threshold adjustment, embedding tuning, or architectural design decisions.
- **Development Set (`benchmark/dev_htree_v1.json`)**: Contains 20 synthetic queries (2–3 per SOP) designed to diagnose edge cases (e.g. Cuti vs Aktif, UKT vs IRS).
  - Verified semantically against benchmark (maximum cosine similarity 0.735, all $< 0.75$).
  - Used for H-Tree diagnosis and parent SOP selection variant comparison:
    ```bash
    python scripts/diagnose_htree_dev.py --embedding-model v3 --k 3
    ```

---

## 5. Data Contamination & Leakage Checks

Training datasets (`datasets/train_v3/*.json`) must pass both lexical and semantic leakage checks against the 40 benchmark questions before fine-tuning:

```bash
# 1. Lexical leakage check (ROUGE-L F1, alert threshold >= 0.50)
python evaluate_benchmark.py check-leakage --train-dir datasets/train_v3

# 2. Semantic leakage check (Dense cosine similarity via base model, alert threshold >= 0.85)
python evaluate_benchmark.py check-leakage --train-dir datasets/train_v3 --semantic
```

---

## 6. Test Set — `benchmark/sop_benchmark_v1.json`

40 questions written from the SOP PDFs (not generated by any evaluated system, not reused from training datasets):

| Category | n | What it tests |
|---|---|---|
| procedural | 11 | full step order and actors (7 formal + 4 informal student phrasing) |
| actor | 7 | who signs / checks / receives |
| documents | 7 | required documents and signatures |
| time_location | 7 | durations, where to pick up, which system |
| cross_sop | 3 | confusing two SOPs (cross-contamination) |
| unanswerable | 5 | information that is NOT in any SOP — correct behaviour is to say so |

**Verification Requirement**: Before reporting final figures in publications, verify every item against the source PDF, set `"verified": true`, and execute with `--require-verified`.

---

## 7. Metrics

### Deterministic (No LLM Judge, Fully Reproducible):
* **Key-Fact Recall (KFR)** – share of required facts (`must_include`) present in the answer.
* **ROUGE-1/2/L** – F1 against the reference (Indonesian whitespace tokenization, no English stemmer).
* **Step Coverage / TIR / AMR** – for procedural items, gold steps are aligned to answer clauses:
  - **TIR (Transition Inversion Rate)**: Share of step pairs presented in inverted order.
  - **AMR (Actor Mismatch Rate)**: Share of steps attributed to the wrong official/role.
  - Unattributed steps are reported separately.
* **Refusal Accuracy** (unanswerable items) and **False-Refusal Rate** (answerable items).
* **Efficiency Metrics (RQ3)**:
  - Latency (median & mean wall clock).
  - Prompt tokens, Completion tokens, and total LLM calls (tracked via `RAGCore.call_llm`).

### LLM-Judged (Optional, `--ragas`):
* Ragas Faithfulness, Answer Relevancy, Context Recall.
* Judge model must be **distinct from the generator** (e.g. generator `gemma4:e2b`, judge `llama3.1:8b`).

### Statistical Significance:
* 95% bootstrap confidence intervals (10,000 resamples) per system.
* Paired Wilcoxon signed-rank tests against Naive RAG with Holm correction (`significance.csv`).

---

## 8. Execution Workflow

```bash
# 1. Run unit tests
python -m pytest -q

# 2. Run data leakage audit
python evaluate_benchmark.py check-leakage --train-dir datasets/train_v3
python evaluate_benchmark.py check-leakage --train-dir datasets/train_v3 --semantic

# 3. Pull required Ollama models
ollama pull gemma4:e2b
ollama pull llama3.1:8b  # if running Ragas scoring

# 4. Smoke check (3 questions, clean working tree)
python evaluate_benchmark.py generate --run-name smoke_check --embedding-model v3 --systems naive,htree --limit 3
python evaluate_benchmark.py score --run-name smoke_check

# 5. Full evaluation run (40 questions, all 10 systems + htree_v0)
python evaluate_benchmark.py generate --run-name gemma4_v3 --embedding-model v3 --systems naive,agentic,crag,graph_sop,workflow,self_rag,htree,htree_v0,llm_only,full_context
python evaluate_benchmark.py score --run-name gemma4_v3 --ragas --judge-model llama3.1:8b --bertscore
```

Outputs are persisted in `results/<run-name>/` with full runtime configuration (`run_config.json`), model weights SHA256, index manifest, and generations for complete auditability.
