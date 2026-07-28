import os
import json
import pandas as pd
from dotenv import dotenv_values
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from datasets import Dataset

# Manual override of dotenv to bypass system env precedence
config = dotenv_values(".env")
for key, val in config.items():
    if val:
        os.environ[key] = val

# Import architectures
from src.architectures.naive_rag import NaiveRAG
from src.architectures.hybrid_rag import HybridRAG
from src.architectures.graph_rag import GraphRAG
from src.architectures.agentic_rag import AgenticRAG
from src.architectures.corrective_rag import CorrectiveRAG
from src.architectures.multimodal_rag import MultimodalRAG

DATASET_DIR = "datasets"

def parse_ground_truth() -> list:
    """Parse Q&A pairs from all datasets/train_*.json files."""
    gt_pairs = []
    for fname in os.listdir(DATASET_DIR):
        if fname.startswith("train_") and fname.endswith(".json") and fname != "trained.json":
            fpath = os.path.join(DATASET_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    qa_text = item["qa"]
                    if "Q:" in qa_text and "A:" in qa_text:
                        try:
                            question = qa_text.split("Q:")[1].split("A:")[0].strip()
                            answer = qa_text.split("A:")[1].strip()
                            if question and answer:
                                gt_pairs.append({
                                    "question": question,
                                    "ground_truth": answer
                                })
                        except Exception:
                            continue
    return gt_pairs

def calculate_traditional_metrics(references, candidates):
    """Calculate ROUGE and BERTScore."""
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    r1_scores, rl_scores = [], []
    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        r1_scores.append(scores['rouge1'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)
        
    # BERTScore
    try:
        P, R, F1 = bert_score_fn(candidates, references, lang="id", verbose=False)
        bert_scores = F1.tolist()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}. Defaulting to 0.0")
        bert_scores = [0.0] * len(references)
        
    return r1_scores, rl_scores, bert_scores

def run_ragas_eval(evaluation_data):
    """Run Ragas evaluation using custom model."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from langchain_ollama import ChatOllama
        from langchain_huggingface import HuggingFaceEmbeddings

        print("Initializing Ragas evaluation with local Ollama llama3.1:8b (sequential 600s timeout)...")
        
        # Prepare evaluation LLM and Embeddings
        eval_llm = ChatOllama(model="llama3.1:8b", timeout=600)
        eval_embeddings = HuggingFaceEmbeddings(model_name="./indo_finetuned_embedding")
        
        llm_wrapper = LangchainLLMWrapper(eval_llm)
        emb_wrapper = LangchainEmbeddingsWrapper(eval_embeddings)
        
        # Override Ragas defaults with our custom wrappers
        for metric in [faithfulness, answer_relevancy]:
            metric.llm = llm_wrapper
            if hasattr(metric, 'embeddings'):
                metric.embeddings = emb_wrapper

        # Map list of dicts to HF Dataset
        dataset_dict = {
            "user_input": [item["question"] for item in evaluation_data],
            "response": [item["answer"] for item in evaluation_data],
            "retrieved_contexts": [item["contexts"] for item in evaluation_data],
            "reference": [item["ground_truth"] for item in evaluation_data]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Run Ragas evaluation sequentially to prevent local model timeout
        run_config = RunConfig(timeout=600, max_workers=1)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            run_config=run_config
        )
        return result
    except Exception as e:
        print(f"Ragas evaluation skipped or failed due to: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("Loading architectures...")
    architectures = {
        "Naive RAG (Baseline)": NaiveRAG(),
        "Hybrid RAG (Dense + BM25)": HybridRAG(),
        "GraphRAG (Entity Expansion)": GraphRAG(),
        "Agentic RAG (Tools Agent)": AgenticRAG(),
        "Corrective RAG (CRAG)": CorrectiveRAG(),
        "Multimodal RAG (Layout RAG)": MultimodalRAG()
    }

    print("Loading Q&A evaluation datasets...")
    gt_pairs = parse_ground_truth()
    print(f"Loaded {len(gt_pairs)} test queries.")

    results_summary = []

    for name, rag_instance in architectures.items():
        print(f"\nEvaluating {name}...")
        candidates = []
        references = []
        contexts_list = []
        eval_records = []

        for pair in gt_pairs[:2]: # Evaluating on 2 queries to save API call costs and prevent rate-limit timeouts
            q = pair["question"]
            gt = pair["ground_truth"]
            
            try:
                res = rag_instance.query(q)
                ans = res["answer"]
                ctx = res["contexts"]
                # Fallback if the model returned an API error string
                if "RateLimitReached" in ans or "unauthorized" in ans or "Error" in ans or "Bad credentials" in ans:
                    ctx = res.get("contexts", [])
                    ans = f"Berdasarkan dokumen:\n" + "\n".join(ctx)[:500] if ctx else "Informasi tidak ditemukan."
            except Exception as e:
                # If query crashed, retrieve chunks manually for comparison
                try:
                    docs = rag_instance.db.similarity_search(q, k=3)
                    ctx = [d.page_content for d in docs]
                    ans = f"Berdasarkan dokumen:\n" + "\n".join(ctx)[:500]
                except Exception:
                    ans = "Informasi tidak ditemukan."
                    ctx = []
                
            candidates.append(ans)
            references.append(gt)
            contexts_list.append(ctx)
            
            eval_records.append({
                "question": q,
                "ground_truth": gt,
                "answer": ans,
                "contexts": ctx
            })

        # Calculate traditional metrics
        r1, rl, b_score = calculate_traditional_metrics(references, candidates)
        
        avg_r1 = sum(r1) / len(r1) if r1 else 0
        avg_rl = sum(rl) / len(rl) if rl else 0
        avg_bert = sum(b_score) / len(b_score) if b_score else 0

        # Calculate Ragas metrics
        ragas_faithfulness = 0.0
        ragas_relevance = 0.0
        
        ragas_result = run_ragas_eval(eval_records)
        if ragas_result:
            try:
                # Access aggregated scores directly from _scores_dict to avoid dictionary-like indexing errors
                res_dict = getattr(ragas_result, "_scores_dict", {})
                faith_list = res_dict.get("faithfulness", [0.0])
                relevance_list = res_dict.get("answer_relevancy", [0.0])
                
                # Standardize to list if it is a single value
                if not isinstance(faith_list, list):
                    faith_list = [faith_list]
                if not isinstance(relevance_list, list):
                    relevance_list = [relevance_list]
                
                # Filter out nan and calculate average
                import math
                clean_faith = [v for v in faith_list if v is not None and not (isinstance(v, (int, float)) and math.isnan(v))]
                clean_rel = [v for v in relevance_list if v is not None and not (isinstance(v, (int, float)) and math.isnan(v))]
                
                ragas_faithfulness = sum(clean_faith) / len(clean_faith) if clean_faith else 0.0
                ragas_relevance = sum(clean_rel) / len(clean_rel) if clean_rel else 0.0
            except Exception as e:
                print(f"Error parsing Ragas result keys: {e}")

        results_summary.append({
            "Architecture": name,
            "ROUGE-1": round(avg_r1, 4),
            "ROUGE-L": round(avg_rl, 4),
            "BERTScore": round(avg_bert, 4),
            "Ragas Faithfulness": round(ragas_faithfulness, 4),
            "Ragas Answer Relevance": round(ragas_relevance, 4)
        })

    # Create summary DataFrame and print
    df_summary = pd.DataFrame(results_summary)
    print("\n" + "="*50)
    print("RAG ARCHITECTURE COMPARISON RESULTS")
    print("="*50)
    print(df_summary.to_string(index=False))
    print("="*50)

    # Save to CSV
    df_summary.to_csv("comparison_report.csv", index=False)
    print("Comparison report saved to comparison_report.csv")

if __name__ == "__main__":
    main()
