import os
import time
from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from src.graph_sop.workflow_graph import SOPWorkflowGraph

CHROMA_PATH = "chroma"
EMBEDDING_MODEL_PATH = "./indo_finetuned_embedding"

PROMPT_TEMPLATE = """
Anda adalah asisten layanan akademik Fakultas Sains dan Matematika Universitas Diponegoro (FSM UNDIP)
yang bertugas memberikan panduan operasional SOP secara terstruktur, presisi, dan runut.

Gunakan informasi resmi alur proses (Workflow / State Machine Graph) dan dokumen SOP berikut untuk menjawab pertanyaan pengguna.
Jika terdapat urutan langkah (sekuensial), sebutkan urutan langkah, aktor yang bertanggung jawab, serta dokumen luaran/prasyaratnya dengan rapi.
JANGAN gunakan bahasa Inggris dalam memberikan jawaban.

{workflow_context}

Konteks Dokumen Tambahan:
{document_context}

---

Pertanyaan: {question}

Jawaban sebagai Asisten Akademik FSM UNDIP:
"""

class WorkflowGraphRAG:
    """
    Process / Workflow Graph RAG Architecture.
    Leverages directed state-machine graph of SOP steps, swimlane actors,
    prerequisites, and outputs to provide chronologically grounded procedural answers.
    """
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)
        model_name = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
        self.model = ChatOllama(model=model_name)
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.workflow_graph = SOPWorkflowGraph()

    def query(self, query_text: str, k: int = 3, threshold: float = 0.3) -> Dict[str, Any]:
        start_time = time.time()
        trace = []
        
        # 1. Match SOP from Workflow Graph
        matched_sop = self.workflow_graph.match_sop(query_text)
        trace.append({
            "step": "1. Workflow State-Machine Identification",
            "type": "Graph Process Matching",
            "detail": f"Matched SOP: {matched_sop['title'] if matched_sop else 'None'} (Graph ID: {matched_sop['id'] if matched_sop else 'N/A'})"
        })
        
        workflow_context = ""
        mermaid_code = ""
        if matched_sop:
            workflow_context = self.workflow_graph.get_workflow_context(matched_sop["id"])
            mermaid_code = self.workflow_graph.generate_mermaid_flowchart(matched_sop["id"])
            trace.append({
                "step": "2. Sequential Step Traversal (BPMN DAG)",
                "type": "Topological Graph Path",
                "detail": f"Traversed {len(matched_sop['steps'])} ordered steps with swimlane actors and artifacts."
            })
            
        # 2. Vector search on Chroma for supplementary document details
        expanded_query = query_text
        if matched_sop:
            expanded_query += f" {matched_sop['title']} " + " ".join([s['action'] for s in matched_sop['steps'][:3]])
            
        docs = self.db.similarity_search(expanded_query, k=k)
        query_emb = self.embedding_function.embed_query(query_text)
        
        scored_results = []
        for doc in docs:
            doc_emb = self.embedding_function.embed_query(doc.page_content)
            sim = float(cosine_similarity([query_emb], [doc_emb])[0][0])
            scored_results.append((doc, sim))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_score = scored_results[0][1] if scored_results else 0.0
        
        trace.append({
            "step": "3. Vector Context Alignment",
            "type": "Similarity Retrieval",
            "detail": f"Aligned {len(docs)} text chunks from Chroma. Best cosine similarity: {best_score:.4f}"
        })
        
        if not matched_sop and best_score < threshold:
            return {
                "architecture": "Workflow GraphRAG (Process DAG)",
                "answer": "Maaf, prosedur atau informasi terkait pertanyaan Anda tidak ditemukan dalam dokumen SOP resmi FSM UNDIP.",
                "sources": [],
                "contexts": [],
                "scores": [],
                "mermaid": "",
                "trace": trace,
                "latency": round(time.time() - start_time, 3)
            }
            
        # 3. Context assembly & synthesis
        doc_context_text = "\n\n---\n\n".join([d.page_content for d, _ in scored_results])
        prompt = self.prompt_template.format(
            workflow_context=workflow_context,
            document_context=doc_context_text,
            question=query_text
        )
        
        try:
            resp = self.model.invoke(prompt)
            answer = resp.content.strip()
        except Exception:
            # High-fidelity fallback
            answer = f"Berdasarkan alur resmi SOP FSM UNDIP untuk {matched_sop['title']}:\n\n"
            for s in matched_sop['steps']:
                answer += f"{s['step_num']}. **{s['actor']}**: {s['action']}\n"
                if s.get("output"):
                    answer += f"   - Luaran: *{s['output']}*\n"
                    
        sources = [d.metadata.get("id", "SOP.pdf") for d, _ in scored_results]
        contexts = [d.page_content for d, _ in scored_results]
        scores = [s for _, s in scored_results]
        
        trace.append({
            "step": "4. Procedural Workflow Synthesis",
            "type": "LLM Synthesis + BPMN Output",
            "detail": f"Generated chronologically verified procedural answer with linked visual diagram."
        })
        
        return {
            "architecture": "Workflow GraphRAG (Process DAG)",
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "scores": scores,
            "mermaid": mermaid_code,
            "matched_sop": matched_sop["title"] if matched_sop else None,
            "trace": trace,
            "latency": round(time.time() - start_time, 3)
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    rag = WorkflowGraphRAG()
    res = rag.query("Bagaimana urutan pengajuan cuti akademik?")
    print("Answer:\n", res["answer"][:300])
    print("Mermaid diagram exists:", bool(res["mermaid"]))
