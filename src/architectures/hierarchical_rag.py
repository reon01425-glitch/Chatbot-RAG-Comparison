import os
import time
from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from src.graph_sop.hierarchical_chunker import SOPHierarchicalChunker

CHROMA_PATH = "chroma"
EMBEDDING_MODEL_PATH = "./indo_finetuned_embedding"

PROMPT_TEMPLATE = """
Anda adalah asisten layanan mahasiswa Fakultas Sains dan Matematika Universitas Diponegoro (FSM UNDIP)
yang menyajikan informasi berbasis struktur pohon dokumen SOP (Hierarchical Document Tree).

Gunakan konteks hierarkis dokumen (Parent SOP -> Sub-Langkah / Leaf Details) di bawah ini:
{hierarchical_context}

---

Pertanyaan: {question}

Jawaban terstruktur sebagai asisten FSM UNDIP:
"""

class HierarchicalRAG:
    """
    Hierarchical / Tree-structured Chunking RAG.
    Maintains parent document structure and injects parent metadata into retrieved step leaves.
    """
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)
        model_name = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
        self.model = ChatOllama(model=model_name)
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chunker = SOPHierarchicalChunker()

    def query(self, query_text: str, k: int = 3, threshold: float = 0.3) -> Dict[str, Any]:
        start_time = time.time()
        trace = []
        
        # 1. Similarity search to locate relevant leaf chunks
        docs = self.db.similarity_search(query_text, k=k)
        query_emb = self.embedding_function.embed_query(query_text)
        
        scored_results = []
        for doc in docs:
            doc_emb = self.embedding_function.embed_query(doc.page_content)
            sim = float(cosine_similarity([query_emb], [doc_emb])[0][0])
            scored_results.append((doc, sim))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_score = scored_results[0][1] if scored_results else 0.0
        
        trace.append({
            "step": "1. Leaf Chunk Vector Retrieval",
            "type": "Bottom-Up Traversal",
            "detail": f"Retrieved top-{len(docs)} chunks. Best similarity: {best_score:.4f}"
        })
        
        if best_score < threshold:
            return {
                "architecture": "Hierarchical Tree RAG",
                "answer": "Maaf, rincian hierarki atau informasi terkait pertanyaan Anda tidak ditemukan dalam dokumen SOP resmi.",
                "sources": [],
                "contexts": [],
                "scores": [],
                "trace": trace,
                "latency": round(time.time() - start_time, 3)
            }
            
        # 2. Parent tree context expansion
        # Identify parent document from top hit
        top_doc = scored_results[0][0]
        source_file = os.path.basename(top_doc.metadata.get("source", ""))
        
        tree_context = ""
        if source_file in self.chunker.document_tree:
            tree_context = self.chunker.get_hierarchical_context_for_doc(source_file)
            trace.append({
                "step": "2. Parent Document Tree Expansion",
                "type": "Top-Down Context Injection",
                "detail": f"Expanded root node: {source_file} with its full procedural tree & leaf metadata."
            })
        else:
            tree_context = "\n\n---\n\n".join([d.page_content for d, _ in scored_results])
            
        # 3. LLM Generation
        prompt = self.prompt_template.format(
            hierarchical_context=tree_context,
            question=query_text
        )
        
        try:
            resp = self.model.invoke(prompt)
            answer = resp.content.strip()
        except Exception:
            answer = f"Berdasarkan hierarki dokumen SOP FSM UNDIP ({source_file}):\n\n"
            answer += tree_context[:600]
            
        sources = [d.metadata.get("id", source_file) for d, _ in scored_results]
        contexts = [d.page_content for d, _ in scored_results]
        scores = [s for _, s in scored_results]
        
        trace.append({
            "step": "3. Tree-Synthesized Response",
            "type": "Hierarchical Answer",
            "detail": f"Generated answer retaining complete hierarchical structure and attributes."
        })
        
        return {
            "architecture": "Hierarchical Tree RAG",
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "scores": scores,
            "parent_doc": source_file,
            "trace": trace,
            "latency": round(time.time() - start_time, 3)
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    rag = HierarchicalRAG()
    res = rag.query("Apa saja syarat legalisir ijazah?")
    print("Answer:\n", res["answer"][:300])
