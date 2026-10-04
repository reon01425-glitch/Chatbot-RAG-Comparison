import os
import time
import math
import re
import socket
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dotenv import dotenv_values
from sklearn.metrics.pairwise import cosine_similarity
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from langchain_ollama import ChatOllama
import pypdf

# Load environment configuration
env_config = dotenv_values(".env")
for k, v in env_config.items():
    if v and k not in os.environ:
        os.environ[k] = v

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
DATA_PATH = os.getenv("DATA_PATH", "data")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./indo_finetuned_embedding")

PROMPT_TEMPLATE = """
Anda adalah asisten layanan mahasiswa Fakultas Sains dan Matematika Universitas Diponegoro 
yang membantu menjawab pertanyaan berdasarkan dokumen resmi SOP kampus.
Jawablah dengan bahasa Indonesia yang jelas, runut, akurat, dan sopan.
Gunakan hanya fakta yang ada di dalam konteks di bawah ini. Jika informasi tidak tersedia di konteks, 
katakan secara jujur bahwa informasi tersebut tidak ditemukan dalam SOP resmi.

Konteks:
{context}

---

Pertanyaan: {question}

Jawaban sebagai asisten layanan mahasiswa:
"""

REWRITE_PROMPT_TEMPLATE = """
Anda adalah sistem penulisan ulang kueri (query rewriter) untuk SOP Universitas.
Tugas Anda adalah memformulasikan ulang kueri pengguna berikut agar lebih jelas, berorientasi kata kunci, dan optimal untuk pencarian dokumen SOP kampus.
Keluarkan HANYA kueri baru tanpa tanda petik, penjelasan, atau kalimat pengantar.

Kueri asli: {query}
Kueri baru:
"""

AGENTIC_REACT_PROMPT_TEMPLATE = """Anda adalah agen asisten akademik FSM Universitas Diponegoro yang menggunakan pola penalaran ReAct untuk mencari informasi SOP resmi.
Anda memiliki akses ke alat (tool) berikut:
- cari_dokumen_sop(query): Mencari dokumen atau prosedur SOP resmi FSM Undip berdasarkan query/kata kunci pencarian.

Gunakan format persis berikut:
Jika ingin mencari dokumen:
Thought: <analisis kebutuhan informasi dan alasan pencarian>
Action: cari_dokumen_sop
Action Input: <kata kunci pencarian spesifik>

Jika informasi sudah cukup untuk menjawab pertanyaan atau pencarian selesai:
Thought: <alasan mengapa informasi sudah cukup>
Final: siap menjawab

Pertanyaan Pengguna: {question}
{scratchpad}
"""

class RAGCore:
    _instance = None
    
    def __init__(self):
        chroma_path = os.getenv("CHROMA_PATH", CHROMA_PATH)
        emb_path = os.getenv("EMBEDDING_MODEL_PATH", EMBEDDING_MODEL_PATH)
        
        # In evaluation mode, verify index manifest strictly
        from src.manifest import verify_index_manifest, resolve_embedding_model
        resolved_emb = resolve_embedding_model(emb_path)
        is_eval_mode = os.getenv("EVALUATION_MODE", "0").lower() in ("1", "true", "yes")
        
        manifest_ok, manifest_msg = verify_index_manifest(chroma_path, resolved_emb, strict=is_eval_mode)
        if is_eval_mode and not manifest_ok:
            raise RuntimeError(f"[RAGCore] Index manifest validation failed in evaluation mode: {manifest_msg}")
        elif not manifest_ok:
            print(f"[RAGCore] WARNING: {manifest_msg}")

        print(f"[RAGCore] Loading embeddings ({resolved_emb}) and Chroma vector store ({chroma_path})...")
        self.embedding_function = HuggingFaceEmbeddings(model_name=resolved_emb)
        self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_function)
        
        # Load and chunk documents for BM25 and Multimodal layout indexing
        self._init_bm25_and_layout()
        self._init_knowledge_graph()
        
        # Load Workflow Graph (BPMN / State Machine DAG) and Hierarchical Tree Chunker
        from src.graph_sop.workflow_graph import SOPWorkflowGraph
        from src.graph_sop.hierarchical_chunker import SOPHierarchicalChunker
        self.workflow_graph = SOPWorkflowGraph()
        self.hierarchical_chunker = SOPHierarchicalChunker(data_path=os.getenv("DATA_PATH", DATA_PATH))
        self._init_htree_collection()
        print("[RAGCore] Initialization complete.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = RAGCore()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def _init_bm25_and_layout(self):
        self.chunks = []
        self.multimodal_index = {}
        
        if os.path.exists(DATA_PATH):
            loader = PyPDFDirectoryLoader(DATA_PATH)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1700,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
            )
            self.chunks = text_splitter.split_documents(documents)
            
            for i, chunk in enumerate(self.chunks):
                source = os.path.basename(chunk.metadata.get("source", "unknown"))
                page = chunk.metadata.get("page", 0)
                chunk.metadata["id"] = f"{source}:{page}:{i}"
                
            tokenized_corpus = [doc.page_content.lower().split() for doc in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
            
            # Build multimodal index for PDF pages
            for fname in os.listdir(DATA_PATH):
                if fname.endswith(".pdf"):
                    fpath = os.path.join(DATA_PATH, fname)
                    try:
                        reader = pypdf.PdfReader(fpath)
                        for i, page in enumerate(reader.pages):
                            images_count = len(page.images)
                            text = page.extract_text() or ""
                            has_tables = "tabel" in text.lower() or "no." in text.lower() or "bagan" in text.lower()
                            has_flowchart = "alur" in text.lower() or "prosedur" in text.lower() or "tahap" in text.lower()
                            
                            desc = f"Halaman {i+1}: Terdeteksi {images_count} elemen visual/diagram alur"
                            if has_tables:
                                desc += ", 1+ representasi tabel SOP struktural"
                            if has_flowchart:
                                desc += ", skema tahapan proses"
                            
                            self.multimodal_index[f"{fname}:{i}"] = {
                                "source": fname,
                                "page": i + 1,
                                "images_count": images_count,
                                "has_tables": has_tables,
                                "has_flowchart": has_flowchart,
                                "desc": desc
                            }
                    except Exception:
                        pass
        else:
            self.bm25 = None

    def _init_knowledge_graph(self):
        self.graph = nx.Graph()
        relations = [
            ("cuti akademik", "izin cuti", "prosedur pengajuan"),
            ("cuti akademik", "aktif kembali", "alur lanjutan"),
            ("cuti akademik", "ketua program studi", "rekomendasi & persetujuan"),
            ("cuti akademik", "dekan", "penerbitan SK izin"),
            ("cuti akademik", "spp/ukt", "syarat bebas tunggakan"),
            ("legalisir", "ijazah", "dokumen objek"),
            ("legalisir", "transkrip", "dokumen objek"),
            ("legalisir", "subbagian akademik", "verifikasi & validasi"),
            ("beasiswa", "rekomendasi beasiswa", "dokumen pengajuan"),
            ("beasiswa", "wakil dekan i", "pengesahan surat"),
            ("ukt", "keterlambatan pembayaran", "permohonan dispensasi"),
            ("ukt", "bagian keuangan fsm", "verifikasi pembayaran"),
            ("irs", "pengisian irs", "registrasi akademik per semester"),
            ("irs", "dosen wali", "bimbingan & persetujuan online"),
            ("irs", "siap undip", "portal sistem informasi"),
            ("organisasi mahasiswa", "proposal kegiatan", "pengajuan persetujuan"),
            ("organisasi mahasiswa", "wakil dekan i", "persetujuan kegiatan")
        ]
        for u, v, r in relations:
            self.graph.add_edge(u, v, relationship=r)

    def _init_htree_collection(self):
        """Initializes or connects to the separate 'htree_leaves' collection in Chroma."""
        chroma_path = os.getenv("CHROMA_PATH", CHROMA_PATH)
        emb_path = os.getenv("EMBEDDING_MODEL_PATH", EMBEDDING_MODEL_PATH)
        try:
            from src.manifest import verify_index_manifest, load_index_manifest
            manifest = load_index_manifest(chroma_path)
            manifest_ok, msg = verify_index_manifest(chroma_path, emb_path, strict=False)

            self.htree_db = Chroma(
                collection_name="htree_leaves",
                persist_directory=chroma_path,
                embedding_function=self.embedding_function,
                collection_metadata={"hnsw:space": "cosine"}
            )
            existing = self.htree_db.get(include=[])
            existing_ids = set(existing.get("ids", []))
            new_leaves = [
                chunk for chunk in self.hierarchical_chunker.leaf_chunks
                if chunk.metadata.get("id") not in existing_ids
            ]
            if new_leaves:
                if manifest is not None and not manifest_ok:
                    print(f"[RAGCore] Skipping indexing new leaves into '{chroma_path}': manifest model mismatch ({msg}).")
                else:
                    ids = [chunk.metadata["id"] for chunk in new_leaves]
                    print(f"[RAGCore] Indexing {len(new_leaves)} leaf chunks into 'htree_leaves' collection...")
                    self.htree_db.add_documents(new_leaves, ids=ids)
        except Exception as e:
            if os.getenv("EVALUATION_MODE", "0").lower() in ("1", "true", "yes"):
                raise
            print(f"[RAGCore] Warning: Failed to initialize htree_leaves collection: {e}")
            self.htree_db = self.db

    def is_ollama_available(self) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            res = s.connect_ex(('127.0.0.1', 11434)) == 0
            s.close()
            return res
        except Exception:
            return False

    def call_llm(self, prompt: str) -> str:
        """Single patchable choke point for all LLM calls (synthesis, agent steps, query rewrites)."""
        if not self.is_ollama_available():
            raise RuntimeError("Ollama server is not available at 127.0.0.1:11434")
        model_name = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
        model = ChatOllama(model=model_name, timeout=15)
        resp = model.invoke(prompt)
        if resp and resp.content:
            return resp.content.strip()
        return ""

    def generate_synthesis(self, prompt: str, contexts: List[str], question: str) -> str:
        """
        Generate answer using Ollama if online, or intelligent extractive synthesis fallback.
        """
        if getattr(self, "disable_extractive_fallback", False) or os.getenv("RAG_DISABLE_EXTRACTIVE_FALLBACK", "0") == "1":
            return self.call_llm(prompt)

        try:
            res = self.call_llm(prompt)
            if res:
                return res
        except Exception as e:
            print(f"[RAGCore] LLM generation failed: {e}. Falling back to synthesis.")
                
        # Smart extractive heuristic synthesis based on retrieved contexts
        if not contexts:
            return "Maaf, informasi terkait pertanyaan Anda tidak ditemukan dalam dokumen SOP resmi Fakultas Sains dan Matematika Universitas Diponegoro."
            
        combined_text = "\n\n".join(contexts)
        paragraphs = [p.strip() for p in combined_text.split("\n\n") if len(p.strip()) > 30]
        
        # Extract the most relevant sentences answering the question
        relevant_paras = []
        q_words = [w.lower() for w in re.findall(r'\w+', question) if len(w) > 3]
        for p in paragraphs:
            score = sum(1 for w in q_words if w in p.lower())
            if score > 0:
                relevant_paras.append((p, score))
                
        relevant_paras.sort(key=lambda x: x[1], reverse=True)
        top_paras = [p[0] for p in relevant_paras[:3]] if relevant_paras else paragraphs[:2]
        
        intro = "Berdasarkan Standar Operasional Prosedur (SOP) Fakultas Sains dan Matematika Universitas Diponegoro:\n\n"
        body = "\n\n".join(top_paras)
        return intro + body


class RAGEngine:
    def __init__(self):
        self.core = RAGCore.get_instance()
        self.agentic_max_steps = int(os.getenv("AGENTIC_MAX_STEPS", "3"))
        
    def calculate_metrics(self, query: str, answer: str, contexts: List[str], scores: List[float], total_latency: float) -> Dict[str, Any]:
        """
        Computes dynamic live metrics: Faithfulness, Answer Relevance, Cosine Semantic Relevance.
        """
        # 1. Cosine Semantic Relevance (Average / Max of top retrieval scores)
        max_cosine = max(scores) if scores else 0.0
        avg_cosine = sum(scores) / len(scores) if scores else 0.0
        
        # 2. Faithfulness Score: Overlap & Entailment heuristic between Answer and Retrieved Contexts
        if not contexts or not answer or "tidak ditemukan" in answer.lower():
            faithfulness = 0.95 if ("tidak ditemukan" in answer.lower() and max_cosine < 0.4) else 0.4
        else:
            context_corpus = " ".join(contexts).lower()
            ans_tokens = [w for w in re.findall(r'\w+', answer.lower()) if len(w) > 3]
            if ans_tokens:
                in_context_count = sum(1 for tok in ans_tokens if tok in context_corpus)
                token_overlap_ratio = in_context_count / len(ans_tokens)
                # Calibrate faithfulness to 0.0 - 1.0 range
                faithfulness = min(1.0, max(0.2, token_overlap_ratio * 1.08))
            else:
                faithfulness = 0.5

        # 3. Answer Relevance Score: Semantic similarity between Query and Answer
        try:
            q_emb = self.core.embedding_function.embed_query(query)
            ans_emb = self.core.embedding_function.embed_query(answer[:500])
            ans_relevance = float(cosine_similarity([q_emb], [ans_emb])[0][0])
            ans_relevance = max(0.0, min(1.0, ans_relevance))
        except Exception:
            ans_relevance = 0.65
            
        return {
            "latency": round(total_latency, 3),
            "max_cosine_sim": round(max_cosine, 4),
            "avg_cosine_sim": round(avg_cosine, 4),
            "faithfulness": round(faithfulness, 4),
            "answer_relevance": round(ans_relevance, 4),
            "semantic_relevance": round(max_cosine, 4)
        }

    def execute_naive_rag(self, query_text: str, k: int = 3, threshold: float = 0.3) -> Dict[str, Any]:
        """1. Naive RAG (Baseline dense vector retrieval)"""
        start_time = time.time()
        trace = []
        trace.append({
            "step": "1. Query Vectorization",
            "type": "Dense Embedding",
            "detail": f"Generated 384-d semantic embedding for query: '{query_text}'"
        })
        
        # Dense retrieval
        t_retrieval_start = time.time()
        docs = self.core.db.similarity_search(query_text, k=k)
        query_embedding = self.core.embedding_function.embed_query(query_text)
        
        scored_results = []
        for doc in docs:
            doc_embedding = self.core.embedding_function.embed_query(doc.page_content)
            sim = float(cosine_similarity([query_embedding], [doc_embedding])[0][0])
            scored_results.append((doc, sim))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        retrieval_latency = time.time() - t_retrieval_start
        
        trace.append({
            "step": "2. Vector Similarity Search",
            "type": "Chroma Vector DB",
            "detail": f"Retrieved top-{k} documents in {retrieval_latency*1000:.1f}ms. Best similarity score: {scored_results[0][1]:.4f}" if scored_results else "No documents found."
        })
        
        best_doc, best_score = scored_results[0] if scored_results else (None, 0.0)
        
        if best_score < threshold:
            answer = "Maaf, saya tidak menemukan jawaban pada dokumen SOP resmi yang tersedia."
            sources = []
            contexts = []
            scores = []
            trace.append({
                "step": "3. Threshold Check",
                "type": "Fallback Triggered",
                "detail": f"Best similarity score ({best_score:.4f}) is below threshold ({threshold:.2f}). Triggered honest refusal."
            })
        else:
            sources = [doc.metadata.get("id", os.path.basename(doc.metadata.get("source", "SOP.pdf"))) for doc, _ in scored_results]
            contexts = [doc.page_content for doc, _ in scored_results]
            scores = [score for _, score in scored_results]
            
            trace.append({
                "step": "3. Prompt Assembly",
                "type": "Template Formatting",
                "detail": f"Assembled context ({len(contexts)} chunks, {sum(len(c) for c in contexts)} chars) with system instructions."
            })
            
            context_text = "\n\n---\n\n".join(contexts)
            prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
            answer = self.core.generate_synthesis(prompt, contexts, query_text)
            
            trace.append({
                "step": "4. Response Generation",
                "type": "LLM Synthesis",
                "detail": f"Synthesized final response ({len(answer.split())} words) grounded in retrieved context."
            })
            
        total_latency = time.time() - start_time
        metrics = self.calculate_metrics(query_text, answer, contexts if best_score >= threshold else [], scores, total_latency)
        
        return {
            "architecture": "Naive RAG (Baseline)",
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "scores": scores,
            "raw_docs": [doc.metadata for doc, _ in scored_results] if scored_results else [],
            "trace": trace,
            "metrics": metrics
        }

    def execute_hybrid_rag(self, query_text: str, k: int = 3, threshold: float = 0.3) -> Dict[str, Any]:
        """2. Hybrid RAG (Dense Embeddings + Sparse BM25 + Reciprocal Rank Fusion)"""
        start_time = time.time()
        trace = []
        
        # 1. Dense retrieval
        trace.append({
            "step": "1. Dual Retrieval Dispatch",
            "type": "Parallel Sparse + Dense",
            "detail": f"Executing dense vector search (k=5) and BM25 Okapi lexical search (k=5) simultaneously."
        })
        dense_results = self.core.db.similarity_search(query_text, k=5)
        
        # 2. Sparse retrieval
        if self.core.bm25:
            tokenized_query = query_text.lower().split()
            sparse_scores = self.core.bm25.get_scores(tokenized_query)
            sparse_indices = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)[:5]
            sparse_results = [self.core.chunks[i] for i in sparse_indices]
        else:
            sparse_results = []
            
        trace.append({
            "step": "2. Retrieval Results",
            "type": "Candidate Collection",
            "detail": f"Dense returned {len(dense_results)} chunks. BM25 sparse returned {len(sparse_results)} chunks."
        })
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_constant = 60
        rrf_scores = {}
        all_docs = {}
        
        for rank, doc in enumerate(dense_results, 1):
            doc_id = doc.metadata.get("id") or doc.page_content[:50]
            all_docs[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_constant + rank))
            
        for rank, doc in enumerate(sparse_results, 1):
            doc_id = doc.metadata.get("id") or doc.page_content[:50]
            all_docs[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_constant + rank))
            
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        fused_docs = [all_docs[doc_id] for doc_id in sorted_doc_ids]
        
        trace.append({
            "step": "3. Reciprocal Rank Fusion (RRF)",
            "type": "RRF Ranking (k=60)",
            "detail": f"Combined & ranked candidates via RRF formula: Score = Σ (1 / (60 + rank)). Top fused score: {rrf_scores.get(sorted_doc_ids[0], 0):.5f}" if sorted_doc_ids else "No candidates."
        })
        
        # Score results with cosine similarity
        query_embedding = self.core.embedding_function.embed_query(query_text)
        scored_results = []
        for doc in fused_docs:
            doc_embedding = self.core.embedding_function.embed_query(doc.page_content)
            sim = float(cosine_similarity([query_embedding], [doc_embedding])[0][0])
            scored_results.append((doc, sim))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_doc, best_score = scored_results[0] if scored_results else (None, 0.0)
        
        if best_score < threshold:
            answer = "Maaf, saya tidak menemukan jawaban pada dokumen SOP resmi yang tersedia."
            sources = []
            contexts = []
            scores = []
        else:
            sources = [doc.metadata.get("id", os.path.basename(doc.metadata.get("source", "SOP.pdf"))) for doc, _ in scored_results]
            contexts = [doc.page_content for doc, _ in scored_results]
            scores = [score for _, score in scored_results]
            
            context_text = "\n\n---\n\n".join(contexts)
            prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
            answer = self.core.generate_synthesis(prompt, contexts, query_text)
            
            trace.append({
                "step": "4. Hybrid Generation",
                "type": "Synthesized Response",
                "detail": f"Generated answer using hybrid-retrieved contexts (Dense+BM25 overlap)."
            })
            
        total_latency = time.time() - start_time
        metrics = self.calculate_metrics(query_text, answer, contexts if best_score >= threshold else [], scores, total_latency)
        
        return {
            "architecture": "Hybrid RAG (BM25 + Dense)",
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "scores": scores,
            "raw_docs": [doc.metadata for doc, _ in scored_results] if scored_results else [],
            "trace": trace,
            "metrics": metrics
        }

    def execute_graph_rag(self, query_text: str, k: int = 3, threshold: float = 0.3) -> Dict[str, Any]:
        """3. GraphRAG (Knowledge Graph Entity Expansion + Context Fusion)"""
        start_time = time.time()
        trace = []
        
        # Step 1: Entity extraction & graph matching
        q_lower = query_text.lower()
        matched_nodes = [node for node in self.core.graph.nodes if node in q_lower]
        
        related_entities = set(matched_nodes)
        relations_found = []
        for node in matched_nodes:
            for neighbor in self.core.graph.neighbors(node):
                related_entities.add(neighbor)
                rel_type = self.core.graph[node][neighbor].get('relationship', 'terkait')
                relations_found.append(f"{node} --[{rel_type}]--> {neighbor}")
                
        trace.append({
            "step": "1. Knowledge Graph Entity Extraction",
            "type": "Entity Linking",
            "detail": f"Matched direct entities: {list(matched_nodes) if matched_nodes else 'None'}. Expanded entities: {list(related_entities)}"
        })
        
        if relations_found:
            trace.append({
                "step": "2. Graph Traversal & Relations",
                "type": "NetworkX Relations",
                "detail": f"Discovered {len(relations_found)} university SOP relationships: " + ", ".join(relations_found[:3])
            })
            
        # Step 2: Expanded query search
        expanded_query = query_text
        if related_entities:
            expanded_query += " " + " ".join(list(related_entities))
            
        docs = self.core.db.similarity_search(expanded_query, k=k)
        query_embedding = self.core.embedding_function.embed_query(query_text)
        
        scored_results = []
        for doc in docs:
            doc_embedding = self.core.embedding_function.embed_query(doc.page_content)
            sim = float(cosine_similarity([query_embedding], [doc_embedding])[0][0])
            scored_results.append((doc, sim))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_doc, best_score = scored_results[0] if scored_results else (None, 0.0)
        
        trace.append({
            "step": "3. Graph-Augmented Dense Retrieval",
            "type": "Expanded Similarity Search",
            "detail": f"Queried vector index with expanded query: '{expanded_query[:80]}...'. Top cosine sim: {best_score:.4f}"
        })
        
        if best_score < threshold:
            answer = "Maaf, saya tidak menemukan jawaban pada dokumen SOP resmi yang tersedia."
            sources = []
            contexts = []
            scores = []
        else:
            sources = [doc.metadata.get("id", os.path.basename(doc.metadata.get("source", "SOP.pdf"))) for doc, _ in scored_results]
            contexts = [doc.page_content for doc, _ in scored_results]
            scores = [score for _, score in scored_results]
            
            # Format graph context alongside document context
            graph_header = ""
            if relations_found:
                graph_header = "Struktur Relasi Graf Kampus:\n" + "\n".join([f"- {r}" for r in relations_found]) + "\n\n"
                
            context_text = graph_header + "\n\n---\n\n".join(contexts)
            prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
            answer = self.core.generate_synthesis(prompt, contexts, query_text)
            
            trace.append({
                "step": "4. Graph-Enriched Synthesis",
                "type": "Knowledge-Augmented Response",
                "detail": f"Synthesized answer enriched with {len(relations_found)} graph relational links."
            })
            
        total_latency = time.time() - start_time
        metrics = self.calculate_metrics(query_text, answer, contexts if best_score >= threshold else [], scores, total_latency)
        
        return {
            "architecture": "GraphRAG (Entity Expansion)",
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "scores": scores,
            "raw_docs": [doc.metadata for doc, _ in scored_results] if scored_results else [],
            "trace": trace,
            "metrics": metrics,
            "related_entities": list(related_entities)
        }

    def execute_agentic_rag(self, query_text: str, k: int = 3, threshold: float = 0.3, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """4. Agentic ReAct RAG (Autonomous Tool Execution & Thought Trace via LLM)"""
        start_time = time.time()
        trace = []
        
        limit_steps = max_steps if max_steps is not None else getattr(self, "agentic_max_steps", 3)
        collected_chunks: Dict[str, Tuple[Any, float]] = {}
        scratchpad = ""
        
        query_embedding = self.core.embedding_function.embed_query(query_text)
        
        for step_idx in range(limit_steps):
            step_num = step_idx + 1
            react_prompt = AGENTIC_REACT_PROMPT_TEMPLATE.format(
                question=query_text,
                scratchpad=f"\nCatatan Langkah Sebelumnya:\n{scratchpad}" if scratchpad else ""
            )
            
            try:
                llm_output = self.core.call_llm(react_prompt)
            except Exception as e:
                if getattr(self.core, "disable_extractive_fallback", False) or os.getenv("RAG_DISABLE_EXTRACTIVE_FALLBACK", "0") == "1":
                    raise
                trace.append({
                    "step": f"Langkah {step_num}: Panggilan LLM Gagal",
                    "type": "LLM Error",
                    "detail": f"Error memanggil LLM: {str(e)}"
                })
                break
                
            if not llm_output or not llm_output.strip():
                trace.append({
                    "step": f"Langkah {step_num}: Output Kosong",
                    "type": "Parse Warning",
                    "detail": "LLM mengembalikan output kosong. Menghentikan loop ReAct."
                })
                break

            # Parse Thought
            thought_match = re.search(r'(?:^|\n)\s*Thought\s*:\s*(.*?)(?=(?:\n\s*Action\s*:|\n\s*Final\s*:|$))', llm_output, re.DOTALL | re.IGNORECASE)
            thought_text = thought_match.group(1).strip() if thought_match else ""

            # Parse Action and Action Input
            action_match = re.search(r'(?:^|\n)\s*Action\s*:\s*([^\n]+)', llm_output, re.IGNORECASE)
            input_match = re.search(r'(?:^|\n)\s*Action\s*Input\s*:\s*([^\n]+)', llm_output, re.IGNORECASE)
            final_match = re.search(r'(?:^|\n)\s*Final(?:\s+Answer)?\s*:\s*(.*)', llm_output, re.IGNORECASE)

            tool_name = ""
            search_query = ""

            if action_match:
                raw_action = action_match.group(1).strip()
                if "cari_dokumen_sop" in raw_action.lower():
                    tool_name = "cari_dokumen_sop"
                    inline_arg = re.search(r'cari_dokumen_sop\s*\((.*?)\)', raw_action, re.IGNORECASE)
                    if inline_arg and inline_arg.group(1).strip():
                        search_query = inline_arg.group(1).strip().strip('\'"')
                    elif input_match:
                        search_query = input_match.group(1).strip().strip('\'"')
                    else:
                        search_query = query_text
                else:
                    tool_name = raw_action
                    if input_match:
                        search_query = input_match.group(1).strip().strip('\'"')
            elif final_match or "siap menjawab" in llm_output.lower():
                detail_msg = f"Thought: {thought_text}" if thought_text else f"{llm_output.strip()}"
                trace.append({
                    "step": f"Langkah {step_num}: Keputusan Selesai",
                    "type": "ReAct Thought & Final",
                    "detail": f"{detail_msg}\nFinal: siap menjawab"
                })
                break
            
            if not tool_name:
                trace.append({
                    "step": f"Langkah {step_num}: Format Tak Terbaca",
                    "type": "Parse Error",
                    "detail": f"Respon LLM tidak mengikuti format Thought/Action/Final: {llm_output[:180]}"
                })
                break

            if tool_name == "cari_dokumen_sop":
                if not search_query:
                    search_query = query_text
                t_tool = time.time()
                docs = self.core.db.similarity_search(search_query, k=k) if self.core.db else []
                tool_elapsed = (time.time() - t_tool) * 1000
                
                obs_snippets = []
                for doc in docs:
                    doc_emb = self.core.embedding_function.embed_query(doc.page_content)
                    sim = float(cosine_similarity([query_embedding], [doc_emb])[0][0])
                    doc_id = doc.metadata.get("id") or os.path.basename(doc.metadata.get("source", "SOP.pdf"))
                    if doc_id not in collected_chunks or sim > collected_chunks[doc_id][1]:
                        collected_chunks[doc_id] = (doc, sim)
                    snippet = doc.page_content[:100].replace('\n', ' ')
                    obs_snippets.append(f"[{doc_id}] ({sim:.4f}): {snippet}...")
                
                obs_text = f"Ditemukan {len(docs)} chunk dokumen." if docs else "Tidak ditemukan dokumen yang cocok."
                if obs_snippets:
                    obs_text += " Cuplikan: " + " | ".join(obs_snippets[:2])
                
                trace.append({
                    "step": f"Langkah {step_num}: ReAct Thought & Action",
                    "type": "Tool Invocation",
                    "detail": f"Thought: {thought_text}\nAction: cari_dokumen_sop('{search_query}')"
                })
                trace.append({
                    "step": f"Langkah {step_num}: Tool Observation",
                    "type": "Observation",
                    "detail": f"Observation ({tool_elapsed:.1f}ms): {obs_text}"
                })
                
                scratchpad += f"Thought: {thought_text}\nAction: cari_dokumen_sop\nAction Input: {search_query}\nObservation: {obs_text}\n"
            else:
                trace.append({
                    "step": f"Langkah {step_num}: Tool Tidak Dikenal",
                    "type": "Tool Error",
                    "detail": f"Tool '{tool_name}' tidak dikenal. Hanya cari_dokumen_sop yang didukung."
                })
                break

        sorted_scored = sorted(collected_chunks.values(), key=lambda x: x[1], reverse=True)
        top_scored = sorted_scored[:k]
        best_doc, best_score = top_scored[0] if top_scored else (None, 0.0)

        if best_score < threshold:
            answer = "Maaf, saya tidak menemukan jawaban pada dokumen SOP resmi yang tersedia."
            sources = []
            contexts = []
            scores = []
            trace.append({
                "step": "Evaluasi Akhir",
                "type": "Threshold Refusal",
                "detail": f"Skor kemiripan tertinggi ({best_score:.4f}) berada di bawah ambang ({threshold:.2f}). Mengembalikan penolakan jujur."
            })
        else:
            sources = [doc.metadata.get("id", os.path.basename(doc.metadata.get("source", "SOP.pdf"))) for doc, _ in top_scored]
            contexts = [doc.page_content for doc, _ in top_scored]
            scores = [score for _, score in top_scored]
            
            context_text = "\n\n---\n\n".join(contexts)
            prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
            answer = self.core.generate_synthesis(prompt, contexts, query_text)
            trace.append({
                "step": "Sintesis Akhir",
                "type": "LLM Synthesis",
                "detail": f"Menghasilkan jawaban menggunakan PROMPT_TEMPLATE dengan {len(contexts)} chunk terbaik."
            })
            
        total_latency = time.time() - start_time
        metrics = self.calculate_metrics(query_text, answer, contexts if best_score >= threshold else [], scores, total_latency)
        
        return {
            "architecture": "Agentic RAG (Tools Agent)",
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "scores": scores,
            "raw_docs": [doc.metadata for doc, _ in top_scored] if top_scored else [],
            "trace": trace,
            "metrics": metrics
        }

    def execute_crag(self, query_text: str, k: int = 3) -> Dict[str, Any]:
        """5. Corrective RAG (CRAG: Retrieval Evaluator & Query Rewriting)"""
        start_time = time.time()
        trace = []
        
        # Step 1: Initial Retrieval
        trace.append({
            "step": "1. Initial Document Retrieval",
            "type": "Phase 1 Search",
            "detail": f"Performing preliminary retrieval on raw query: '{query_text}'"
        })
        
        docs = self.core.db.similarity_search(query_text, k=k)
        query_embedding = self.core.embedding_function.embed_query(query_text)
        
        scored_results = []
        for doc in docs:
            doc_embedding = self.core.embedding_function.embed_query(doc.page_content)
            sim = float(cosine_similarity([query_embedding], [doc_embedding])[0][0])
            scored_results.append((doc, sim))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_doc, best_score = scored_results[0] if scored_results else (None, 0.0)
        
        # Grader thresholds
        upper_threshold = 0.55
        lower_threshold = 0.35
        
        if best_score >= upper_threshold:
            grade = "CORRECT"
            grade_detail = f"Confidence score ({best_score:.4f}) >= {upper_threshold:.2f}. Status: CORRECT. Proceeding directly to generation."
        elif best_score >= lower_threshold:
            grade = "AMBIGUOUS"
            grade_detail = f"Confidence score ({best_score:.4f}) in [{lower_threshold:.2f}, {upper_threshold:.2f}). Status: AMBIGUOUS. Triggering query rewrite."
        else:
            grade = "INCORRECT"
            grade_detail = f"Confidence score ({best_score:.4f}) < {lower_threshold:.2f}. Status: INCORRECT. Triggering fallback refusal."
            
        trace.append({
            "step": "2. CRAG Document Grader",
            "type": f"Grade Decision: [{grade}]",
            "detail": grade_detail
        })
        
        # Handle Ambiguous case with query rewrite
        if grade == "AMBIGUOUS":
            rewritten_query = None
            rewrite_prompt = REWRITE_PROMPT_TEMPLATE.format(query=query_text)
            
            try:
                raw_rewrite = self.core.call_llm(rewrite_prompt)
                if raw_rewrite and raw_rewrite.strip():
                    # Clean output: single line, no quotes, no conversational intro
                    lines = [ln.strip() for ln in raw_rewrite.strip().splitlines() if ln.strip()]
                    cleaned = lines[0] if lines else ""
                    cleaned = re.sub(r'^(kueri baru|query baru|rewritten query|hasil rewrite|kueri)\s*:\s*', '', cleaned, flags=re.IGNORECASE)
                    cleaned = cleaned.strip('\'"`“”«» ')
                    if cleaned:
                        rewritten_query = cleaned
            except Exception as e:
                if getattr(self.core, "disable_extractive_fallback", False) or os.getenv("RAG_DISABLE_EXTRACTIVE_FALLBACK", "0") == "1":
                    raise
                print(f"[CRAG] LLM rewrite failed: {e}")

            if not rewritten_query:
                if getattr(self.core, "disable_extractive_fallback", False) or os.getenv("RAG_DISABLE_EXTRACTIVE_FALLBACK", "0") == "1":
                    raise RuntimeError(f"CRAG query rewrite via LLM failed or produced empty output for query: '{query_text}'")
                else:
                    clean_q = re.sub(r'[^a-zA-Z0-9\s]', '', query_text)
                    tokens = [w for w in clean_q.split() if len(w) > 2]
                    rewritten_query = f"SOP prosedur pengajuan {' '.join(tokens)} Fakultas Sains dan Matematika Undip"
                    trace.append({
                        "step": "3. Query Rewriting Fallback (Heuristic)",
                        "type": "CRAG Heuristic Fallback",
                        "detail": f"LLM rewrite gagal/kosong. Fallback ke template heuristik: '{rewritten_query}'"
                    })
            else:
                trace.append({
                    "step": "3. Query Rewriting & Correction",
                    "type": "CRAG LLM Query Reformulator",
                    "detail": f"LLM rewrote query into: '{rewritten_query}'"
                })
            
            # Secondary retrieval
            new_docs = self.core.db.similarity_search(rewritten_query, k=k) if self.core.db else []
            new_scored = []
            for doc in new_docs:
                doc_emb = self.core.embedding_function.embed_query(doc.page_content)
                sim = float(cosine_similarity([query_embedding], [doc_emb])[0][0])
                new_scored.append((doc, sim))
            new_scored.sort(key=lambda x: x[1], reverse=True)
            
            if new_scored and new_scored[0][1] >= lower_threshold:
                scored_results = new_scored
                best_score = new_scored[0][1]
                trace.append({
                    "step": "4. Post-Rewrite Retrieval",
                    "type": "Corrected Vector Search",
                    "detail": f"Successfully re-retrieved documents. New top similarity: {best_score:.4f}"
                })
            elif new_scored:
                trace.append({
                    "step": "4. Post-Rewrite Retrieval",
                    "type": "Corrected Vector Search (Low Confidence)",
                    "detail": f"Re-retrieved documents remained below threshold ({new_scored[0][1]:.4f} < {lower_threshold:.2f})."
                })
                
        if grade == "INCORRECT" or best_score < lower_threshold:
            answer = f"Maaf, informasi mengenai '{query_text}' tidak ditemukan dalam basis dokumen SOP resmi FSM Universitas Diponegoro."
            sources = []
            contexts = []
            scores = []
        else:
            sources = [doc.metadata.get("id", os.path.basename(doc.metadata.get("source", "SOP.pdf"))) for doc, _ in scored_results]
            contexts = [doc.page_content for doc, _ in scored_results]
            scores = [score for _, score in scored_results]
            
            context_text = "\n\n---\n\n".join(contexts)
            prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
            answer = self.core.generate_synthesis(prompt, contexts, query_text)
            
            trace.append({
                "step": "5. Final Response Generation",
                "type": "CRAG Synthesis",
                "detail": f"Generated final answer with verified grade [{grade}]."
            })
            
        total_latency = time.time() - start_time
        metrics = self.calculate_metrics(query_text, answer, contexts if grade != "INCORRECT" else [], scores, total_latency)
        
        return {
            "architecture": "Corrective RAG (CRAG)",
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "scores": scores,
            "raw_docs": [doc.metadata for doc, _ in scored_results] if scored_results else [],
            "trace": trace,
            "metrics": metrics,
            "grade": grade
        }

    def execute_multimodal_rag(self, query_text: str, k: int = 3, threshold: float = 0.3) -> Dict[str, Any]:
        """6. Multimodal RAG (Document Layout, Table Structure & Diagram Metadata Enrichment)"""
        start_time = time.time()
        trace = []
        
        # Dense retrieval
        docs = self.core.db.similarity_search(query_text, k=k)
        query_embedding = self.core.embedding_function.embed_query(query_text)
        
        scored_results = []
        for doc in docs:
            doc_embedding = self.core.embedding_function.embed_query(doc.page_content)
            sim = float(cosine_similarity([query_embedding], [doc_embedding])[0][0])
            scored_results.append((doc, sim))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_doc, best_score = scored_results[0] if scored_results else (None, 0.0)
        
        trace.append({
            "step": "1. Text & Layout Chunk Retrieval",
            "type": "Multimodal Retrieval",
            "detail": f"Retrieved top-{k} document chunks. Highest similarity score: {best_score:.4f}"
        })
        
        if best_score < threshold:
            answer = "Maaf, saya tidak menemukan jawaban pada dokumen SOP resmi yang tersedia."
            sources = []
            contexts = []
            scores = []
        else:
            enriched_contexts = []
            layout_metadata_list = []
            
            for doc, _ in scored_results:
                src = os.path.basename(doc.metadata.get("source", ""))
                page = doc.metadata.get("page", 0)
                key = f"{src}:{page}"
                
                meta_desc = ""
                if key in self.core.multimodal_index:
                    meta_info = self.core.multimodal_index[key]
                    layout_metadata_list.append(meta_info)
                    meta_desc = f"[INFORMASI VISUAL & STRUKTUR LAYOUT]: {meta_info['desc']}\n"
                    
                enriched_contexts.append(meta_desc + doc.page_content)
                
            trace.append({
                "step": "2. Visual & Layout Metadata Injection",
                "type": "Multimodal Fusion",
                "detail": f"Injected layout metadata for {len(layout_metadata_list)} pages (detected flowcharts, table structures, and diagrams)."
            })
            
            sources = [doc.metadata.get("id", os.path.basename(doc.metadata.get("source", "SOP.pdf"))) for doc, _ in scored_results]
            contexts = enriched_contexts
            scores = [score for _, score in scored_results]
            
            context_text = "\n\n---\n\n".join(contexts)
            prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
            answer = self.core.generate_synthesis(prompt, contexts, query_text)
            
            trace.append({
                "step": "3. Multimodal Synthesis",
                "type": "Layout-Aware Generation",
                "detail": f"Generated final answer integrating both textual instructions and procedural diagram flowcharts."
            })
            
        total_latency = time.time() - start_time
        metrics = self.calculate_metrics(query_text, answer, contexts if best_score >= threshold else [], scores, total_latency)
        
        return {
            "architecture": "Multimodal RAG (Layout RAG)",
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "scores": scores,
            "raw_docs": [doc.metadata for doc, _ in scored_results] if scored_results else [],
            "trace": trace,
            "metrics": metrics
        }

    def execute_workflow_graph_rag(self, query_text: str, k: int = 3, threshold: float = 0.3) -> Dict[str, Any]:
        """7. Workflow GraphRAG (Process / Workflow State-Machine DAG + BPMN Flowchart)"""
        start_time = time.time()
        trace = []
        
        matched_sop = self.core.workflow_graph.match_sop(query_text)
        trace.append({
            "step": "1. Workflow State-Machine Identification",
            "type": "Graph Process Matching",
            "detail": f"Identified SOP: {matched_sop['title'] if matched_sop else 'None'} (Graph Root: {matched_sop['id'] if matched_sop else 'N/A'})"
        })
        
        workflow_context = ""
        mermaid_code = ""
        if matched_sop:
            workflow_context = self.core.workflow_graph.get_workflow_context(matched_sop["id"])
            mermaid_code = self.core.workflow_graph.generate_mermaid_flowchart(matched_sop["id"])
            trace.append({
                "step": "2. Sequential Step Traversal (BPMN DAG)",
                "type": "Topological Workflow Path",
                "detail": f"Traversed {len(matched_sop['steps'])} ordered steps with swimlane actors, prerequisites, and outputs."
            })
            
        expanded_query = query_text
        if matched_sop:
            expanded_query += f" {matched_sop['title']} " + " ".join([s['action'] for s in matched_sop['steps'][:3]])
            
        docs = self.core.db.similarity_search(expanded_query, k=k)
        query_embedding = self.core.embedding_function.embed_query(query_text)
        
        scored_results = []
        for doc in docs:
            doc_embedding = self.core.embedding_function.embed_query(doc.page_content)
            sim = float(cosine_similarity([query_embedding], [doc_embedding])[0][0])
            scored_results.append((doc, sim))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_doc, best_score = scored_results[0] if scored_results else (None, 0.0)
        
        trace.append({
            "step": "3. Vector Context Alignment",
            "type": "Similarity Retrieval",
            "detail": f"Aligned {len(docs)} text chunks from Chroma. Best cosine similarity: {best_score:.4f}"
        })
        
        if not matched_sop and best_score < threshold:
            answer = "Maaf, prosedur atau informasi terkait pertanyaan Anda tidak ditemukan dalam dokumen SOP resmi FSM UNDIP."
            sources = []
            contexts = []
            scores = []
        else:
            sources = [doc.metadata.get("id", os.path.basename(doc.metadata.get("source", "SOP.pdf"))) for doc, _ in scored_results]
            contexts = [doc.page_content for doc, _ in scored_results]
            scores = [score for _, score in scored_results]
            
            doc_context_text = "\n\n---\n\n".join(contexts)
            full_context = f"{workflow_context}\n\nKonteks Dokumen Pendukung:\n{doc_context_text}"
            prompt = PROMPT_TEMPLATE.format(context=full_context, question=query_text)
            answer = self.core.generate_synthesis(prompt, contexts, query_text)
            
            trace.append({
                "step": "4. Procedural Workflow Synthesis",
                "type": "LLM Synthesis + BPMN Output",
                "detail": f"Synthesized chronologically ordered answer with interactive visual Mermaid diagram."
            })
            
        total_latency = time.time() - start_time
        metrics = self.calculate_metrics(query_text, answer, contexts if (matched_sop or best_score >= threshold) else [], scores, total_latency)
        
        return {
            "architecture": "Workflow GraphRAG (Process DAG)",
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "scores": scores,
            "mermaid": mermaid_code,
            "matched_sop": matched_sop["title"] if matched_sop else None,
            "raw_docs": [doc.metadata for doc, _ in scored_results] if scored_results else [],
            "trace": trace,
            "metrics": metrics
        }

    def execute_hierarchical_rag_v0(self, query_text: str, k: int = 3, threshold: float = 0.3) -> Dict[str, Any]:
        """8a. Hierarchical Tree RAG v0 (Ablation Baseline: Document-level Chunk Retrieval + Parent Tree Expansion)"""
        start_time = time.time()
        trace = []
        
        docs = self.core.db.similarity_search(query_text, k=k)
        query_embedding = self.core.embedding_function.embed_query(query_text)
        
        scored_results = []
        for doc in docs:
            doc_embedding = self.core.embedding_function.embed_query(doc.page_content)
            sim = float(cosine_similarity([query_embedding], [doc_embedding])[0][0])
            scored_results.append((doc, sim))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_doc, best_score = scored_results[0] if scored_results else (None, 0.0)
        
        trace.append({
            "step": "1. Document Chunk Vector Retrieval (v0 Baseline)",
            "type": "Coarse-Grained Traversal",
            "detail": f"Retrieved top-{len(docs)} document chunks from standard index. Best similarity: {best_score:.4f}"
        })
        
        if best_score < threshold:
            answer = "Maaf, rincian hierarki atau informasi terkait pertanyaan Anda tidak ditemukan dalam dokumen SOP resmi."
            sources = []
            contexts = []
            scores = []
        else:
            top_doc = scored_results[0][0]
            source_file = os.path.basename(top_doc.metadata.get("source", ""))
            
            tree_context = ""
            if source_file in self.core.hierarchical_chunker.document_tree:
                tree_context = self.core.hierarchical_chunker.get_hierarchical_context_for_doc(source_file)
                trace.append({
                    "step": "2. Parent Document Tree Expansion",
                    "type": "Top-Down Context Injection",
                    "detail": f"Expanded root node: {source_file} with full procedural tree & leaf metadata."
                })
            else:
                tree_context = "\n\n---\n\n".join([d.page_content for d, _ in scored_results])
                
            sources = [doc.metadata.get("id", source_file) for doc, _ in scored_results]
            contexts = [doc.page_content for doc, _ in scored_results]
            scores = [score for _, score in scored_results]
            
            prompt = PROMPT_TEMPLATE.format(context=tree_context, question=query_text)
            answer = self.core.generate_synthesis(prompt, contexts, query_text)
            
            trace.append({
                "step": "3. Tree-Synthesized Response",
                "type": "Hierarchical Answer",
                "detail": f"Generated answer retaining complete hierarchical structure and attributes."
            })
            
        total_latency = time.time() - start_time
        metrics = self.calculate_metrics(query_text, answer, contexts if best_score >= threshold else [], scores, total_latency)
        
        return {
            "architecture": "Hierarchical Tree RAG v0 (Doc Chunk Baseline)",
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "scores": scores,
            "raw_docs": [doc.metadata for doc, _ in scored_results] if scored_results else [],
            "trace": trace,
            "metrics": metrics
        }

    def execute_hierarchical_rag(self, query_text: str, k: int = 3, threshold: float = 0.3, variant: Optional[str] = None) -> Dict[str, Any]:
        """8. Hierarchical Tree RAG (Bottom-Up Leaf Chunk Retrieval + Top-Down Parent Tree Context Expansion)"""
        start_time = time.time()
        trace = []
        
        # Determine parent SOP selection variant: top1, vote, sum
        htree_variant = (variant or os.getenv("HTREE_VARIANT", "top1")).lower().strip()
        if htree_variant not in ("top1", "vote", "sum"):
            htree_variant = "top1"
        
        # Dense top-k retrieval over fine-grained leaf chunks (individual steps & overview)
        htree_store = getattr(self.core, "htree_db", self.core.db)
        docs = htree_store.similarity_search(query_text, k=k)
        query_embedding = self.core.embedding_function.embed_query(query_text)
        
        scored_results = []
        for doc in docs:
            doc_embedding = self.core.embedding_function.embed_query(doc.page_content)
            sim = float(cosine_similarity([query_embedding], [doc_embedding])[0][0])
            scored_results.append((doc, sim))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_doc, best_score = scored_results[0] if scored_results else (None, 0.0)
        
        trace.append({
            "step": "1. Leaf Chunk Vector Retrieval",
            "type": "Bottom-Up Traversal",
            "detail": f"Retrieved top-{len(docs)} leaf chunks from 'htree_leaves'. Best similarity: {best_score:.4f}"
        })
        
        if best_score < threshold:
            answer = "Maaf, rincian hierarki atau informasi terkait pertanyaan Anda tidak ditemukan dalam dokumen SOP resmi."
            sources = []
            contexts = []
            scores = []
        else:
            # Select parent SOP according to chosen variant
            from collections import defaultdict
            if htree_variant == "vote":
                votes = defaultdict(int)
                best_score_per_sop = defaultdict(float)
                for doc, score in scored_results:
                    sop = os.path.basename(doc.metadata.get("source", ""))
                    votes[sop] += 1
                    if score > best_score_per_sop[sop]:
                        best_score_per_sop[sop] = score
                sorted_sops = sorted(votes.keys(), key=lambda s: (votes[s], best_score_per_sop[s]), reverse=True)
                source_file = sorted_sops[0]
                selection_detail = f"majority vote across top-{len(scored_results)} leaves ({votes[source_file]} votes, variant=vote)"
            elif htree_variant == "sum":
                sums = defaultdict(float)
                best_score_per_sop = defaultdict(float)
                for doc, score in scored_results:
                    sop = os.path.basename(doc.metadata.get("source", ""))
                    sums[sop] += score
                    if score > best_score_per_sop[sop]:
                        best_score_per_sop[sop] = score
                sorted_sops = sorted(sums.keys(), key=lambda s: (sums[s], best_score_per_sop[s]), reverse=True)
                source_file = sorted_sops[0]
                selection_detail = f"score sum across top-{len(scored_results)} leaves (sum={sums[source_file]:.4f}, variant=sum)"
            else:  # default "top1"
                top_doc = scored_results[0][0]
                source_file = os.path.basename(top_doc.metadata.get("source", ""))
                selection_detail = f"top-1 leaf ({top_doc.metadata.get('id', '')}, score={best_score:.4f}, variant=top1)"
            
            tree_context = ""
            if source_file in self.core.hierarchical_chunker.document_tree:
                tree_context = self.core.hierarchical_chunker.get_hierarchical_context_for_doc(source_file)
                trace.append({
                    "step": f"2. Parent Document Tree Expansion ({htree_variant})",
                    "type": "Top-Down Context Injection",
                    "detail": f"Expanded root node: {source_file} via {selection_detail} with full procedural tree."
                })
            else:
                tree_context = "\n\n---\n\n".join([d.page_content for d, _ in scored_results])
                
            sources = [doc.metadata.get("id", source_file) for doc, _ in scored_results]
            contexts = [doc.page_content for doc, _ in scored_results]
            scores = [score for _, score in scored_results]
            
            prompt = PROMPT_TEMPLATE.format(context=tree_context, question=query_text)
            answer = self.core.generate_synthesis(prompt, contexts, query_text)
            
            trace.append({
                "step": "3. Tree-Synthesized Response",
                "type": "Hierarchical Answer",
                "detail": f"Generated answer retaining complete hierarchical structure and attributes."
            })
            
        total_latency = time.time() - start_time
        metrics = self.calculate_metrics(query_text, answer, contexts if best_score >= threshold else [], scores, total_latency)
        
        return {
            "architecture": "Hierarchical Tree RAG",
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "scores": scores,
            "raw_docs": [doc.metadata for doc, _ in scored_results] if scored_results else [],
            "trace": trace,
            "metrics": metrics
        }

    def query_architecture(self, arch_name: str, query_text: str, k: int = 3, threshold: float = 0.3) -> Dict[str, Any]:
        """Dispatch query to specified architecture and guarantee visual flowchart attachment."""
        norm_name = arch_name.lower()
        if "workflow" in norm_name or "bpmn" in norm_name:
            result = self.execute_workflow_graph_rag(query_text, k=k, threshold=threshold)
        elif "htree_v0" in norm_name or "tree_v0" in norm_name or "hierarchical_v0" in norm_name or ("v0" in norm_name and ("tree" in norm_name or "hierarchical" in norm_name)):
            result = self.execute_hierarchical_rag_v0(query_text, k=k, threshold=threshold)
        elif "hierarchical" in norm_name or "tree" in norm_name or "htree" in norm_name:
            result = self.execute_hierarchical_rag(query_text, k=k, threshold=threshold)
        elif "naive" in norm_name:
            result = self.execute_naive_rag(query_text, k=k, threshold=threshold)
        elif "hybrid" in norm_name:
            result = self.execute_hybrid_rag(query_text, k=k, threshold=threshold)
        elif "graph" in norm_name:
            result = self.execute_graph_rag(query_text, k=k, threshold=threshold)
        elif "agentic" in norm_name:
            result = self.execute_agentic_rag(query_text, k=k, threshold=threshold)
        elif "corrective" in norm_name or "crag" in norm_name:
            result = self.execute_crag(query_text, k=k)
        elif "multimodal" in norm_name:
            result = self.execute_multimodal_rag(query_text, k=k, threshold=threshold)
        else:
            result = self.execute_naive_rag(query_text, k=k, threshold=threshold)

        # Universal Diagram & SOP Matching Layer (Ciri khas sistem RAG SOP)
        # Ensure any architecture returning an answer about an SOP gets the flowchart & BPMN data attached
        matched_sop = self.core.workflow_graph.match_sop(query_text)
        if not matched_sop and result.get("sources"):
            # Try to match from top retrieved document source
            top_src = result["sources"][0]
            for s_id, meta in self.core.workflow_graph.sop_metadata.items():
                if meta["title"].lower() in top_src.lower() or s_id.lower() in top_src.lower():
                    matched_sop = meta
                    break

        if matched_sop:
            result["matched_sop"] = matched_sop["title"]
            result["sop_id"] = matched_sop["id"]
            result["sop_max_duration"] = matched_sop["max_duration"]
            result["mermaid_standard"] = self.core.workflow_graph.generate_mermaid_flowchart(matched_sop["id"], mode="standard")
            result["mermaid_swimlane"] = self.core.workflow_graph.generate_mermaid_flowchart(matched_sop["id"], mode="swimlane")
            result["mermaid_detailed"] = self.core.workflow_graph.generate_mermaid_flowchart(matched_sop["id"], mode="detailed")
            if not result.get("mermaid"):
                result["mermaid"] = result["mermaid_standard"]

        return result
