import os
import networkx as nx
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

CHROMA_PATH = "chroma"
EMBEDDING_MODEL_PATH = "./indo_finetuned_embedding"

PROMPT_TEMPLATE = """
Anda adalah asisten layanan mahasiswa Fakultas Sains dan Matematika Universitas Diponegoro 
yang membantu menjawab pertanyaan berdasarkan dokumen resmi kampus dengan basis pengetahuan grafik relasi (GraphRAG).
Jawablah dengan bahasa Indonesia yang jelas, singkat, dan sopan. 
Jika jawaban tidak ada dalam konteks, katakan dengan jujur 
bahwa informasi tersebut tidak tersedia. JANGAN gunakan bahasa inggris dalam memberi respon.

Konteks Graf Relasi & Dokumen:
{context}

---

Pertanyaan: {question}

Jawaban sebagai asisten layanan mahasiswa:
"""

class GraphRAG:
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)
        self.model = ChatOllama(model="llama3.1:8b")
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        # Build local relationship Graph using NetworkX
        self.graph = nx.Graph()
        
        # Add entity relations extracted from university SOP
        relations = [
            ("cuti akademik", "izin cuti", "terkait"),
            ("cuti akademik", "aktif kembali", "dilanjutkan dengan"),
            ("cuti akademik", "ketua program studi", "meminta persetujuan"),
            ("cuti akademik", "dekan", "mengeluarkan surat izin"),
            ("legalisir", "ijazah", "tujuan berkas"),
            ("legalisir", "transkrip", "tujuan berkas"),
            ("beasiswa", "rekomendasi beasiswa", "syarat pengajuan"),
            ("beasiswa", "spp/ukt", "bukti pembayaran"),
            ("ukt", "keterlambatan pembayaran", "permohonan izin"),
            ("irs", "pengisian irs", "proses registrasi akademik"),
            ("irs", "dosen wali", "menyetujui pengisian")
        ]
        for u, v, r in relations:
            self.graph.add_edge(u, v, relationship=r)

    def _find_related_entities(self, query_text: str) -> list:
        # Simple entity matching against query string
        matched = []
        q_lower = query_text.lower()
        for node in self.graph.nodes:
            if node in q_lower:
                matched.append(node)
                
        # Find neighbors (related entities)
        related = set(matched)
        for node in matched:
            related.update(self.graph.neighbors(node))
            
        return list(related)

    def query(self, query_text: str) -> dict:
        # Step 1: Query expansion using graph relations
        related_entities = self._find_related_entities(query_text)
        
        # Build query based on expanded entities
        expanded_query = query_text
        if related_entities:
            expanded_query += " " + " ".join(related_entities)

        # Step 2: Retrieve based on expanded query
        docs = self.db.similarity_search(expanded_query, k=3)
        
        query_embedding = self.embedding_function.embed_query(query_text)
        scored_results = []
        for doc in docs:
            doc_embedding = self.embedding_function.embed_query(doc.page_content)
            cosine_sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            scored_results.append((doc, cosine_sim))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_doc, best_score = scored_results[0]
        threshold = 0.3

        if best_score < threshold:
            return {
                "answer": "Maaf, saya tidak menemukan jawaban pada dokumen yang tersedia.",
                "sources": [],
                "contexts": []
            }

        # Step 3: Format graph context alongside document context
        graph_context = ""
        if related_entities:
            graph_context = "Relasi Entitas Kampus Terkait:\n"
            for entity in related_entities:
                for neighbor in self.graph.neighbors(entity):
                    rel_type = self.graph[entity][neighbor]['relationship']
                    graph_context += f"- {entity} ({rel_type} -> {neighbor})\n"
            graph_context += "\n"

        context_text = graph_context + "\n\n---\n\n".join([doc.page_content for doc, _ in scored_results])
        prompt = self.prompt_template.format(context=context_text, question=query_text)
        
        response = self.model.invoke(prompt)
        response_text = response.content

        sources = [doc.metadata.get("id", "Unknown") for doc, _ in scored_results]
        contexts = [doc.page_content for doc, _ in scored_results]

        return {
            "answer": response_text,
            "sources": sources,
            "contexts": contexts,
            "related_entities": related_entities
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    rag = GraphRAG()
    res = rag.query("Bagaimana cara melakukan pengajuan cuti akademik?")
    print("Answer:", res["answer"])
    print("Related Entities:", res.get("related_entities"))
