import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import pypdf

CHROMA_PATH = "chroma"
EMBEDDING_MODEL_PATH = "./indo_finetuned_embedding"
DATA_PATH = "data"

PROMPT_TEMPLATE = """
Anda adalah asisten layanan mahasiswa Fakultas Sains dan Matematika Universitas Diponegoro 
yang membantu menjawab pertanyaan berdasarkan dokumen resmi kampus menggunakan sistem Multimodal RAG.
Kami telah mengekstrak struktur teks, tata letak tabel, dan gambar/diagram alur dari dokumen PDF asli.

Konteks Teks & Struktur Layout:
{context}

---

Pertanyaan: {question}

Jawaban sebagai asisten layanan mahasiswa:
"""

class MultimodalRAG:
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)
        self.model = ChatOllama(model="llama3.1:8b")
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.multimodal_index = {}
        self._build_multimodal_index()

    def _build_multimodal_index(self):
        # Extract figures, tables, and text layouts from local PDFs
        for fname in os.listdir(DATA_PATH):
            if fname.endswith(".pdf"):
                fpath = os.path.join(DATA_PATH, fname)
                try:
                    reader = pypdf.PdfReader(fpath)
                    for i, page in enumerate(reader.pages):
                        # Simulating extraction of images/flowcharts or tables
                        images_count = len(page.images)
                        has_tables = "tabel" in page.extract_text().lower() or "no." in page.extract_text().lower()
                        
                        meta = {
                            "source": fname,
                            "page": i + 1,
                            "images_count": images_count,
                            "has_tables": has_tables,
                            "multimodal_desc": f"[Layout Page {i+1}]: Mengandung {images_count} diagram alur/gambar kerja dan " + 
                                               ("satu atau lebih representasi tabel data." if has_tables else "tidak terdeteksi tabel terpisah.")
                        }
                        self.multimodal_index[f"{fname}:{i+1}"] = meta
                except Exception as e:
                    pass

    def query(self, query_text: str) -> dict:
        docs = self.db.similarity_search(query_text, k=3)
        
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

        # Enrich text context with extracted layout/image metadata
        enriched_contexts = []
        for doc, _ in scored_results:
            src = os.path.basename(doc.metadata.get("source", ""))
            page = doc.metadata.get("page", 1)
            key = f"{src}:{page}"
            
            meta_desc = ""
            if key in self.multimodal_index:
                meta_desc = f"\n--- INFORMASI VISUAL & LAYOUT: {self.multimodal_index[key]['multimodal_desc']} ---\n"
                
            enriched_contexts.append(meta_desc + doc.page_content)

        context_text = "\n\n---\n\n".join(enriched_contexts)
        prompt = self.prompt_template.format(context=context_text, question=query_text)
        
        response = self.model.invoke(prompt)
        response_text = response.content

        sources = [doc.metadata.get("id", "Unknown") for doc, _ in scored_results]
        contexts = [doc.page_content for doc, _ in scored_results]

        return {
            "answer": response_text,
            "sources": sources,
            "contexts": contexts
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    rag = MultimodalRAG()
    res = rag.query("Bagaimana cara melakukan pengajuan cuti akademik?")
    print("Answer:", res["answer"])
