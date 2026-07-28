import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

CHROMA_PATH = "chroma"
EMBEDDING_MODEL_PATH = "./indo_finetuned_embedding"

PROMPT_TEMPLATE = """
Anda adalah asisten layanan mahasiswa Fakultas Sains dan Matematika Universitas Diponegoro 
yang membantu menjawab pertanyaan berdasarkan dokumen resmi kampus.
Jawablah dengan bahasa Indonesia yang jelas, singkat, dan sopan. 
Jika jawaban tidak ada dalam konteks, katakan dengan jujur 
bahwa informasi tersebut tidak tersedia. JANGAN gunakan bahasa inggris dalam memberi respon.

Konteks:
{context}

---

Pertanyaan: {question}

Jawaban sebagai asisten layanan mahasiswa:
"""

REWRITE_PROMPT = """
Anda adalah sistem penulisan ulang kueri (query rewriter).
Tugas Anda adalah memformulasikan ulang kueri pengguna berikut agar lebih jelas, berorientasi kata kunci, dan lebih mudah dicari dalam basis data dokumen SOP universitas.
Hanya kembalikan kueri yang sudah ditulis ulang, tanpa kalimat pengantar atau penjelasan tambahan.

Kueri asli: {query}
Kueri baru:
"""

class CorrectiveRAG:
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)
        self.model = ChatOllama(model="llama3.1:8b")
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.rewrite_prompt_template = ChatPromptTemplate.from_template(REWRITE_PROMPT)

    def _retrieve_and_score(self, query_text: str):
        query_embedding = self.embedding_function.embed_query(query_text)
        docs = self.db.similarity_search(query_text, k=3)
        results = []
        for doc in docs:
            doc_embedding = self.embedding_function.embed_query(doc.page_content)
            cosine_sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            results.append((doc, cosine_sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def query(self, query_text: str) -> dict:
        # Step 1: Initial Retrieval
        results = self._retrieve_and_score(query_text)
        best_doc, best_score = results[0]

        # Grader thresholds
        upper_threshold = 0.55
        lower_threshold = 0.35

        # Decision Node
        if best_score >= upper_threshold:
            # CORRECT: Go directly to generation
            grade = "CORRECT"
        elif best_score >= lower_threshold:
            # AMBIGUOUS: Rewrite query and search again
            grade = "AMBIGUOUS"
            rewrite_prompt = self.rewrite_prompt_template.format(query=query_text)
            rewritten_query = self.model.invoke(rewrite_prompt).content.strip()
            
            # Retrieve again with rewritten query
            new_results = self._retrieve_and_score(rewritten_query)
            new_best_doc, new_best_score = new_results[0]
            
            if new_best_score >= lower_threshold:
                results = new_results
            # If still low, it falls back to incorrect
        else:
            # INCORRECT: Trigger fallback
            grade = "INCORRECT"
            
        if grade == "INCORRECT" or results[0][1] < lower_threshold:
            fallback_answer = "Maaf, informasi tentang '" + query_text + "' tidak ditemukan dalam dokumen SOP resmi Fakultas Sains dan Matematika Universitas Diponegoro."
            return {
                "answer": fallback_answer,
                "sources": [],
                "contexts": [],
                "grade": "INCORRECT"
            }

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        prompt = self.prompt_template.format(context=context_text, question=query_text)
        
        response = self.model.invoke(prompt)
        response_text = response.content

        sources = [doc.metadata.get("id", "Unknown") for doc, _ in results]
        contexts = [doc.page_content for doc, _ in results]

        return {
            "answer": response_text,
            "sources": sources,
            "contexts": contexts,
            "grade": grade
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    rag = CorrectiveRAG()
    res = rag.query("Bagaimana cara melakukan pengajuan cuti akademik?")
    print("Answer:", res["answer"])
    print("Grade:", res.get("grade"))
