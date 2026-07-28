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

class NaiveRAG:
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)
        self.model = ChatOllama(model="llama3.1:8b")
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def query(self, query_text: str) -> dict:
        query_embedding = self.embedding_function.embed_query(query_text)
        docs = self.db.similarity_search(query_text, k=3)
        
        results = []
        for doc in docs:
            doc_embedding = self.embedding_function.embed_query(doc.page_content)
            cosine_sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            results.append((doc, cosine_sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        best_doc, best_score = results[0]
        threshold = 0.3

        if best_score < threshold:
            return {
                "answer": "Maaf, saya tidak menemukan jawaban pada dokumen yang tersedia.",
                "sources": [],
                "contexts": []
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
            "contexts": contexts
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    rag = NaiveRAG()
    res = rag.query("Bagaimana cara melakukan pengajuan cuti akademik?")
    print("Answer:", res["answer"])
    print("Sources:", res["sources"])
