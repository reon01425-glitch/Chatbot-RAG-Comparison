import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

CHROMA_PATH = "chroma"
DATA_PATH = "data"
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

class HybridRAG:
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)
        self.model = ChatOllama(model="llama3.1:8b")
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        # Initialize BM25 with document chunks
        loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1700,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        self.chunks = text_splitter.split_documents(documents)
        # Assign IDs to chunks if they don't have them
        for i, chunk in enumerate(self.chunks):
            source = os.path.basename(chunk.metadata.get("source", "unknown"))
            page = chunk.metadata.get("page", 0)
            chunk.metadata["id"] = f"{source}:{page}:{i}"
            
        tokenized_corpus = [doc.page_content.lower().split(" ") for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def query(self, query_text: str) -> dict:
        # 1. Retrieve Dense (Vector Store) - Top 5
        dense_results = self.db.similarity_search(query_text, k=5)
        
        # 2. Retrieve Sparse (BM25) - Top 5
        tokenized_query = query_text.lower().split(" ")
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_indices = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)[:5]
        sparse_results = [self.chunks[i] for i in sparse_indices]

        # 3. Reciprocal Rank Fusion (RRF)
        # Combine results based on rank
        rrf_scores = {}
        k = 60  # RRF constant
        
        for rank, doc in enumerate(dense_results, 1):
            doc_id = doc.metadata.get("id") or doc.page_content[:50]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
            
        for rank, doc in enumerate(sparse_results, 1):
            doc_id = doc.metadata.get("id") or doc.page_content[:50]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)

        # Get final union of documents
        all_docs = {doc.metadata.get("id") or doc.page_content[:50]: doc for doc in dense_results + sparse_results}
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:3]
        results_docs = [all_docs[doc_id] for doc_id in sorted_doc_ids]

        # 4. Score results with embeddings for threshold checking
        query_embedding = self.embedding_function.embed_query(query_text)
        scored_results = []
        for doc in results_docs:
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

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in scored_results])
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
    rag = HybridRAG()
    res = rag.query("Bagaimana cara melakukan pengajuan cuti akademik?")
    print("Answer:", res["answer"])
    print("Sources:", res["sources"])
