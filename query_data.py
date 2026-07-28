import argparse
import os
import time
import csv
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(override=True)

CHROMA_PATH = "chroma"
EMBEDDING_MODEL_PATH = "./indo_finetuned_embedding"
CSV_PATH = "results.csv"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
    query_embedding = embedding_function.embed_query(query_text)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    docs = db.similarity_search(query_text, k=3)
    
    results = []
    for doc in docs:
        doc_embedding = embedding_function.embed_query(doc.page_content)
        cosine_sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        results.append((doc, cosine_sim))
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    best_doc, best_score = results[0]
    threshold = 0.3

    if best_score < threshold:
        answer = "Maaf, saya tidak menemukan jawaban pada dokumen yang tersedia."
        coverage = 0
        sources = []
        response_time = 0.0
        print("\nJawaban:\n", answer)
        save_to_csv(query_text, answer, sources, response_time, coverage)
        return answer

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ["GITHUB_TOKEN"],
        base_url="https://models.inference.ai.azure.com"
    )

    start_time = time.time()
    response = model.invoke(prompt)
    end_time = time.time()
    response_time = end_time - start_time
    response_text = response.content  
    coverage = 1

    sources = [doc.metadata.get("id", None) for doc, _ in results]

    print(response_text)
    print(f"\nAnswered in : {response_time:.2f} sec")

    save_to_csv(query_text, response_text, sources, response_time, coverage)

    return response_text


def save_to_csv(question: str, answer: str, sources: list, response_time: float, coverage: int):
    """Simpan hasil query ke file CSV"""
    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Pertanyaan", "Jawaban", "Sumber", "Response Time (sec)", "Coverage"])

        writer.writerow([question, answer, "; ".join(sources), f"{response_time:.2f}", coverage])


if __name__ == "__main__":
    main()


# import argparse
# import os
# import time
# import json
# import csv
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity

# load_dotenv()

# CHROMA_PATH = "chroma"
# EMBEDDING_MODEL_PATH = "./indo_finetuned_embedding"
# CSV_PATH = "results.csv"
# HISTORY_PATH = "history.json"  # persisted conversation history
# MAX_PAIRS = 3  # 3 user + 3 assistant = 6 turns total

# PROMPT_TEMPLATE = """
# Anda adalah asisten layanan mahasiswa Fakultas Sains dan Matematika Universitas Diponegoro 
# yang membantu menjawab pertanyaan berdasarkan dokumen resmi kampus.
# Jawablah dengan bahasa Indonesia yang jelas, singkat, dan sopan. 
# Jika jawaban tidak ada dalam konteks, katakan dengan jujur 
# bahwa informasi tersebut tidak tersedia. JANGAN gunakan bahasa inggris dalam memberi respon.

# Percakapan terbaru:
# {history}

# Konteks:
# {context}

# ---

# Pertanyaan: {question}

# Jawaban sebagai asisten layanan mahasiswa:
# """


# def load_history() -> list:
#     """Load conversation history from HISTORY_PATH, or return [] if missing."""
#     if not os.path.isfile(HISTORY_PATH):
#         return []
#     try:
#         with open(HISTORY_PATH, "r", encoding="utf-8") as f:
#             data = json.load(f)
#             if isinstance(data, list):
#                 return data
#     except Exception:
#         pass
#     return []


# def save_history(history: list):
#     """Persist conversation history to HISTORY_PATH (keeps last MAX_PAIRS*2 turns)."""
#     try:
#         # Ensure we only keep last MAX_PAIRS user+assistant pairs
#         max_turns = MAX_PAIRS * 2
#         trimmed = history[-max_turns:]
#         with open(HISTORY_PATH, "w", encoding="utf-8") as f:
#             json.dump(trimmed, f, ensure_ascii=False, indent=2)
#     except Exception as e:
#         print("Gagal menyimpan history:", e)


# def push_history(history: list, role: str, text: str):
#     """Append a turn and persist trimmed history."""
#     history.append({"role": role, "text": text})
#     # Trim if necessary
#     max_turns = MAX_PAIRS * 2
#     if len(history) > max_turns:
#         history[:] = history[-max_turns:]
#     save_history(history)


# def format_history_for_prompt(history: list) -> str:
#     """Format history into a readable string for the prompt (last MAX_PAIRS pairs)."""
#     # Keep last MAX_PAIRS * 2 entries
#     relevant = history[-(MAX_PAIRS * 2) :]
#     lines = []
#     for turn in relevant:
#         label = "Pengguna" if turn.get("role") == "user" else "Asisten"
#         lines.append(f"{label}: {turn.get('text', '')}")
#     return "\n".join(lines)


# def rewrite_followup(model: ChatOpenAI, history: list, new_query: str) -> str:
#     """
#     Ask the LLM to rewrite a follow-up into a standalone question (Indonesian).
#     If the rewrite fails, return original new_query.
#     """
#     hist_snippet = format_history_for_prompt(history)
#     rewrite_prompt = f"""
# Anda adalah asisten yang menulis ulang pertanyaan follow-up menjadi pertanyaan lengkap dan berdiri sendiri (bahasa Indonesia).
# Gunakan konteks percakapan berikut untuk menyelesaikan ambiguitas.

# Konteks percakapan (jika ada):
# {hist_snippet}

# Pertanyaan pengguna (mungkin follow-up): "{new_query}"

# Tulis ulang menjadi sebuah pertanyaan tunggal yang berdiri sendiri (jangan jawab), keluarkan hanya pertanyaan yang lengkap.
# """
#     try:
#         resp = model.invoke(rewrite_prompt)
#         rewritten = resp.content.strip().strip('"').strip()
#         if rewritten:
#             return rewritten
#     except Exception as e:
#         print("Rewrite gagal:", e)
#     return new_query


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     parser.add_argument("--new", action="store_true", help="Start a new chat (delete history.json)")
#     parser.add_argument("--no-rewrite", action="store_true", help="Disable question rewriting")
#     parser.add_argument("--k", type=int, default=3, help="Number of docs to retrieve (k)")
#     args = parser.parse_args()

#     if args.new and os.path.isfile(HISTORY_PATH):
#         try:
#             os.remove(HISTORY_PATH)
#             print("History dihapus. Memulai obrolan baru.")
#         except Exception as e:
#             print("Gagal menghapus history:", e)

#     query_text = args.query_text
#     query_rag(query_text, use_rewrite=not args.no_rewrite, k=args.k)


# def query_rag(query_text: str, use_rewrite: bool = True, k: int = 3):
#     # Load embedding model and DB
#     embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Load persisted conversation history
#     conversation_history = load_history()

#     # Instantiate LLM (used for rewriting and final answer)
#     model = ChatOpenAI(
#         model="gpt-5",
#         api_key=os.environ.get("GITHUB_TOKEN"),
#         base_url="https://models.inference.ai.azure.com"
#     )

#     # Optionally rewrite the follow-up to a standalone query (helps retrieval)
#     query_for_embedding = query_text
#     if use_rewrite and len(conversation_history) > 0:
#         rewritten = rewrite_followup(model, conversation_history, query_text)
#         if rewritten and rewritten != query_text:
#             print(f"[Rewrite] Pertanyaan diubah menjadi: {rewritten}")
#             query_for_embedding = rewritten
#         else:
#             print("[Rewrite] Tidak ada perubahan pada pertanyaan.")
#     else:
#         if use_rewrite:
#             print("[Rewrite] Tidak ada history, melewati rewriting.")
#         else:
#             print("[Rewrite] Rewriting dinonaktifkan.")

#     # Compute embedding for query_for_embedding
#     query_embedding = embedding_function.embed_query(query_for_embedding)

#     # Retrieve top-k documents using text used for embedding
#     docs = db.similarity_search(query_for_embedding, k=k)

#     # Score results with cosine similarity
#     results = []
#     for doc in docs:
#         doc_embedding = embedding_function.embed_query(doc.page_content)
#         cosine_sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
#         results.append((doc, cosine_sim))

#     results.sort(key=lambda x: x[1], reverse=True)
#     best_doc, best_score = results[0]
#     threshold = 0.3

#     if best_score < threshold:
#         answer = "Maaf, saya tidak menemukan jawaban pada dokumen yang tersedia."
#         coverage = 0
#         sources = []
#         response_time = 0.0

#         # persist to history
#         push_history(conversation_history, "user", query_text)
#         push_history(conversation_history, "assistant", answer)

#         print("\nJawaban:\n", answer)
#         save_to_csv(query_text, answer, sources, response_time, coverage)
#         return answer

#     # Build context from retrieved docs
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
#     sources = [doc.metadata.get("id", None) for doc, _ in results]

#     # Format history for prompt (last MAX_PAIRS pairs)
#     hist_to_attach = format_history_for_prompt(conversation_history)

#     # Build prompt including history and context
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(history=hist_to_attach, context=context_text, question=query_text)

#     # Call the model
#     start_time = time.time()
#     response = model.invoke(prompt)
#     end_time = time.time()
#     response_time = end_time - start_time
#     response_text = response.content
#     coverage = 1

#     # Persist history (user query then assistant response)
#     push_history(conversation_history, "user", query_text)
#     push_history(conversation_history, "assistant", response_text)

#     # Print, save, and return
#     print(response_text)
#     print(f"\nAnswered in : {response_time:.2f} sec")

#     save_to_csv(query_text, response_text, sources, response_time, coverage)
#     return response_text


# def save_to_csv(question: str, answer: str, sources: list, response_time: float, coverage: int):
#     """Simpan hasil query ke file CSV"""
#     file_exists = os.path.isfile(CSV_PATH)

#     with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)

#         if not file_exists:
#             writer.writerow(["Pertanyaan", "Jawaban", "Sumber", "Response Time (sec)", "Coverage"])

#         writer.writerow([question, answer, "; ".join([s for s in sources if s]), f"{response_time:.2f}", coverage])


# if __name__ == "__main__":
#     main()