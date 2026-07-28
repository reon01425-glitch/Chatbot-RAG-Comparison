import os, json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

DATA_PATH = "data"
DATASET_PATH = "datasets"

llm = OllamaLLM(model="llama2")

def split_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1700,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)

    print(f"\nFile: {os.path.basename(file_path)} -> {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i}/{len(chunks)} ---")
        print(chunk.page_content)

    return chunks

def generate_qa_for_doc(doc_path, out_file):
    chunks = split_document(doc_path)
    print(f"{os.path.basename(doc_path)} -> {len(chunks)} chunks")

    qa_pairs = []
    for i, chunk in enumerate(chunks):
        prompt = f"""
        Buatlah SATU (1) pasang pertanyaan dan jawaban dalam bahasa Indonesia
        berdasarkan teks berikut.

        Hanya kembalikan hasil dengan format persis seperti ini:

        Q: <pertanyaan>
        A: <jawaban>

        Jangan tambahkan penjelasan lain, kalimat pengantar, atau teks tambahan.

        Teks:
        \"\"\"{chunk.page_content}\"\"\"        
        """

        response_id = llm.invoke(prompt).strip()

        qa_pairs.append({
            "id": f"{chunk.metadata.get('source', os.path.basename(doc_path))}:{chunk.metadata.get('page', 0)}:{i}",
            "qa": response_id,
            "source": chunk.metadata.get("source", os.path.basename(doc_path))
        })

        print(f"Processed {i+1}/{len(chunks)} chunks")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"QA pairs saved: {out_file}")

def main():
    os.makedirs(DATASET_PATH, exist_ok=True)

    for fname in os.listdir(DATA_PATH):
        if fname.endswith(".pdf"):
            doc_path = os.path.join(DATA_PATH, fname)
            dataset_file = os.path.join(DATASET_PATH, f"train_{os.path.splitext(fname)[0]}.json")

            if not os.path.exists(dataset_file):
                print(f"Dokumen baru ditemukan: {fname}")
                generate_qa_for_doc(doc_path, dataset_file)
            else:
                print(f"Dataset sudah ada untuk {fname}, skip.")

if __name__ == "__main__":
    main()