import argparse
import os
import shutil
import json
import subprocess
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "chroma"
DATA_PATH = "data"
DATASET_PATH = "datasets"
TRAINED_JSON = os.path.join(DATASET_PATH, "trained.json")
EMBEDDING_MODEL_PATH = "./indo_finetuned_embedding"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1700,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"}
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[]) 
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    docs_grouped = {}
    for chunk in chunks_with_ids:
        src = os.path.basename(chunk.metadata.get("source", ""))
        docs_grouped.setdefault(src, []).append(chunk)

    if os.path.exists(TRAINED_JSON):
        with open(TRAINED_JSON, "r", encoding="utf-8") as f:
            trained_files = set(json.load(f))
    else:
        trained_files = set()

    for doc_name, doc_chunks in docs_grouped.items():
        print(f"\nProcessing {doc_name} ...")
        new_chunks = [c for c in doc_chunks if c.metadata["id"] not in existing_ids]

        if not new_chunks:
            print(f"{doc_name} already in DB, skipping...")
            continue

        train_file = f"train_{os.path.splitext(doc_name)[0]}.json"

        if train_file not in trained_files:
            print(f"{doc_name} not yet trained -> Generating QA & Fine-tuning...")
            subprocess.run(["python", "generate_qa.py", "--doc", doc_name], check=True)
            subprocess.run(["python", "finetune_embeddings.py"], check=True)
            embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
            db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embedding_function
            )
        else:
            print(f"{doc_name} already trained, skipping training...")

        print(f"Adding {len(new_chunks)} chunks from {doc_name} to DB")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()