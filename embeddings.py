import argparse
import os
import shutil
import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
DATA_PATH = os.getenv("DATA_PATH", "data")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./indo_finetuned_embedding")


def main():
    parser = argparse.ArgumentParser(description="Build Chroma vector store for SOP documents.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--chroma-path", default=os.getenv("CHROMA_PATH", "chroma"), help="Persist directory for Chroma DB")
    parser.add_argument("--model-path", default=os.getenv("EMBEDDING_MODEL_PATH", "./indo_finetuned_embedding"), help="Path or HuggingFace ID of embedding model")
    parser.add_argument("--data-path", default=os.getenv("DATA_PATH", "data"), help="Path to SOP PDFs")
    args = parser.parse_args()

    if args.reset:
        print(f"Clearing Database at {args.chroma_path}")
        clear_database(args.chroma_path)

    documents = load_documents(args.data_path)
    chunks = split_documents(documents)
    add_to_chroma(chunks, chroma_path=args.chroma_path, model_path=args.model_path)
    add_htree_to_chroma(chroma_path=args.chroma_path, model_path=args.model_path, data_path=args.data_path)

    from src.manifest import create_index_manifest, save_index_manifest
    manifest = create_index_manifest(
        chroma_path=args.chroma_path,
        model_id_or_path=args.model_path,
        data_path=args.data_path,
    )
    save_index_manifest(args.chroma_path, manifest)
    print(f"Index manifest saved to {args.chroma_path}/index_manifest.json")


def add_htree_to_chroma(chroma_path: str = CHROMA_PATH, model_path: str = EMBEDDING_MODEL_PATH, data_path: str = DATA_PATH):
    from src.graph_sop.hierarchical_chunker import SOPHierarchicalChunker
    embedding_function = HuggingFaceEmbeddings(model_name=model_path)
    htree_chunker = SOPHierarchicalChunker(data_path=data_path)

    db = Chroma(
        collection_name="htree_leaves",
        persist_directory=chroma_path,
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"}
    )
    existing_items = db.get(include=[])
    existing_ids = set(existing_items.get("ids", []))
    print(f"Number of existing leaf chunks in DB ({chroma_path} / htree_leaves): {len(existing_ids)}")

    new_chunks = [c for c in htree_chunker.leaf_chunks if c.metadata.get("id") not in existing_ids]
    if new_chunks:
        print(f"Adding {len(new_chunks)} leaf chunks to htree_leaves ({chroma_path}) with model {model_path}...")
        new_ids = [c.metadata["id"] for c in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)
        print("Leaf chunks indexed successfully.")
    else:
        print("No new leaf chunks to add to htree_leaves.")


def load_documents(data_path: str = DATA_PATH):
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1700,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], chroma_path: str = CHROMA_PATH, model_path: str = EMBEDDING_MODEL_PATH):
    embedding_function = HuggingFaceEmbeddings(model_name=model_path)

    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"}
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB ({chroma_path}): {len(existing_ids)}")

    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]
    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks to DB ({chroma_path}) with model {model_path}...")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("Indexed successfully.")
    else:
        print("No new documents to add.")


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


def clear_database(chroma_path: str = CHROMA_PATH):
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


if __name__ == "__main__":
    main()