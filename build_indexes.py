#!/usr/bin/env python3
"""Build or rebuild vector indexes for SOP documents from scratch for a specified embedding model.

Usage:
  python build_indexes.py --embedding-model v3
  python build_indexes.py --embedding-model base --reset
  python build_indexes.py --embedding-model v2 --manifest-only
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.manifest import (
    resolve_embedding_model,
    resolve_chroma_path,
    create_index_manifest,
    save_index_manifest,
    verify_index_manifest,
    get_model_weights_sha256,
)
from src.graph_sop.hierarchical_chunker import SOPHierarchicalChunker
from embeddings import calculate_chunk_ids


def build_index(
    embedding_model: str,
    chroma_path: str = None,
    data_path: str = "data",
    reset: bool = True,
    manifest_only: bool = False,
):
    model_name = resolve_embedding_model(embedding_model)
    target_chroma = resolve_chroma_path(embedding_model, chroma_path)

    print(f"==================================================")
    print(f"Building/Validating Index for: {embedding_model}")
    print(f"  - Model Path/ID:   {model_name}")
    print(f"  - Model SHA256:    {get_model_weights_sha256(model_name)}")
    print(f"  - Target Chroma:   {target_chroma}")
    print(f"  - Data Path:       {data_path}")
    print(f"  - Reset DB:        {reset}")
    print(f"==================================================")

    if manifest_only:
        print(f"Generating manifest for existing index at '{target_chroma}'...")
        manifest = create_index_manifest(
            chroma_path=target_chroma,
            model_id_or_path=model_name,
            data_path=data_path,
        )
        saved_path = save_index_manifest(target_chroma, manifest)
        print(f"Manifest written to {saved_path}:")
        print(manifest)
        ok, msg = verify_index_manifest(target_chroma, model_name, strict=True)
        print(f"Verification: {msg}")
        return

    if reset and os.path.exists(target_chroma):
        print(f"Clearing existing database at: {target_chroma}")
        shutil.rmtree(target_chroma)

    os.makedirs(target_chroma, exist_ok=True)

    print("\n1. Loading and splitting standard PDF documents...")
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1700,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    chunks_with_ids = calculate_chunk_ids(chunks)
    print(f"   Loaded {len(documents)} pages, created {len(chunks_with_ids)} document chunks.")

    print("\n2. Initializing embedding function...")
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    print(f"\n3. Indexing document chunks into '{target_chroma}' (collection: langchain)...")
    db_main = Chroma(
        persist_directory=target_chroma,
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"}
    )
    chunk_ids = [c.metadata["id"] for c in chunks_with_ids]
    db_main.add_documents(chunks_with_ids, ids=chunk_ids)
    print(f"   Indexed {len(chunks_with_ids)} documents into 'langchain'.")

    print(f"\n4. Indexing hierarchical tree leaves into '{target_chroma}' (collection: htree_leaves)...")
    htree_chunker = SOPHierarchicalChunker(data_path=data_path)
    leaf_chunks = htree_chunker.leaf_chunks
    leaf_ids = [c.metadata["id"] for c in leaf_chunks]

    db_htree = Chroma(
        collection_name="htree_leaves",
        persist_directory=target_chroma,
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"}
    )
    db_htree.add_documents(leaf_chunks, ids=leaf_ids)
    print(f"   Indexed {len(leaf_chunks)} leaf chunks into 'htree_leaves'.")

    print("\n5. Creating and writing index manifest...")
    manifest = create_index_manifest(
        chroma_path=target_chroma,
        model_id_or_path=model_name,
        data_path=data_path,
        collections_count={
            "langchain": len(chunks_with_ids),
            "htree_leaves": len(leaf_chunks),
        },
    )
    saved_path = save_index_manifest(target_chroma, manifest)
    print(f"   Manifest written to {saved_path}")

    ok, msg = verify_index_manifest(target_chroma, model_name, strict=True)
    if not ok:
        raise RuntimeError(f"Manifest verification failed immediately after building: {msg}")
    print(f"   Validation: {msg}")
    print("\nIndex build complete and verified!")


def main():
    parser = argparse.ArgumentParser(description="Build Chroma vector store and manifest for SOP documents.")
    parser.add_argument(
        "--embedding-model",
        required=True,
        choices=["base", "v1", "v2", "v3"] + list(resolve_embedding_model.__annotations__.values()),
        help="Embedding model choice: {base, v1, v2, v3, <path>}",
    )
    parser.add_argument("--chroma-path", default=None, help="Persist directory for Chroma DB (default inferred)")
    parser.add_argument("--data-path", default="data", help="Path to SOP PDFs (default: data)")
    parser.add_argument("--reset", action="store_true", default=True, help="Clear directory before indexing (default: True)")
    parser.add_argument("--no-reset", action="store_false", dest="reset", help="Do not clear directory before indexing")
    parser.add_argument("--manifest-only", action="store_true", help="Only generate manifest for existing index")

    args = parser.parse_args()
    build_index(
        embedding_model=args.embedding_model,
        chroma_path=args.chroma_path,
        data_path=args.data_path,
        reset=args.reset,
        manifest_only=args.manifest_only,
    )


if __name__ == "__main__":
    main()
