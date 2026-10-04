import os
import json
import hashlib
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

EMBEDDING_MODEL_ALIASES: Dict[str, str] = {
    "base": "LazarusNLP/all-indo-e5-small-v4",
    "v1": "./indo_finetuned_embedding",
    "v2": "./indo_finetuned_embedding_v2",
    "v3": "./indo_finetuned_embedding_v3",
}

DEFAULT_CHROMA_PATHS: Dict[str, str] = {
    "base": "chroma_base",
    "v1": "chroma",
    "v2": "chroma_v2",
    "v3": "chroma_v3",
}


def resolve_embedding_model(alias_or_path: Optional[str]) -> str:
    """Resolve embedding model alias or path."""
    if not alias_or_path:
        raise ValueError("Embedding model cannot be empty. Must be one of {base, v1, v2, v3, <path>}.")
    cleaned = alias_or_path.strip()
    return EMBEDDING_MODEL_ALIASES.get(cleaned.lower(), cleaned)


def resolve_chroma_path(model_alias_or_path: str, chroma_path: Optional[str] = None) -> str:
    """Resolve chroma persist directory based on model if not explicitly specified."""
    if chroma_path:
        return chroma_path
    
    cleaned = model_alias_or_path.strip().lower()
    if cleaned in DEFAULT_CHROMA_PATHS:
        return DEFAULT_CHROMA_PATHS[cleaned]
    
    # Check resolved or substring
    resolved = resolve_embedding_model(model_alias_or_path).lower()
    if "all-indo-e5-small-v4" in resolved:
        return "chroma_base"
    elif "v3" in resolved:
        return "chroma_v3"
    elif "v2" in resolved:
        return "chroma_v2"
    elif "indo_finetuned_embedding" in resolved:
        return "chroma"
    
    return "chroma"


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA256 of a single file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_model_weights_sha256(model_name_or_path: str) -> str:
    """Compute or lookup SHA256 of embedding model weights (.safetensors or .bin)."""
    resolved = resolve_embedding_model(model_name_or_path)
    p = Path(resolved)
    if p.exists() and p.is_dir():
        for fname in ["model.safetensors", "pytorch_model.bin"]:
            w_file = p / fname
            if w_file.exists():
                return compute_file_sha256(w_file)
    
    # Check Hugging Face hub cache
    safe_name = resolved.replace("/", "--")
    hf_cache_dir = Path.home() / f".cache/huggingface/hub/models--{safe_name}/snapshots"
    if hf_cache_dir.exists():
        for snap in sorted(hf_cache_dir.iterdir(), reverse=True):
            if snap.is_dir():
                for fname in ["model.safetensors", "pytorch_model.bin"]:
                    w_file = snap / fname
                    if w_file.exists():
                        return compute_file_sha256(w_file)
                        
    return "unknown"


def compute_corpus_hash(data_dir: Union[str, Path] = "data") -> str:
    """Compute a combined SHA256 hash of all PDF files in data_dir sorted by name."""
    p = Path(data_dir)
    if not p.exists():
        return "empty"
    
    h = hashlib.sha256()
    for pdf in sorted(p.glob("*.pdf")):
        h.update(pdf.name.encode("utf-8"))
        h.update(compute_file_sha256(pdf).encode("utf-8"))
    return h.hexdigest()


def create_index_manifest(
    chroma_path: Union[str, Path],
    model_id_or_path: str,
    data_path: Union[str, Path] = "data",
    splitter_params: Optional[Dict[str, Any]] = None,
    collections_count: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Create index manifest dictionary."""
    resolved_model = resolve_embedding_model(model_id_or_path)
    weights_sha256 = get_model_weights_sha256(resolved_model)
    corpus_hash = compute_corpus_hash(data_path)
    
    if splitter_params is None:
        splitter_params = {
            "langchain": {
                "chunk_size": 1700,
                "chunk_overlap": 100,
                "length_function": "len",
            },
            "htree_leaves": {
                "chunker": "SOPHierarchicalChunker",
                "types": ["document_overview", "step"],
            },
        }
        
    if collections_count is None:
        collections_count = {}
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_path))
            for col in client.list_collections():
                collections_count[col.name] = col.count()
        except Exception:
            pass

    manifest = {
        "model_id_or_path": resolved_model,
        "model_weights_sha256": weights_sha256,
        "corpus_hash": corpus_hash,
        "splitter_params": splitter_params,
        "collections": collections_count,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return manifest


def save_index_manifest(chroma_path: Union[str, Path], manifest: Dict[str, Any]) -> Path:
    """Write index_manifest.json inside the chroma persist directory."""
    p = Path(chroma_path)
    p.mkdir(parents=True, exist_ok=True)
    manifest_path = p / "index_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path


def load_index_manifest(chroma_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load index_manifest.json from chroma persist directory if it exists."""
    p = Path(chroma_path) / "index_manifest.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def verify_index_manifest(
    chroma_path: Union[str, Path],
    expected_model: str,
    strict: bool = True
) -> Tuple[bool, str]:
    """Verify that index_manifest.json exists and matches expected model.
    Returns (is_valid, reason_message).
    """
    manifest = load_index_manifest(chroma_path)
    if manifest is None:
        msg = f"Index manifest not found at {Path(chroma_path)/'index_manifest.json'}."
        return (False if strict else True, msg)
        
    indexed_model = manifest.get("model_id_or_path", "")
    exp_resolved = resolve_embedding_model(expected_model)
    ind_resolved = resolve_embedding_model(indexed_model)
    
    # Compare paths / model names
    p_exp = Path(exp_resolved).resolve() if Path(exp_resolved).exists() else None
    p_ind = Path(ind_resolved).resolve() if Path(ind_resolved).exists() else None
    
    models_match = (
        exp_resolved == ind_resolved
        or (p_exp is not None and p_ind is not None and p_exp == p_ind)
    )
    
    if not models_match:
        msg = (
            f"Index manifest model mismatch in '{chroma_path}': "
            f"index built with '{indexed_model}', but requested model is '{expected_model}'."
        )
        return (False, msg)
        
    # Check collections presence
    collections = manifest.get("collections", {})
    if not collections or all(cnt == 0 for cnt in collections.values()):
        msg = f"Index in '{chroma_path}' contains empty collections: {collections}."
        return (False, msg)

    return (True, "Index manifest verified successfully.")
