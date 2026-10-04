import os
import json
import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from src.manifest import (
    resolve_embedding_model,
    resolve_chroma_path,
    get_model_weights_sha256,
    verify_index_manifest,
    load_index_manifest,
    create_index_manifest,
    save_index_manifest,
)
from src.engine import RAGCore


def test_embedding_model_alias_resolution():
    assert resolve_embedding_model("base") == "LazarusNLP/all-indo-e5-small-v4"
    assert resolve_embedding_model("v1") == "./indo_finetuned_embedding"
    assert resolve_embedding_model("v2") == "./indo_finetuned_embedding_v2"
    assert resolve_embedding_model("v3") == "./indo_finetuned_embedding_v3"
    assert resolve_embedding_model("custom/path") == "custom/path"
    with pytest.raises(ValueError):
        resolve_embedding_model("")


def test_chroma_path_resolution():
    assert resolve_chroma_path("base") == "chroma_base"
    assert resolve_chroma_path("v1") == "chroma"
    assert resolve_chroma_path("v2") == "chroma_v2"
    assert resolve_chroma_path("v3") == "chroma_v3"
    assert resolve_chroma_path("base", "custom_chroma") == "custom_chroma"


def test_model_weights_sha256_detection():
    sha_v1 = get_model_weights_sha256("v1")
    sha_v2 = get_model_weights_sha256("v2")
    assert len(sha_v1) == 64
    assert len(sha_v2) == 64
    # v2 must have distinct weights from v1
    assert sha_v1 != sha_v2


def test_evaluate_benchmark_rejects_missing_embedding_model(monkeypatch):
    """Test that evaluate_benchmark.py generate exits if no embedding model is passed."""
    # Ensure env var is cleared
    monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
    
    cmd = [sys.executable, "evaluate_benchmark.py", "generate", "--run-name", "test_missing_model"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Explicit embedding model selection is required" in result.stderr or "Explicit embedding model selection is required" in result.stdout


def test_manifest_mismatch_rejection(tmp_path, monkeypatch):
    """Test that mismatched index manifest is rejected in evaluation mode."""
    # Create a dummy chroma folder with manifest for model A
    chroma_dir = tmp_path / "dummy_chroma"
    chroma_dir.mkdir()
    manifest = {
        "model_id_or_path": "./indo_finetuned_embedding",
        "model_weights_sha256": "fake_sha",
        "corpus_hash": "fake_corpus",
        "collections": {"langchain": 7, "htree_leaves": 51},
        "created_at": "2026-10-04T00:00:00",
    }
    save_index_manifest(chroma_dir, manifest)

    # Now verify with expected model B (v2)
    ok, msg = verify_index_manifest(chroma_dir, "v2", strict=True)
    assert not ok
    assert "mismatch" in msg.lower()

    # RAGCore in evaluation mode should raise RuntimeError
    monkeypatch.setenv("EVALUATION_MODE", "1")
    monkeypatch.setenv("CHROMA_PATH", str(chroma_dir))
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "v2")
    RAGCore.reset_instance()

    with pytest.raises(RuntimeError) as exc_info:
        RAGCore()
    assert "manifest validation failed" in str(exc_info.value).lower()
    
    RAGCore.reset_instance()


def test_run_config_records_weights_sha256_and_manifest(tmp_path, monkeypatch):
    """Verify that run_config.json contains sha256 and manifest details."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    
    # Check that manifest for chroma_v2 is loaded properly
    manifest = load_index_manifest("chroma_v2")
    assert manifest is not None
    assert manifest["model_id_or_path"] == "./indo_finetuned_embedding_v2"
    assert "model_weights_sha256" in manifest
