import os
import pytest
from unittest.mock import MagicMock, patch
from langchain.schema.document import Document
from src.engine import RAGEngine


def _create_mock_leaf(doc_name: str, step_num: int, content: str):
    return Document(
        page_content=content,
        metadata={
            "source": doc_name,
            "level": "procedural_step",
            "step_num": step_num,
            "id": f"{doc_name}:step_{step_num}",
        }
    )


def test_htree_variant_top1_selection():
    """Verify that top1 variant selects the parent SOP with the highest single score."""
    engine = RAGEngine()
    
    mock_leaves = [
        _create_mock_leaf("SOP_Izin_Cuti_Akademik.pdf", 1, "Prosedur Cuti A"),
        _create_mock_leaf("SOP_Pengisian_IRS.pdf", 1, "Prosedur IRS B"),
        _create_mock_leaf("SOP_Pengisian_IRS.pdf", 2, "Prosedur IRS C"),
    ]
    
    mock_emb = MagicMock()
    mock_emb.embed_query.side_effect = [
        [1.0, 0.0],  # query
        [0.9, 0.0],  # doc 1 -> sim 0.90 (Cuti)
        [0.7, 0.0],  # doc 2 -> sim 0.70 (IRS)
        [0.6, 0.0],  # doc 3 -> sim 0.60 (IRS)
    ]
    
    with patch.object(engine.core.htree_db, "similarity_search", return_value=mock_leaves), \
         patch.object(engine.core, "embedding_function", mock_emb), \
         patch.object(engine.core, "call_llm", return_value="Jawaban"):
        
        res = engine.execute_hierarchical_rag("pertanyaan uji", variant="top1")
        assert "SOP_Izin_Cuti_Akademik.pdf" in res["trace"][1]["detail"]
        assert "variant=top1" in res["trace"][1]["detail"]
        assert "Parent Document Tree Expansion (top1)" in res["trace"][1]["step"]


def test_htree_variant_vote_selection():
    """Verify that vote variant selects majority parent SOP even if top-1 leaf is from another SOP."""
    engine = RAGEngine()
    
    mock_leaves = [
        _create_mock_leaf("SOP_Izin_Cuti_Akademik.pdf", 1, "Prosedur Cuti A"),
        _create_mock_leaf("SOP_Pengisian_IRS.pdf", 1, "Prosedur IRS B"),
        _create_mock_leaf("SOP_Pengisian_IRS.pdf", 2, "Prosedur IRS C"),
    ]
    
    mock_emb = MagicMock()
    mock_emb.embed_query.side_effect = [
        [1.0, 0.0],  # query
        [0.9, 0.0],  # doc 1 -> sim 0.90 (Cuti)
        [0.7, 0.0],  # doc 2 -> sim 0.70 (IRS)
        [0.6, 0.0],  # doc 3 -> sim 0.60 (IRS)
    ]
    
    with patch.object(engine.core.htree_db, "similarity_search", return_value=mock_leaves), \
         patch.object(engine.core, "embedding_function", mock_emb), \
         patch.object(engine.core, "call_llm", return_value="Jawaban"):
        
        res = engine.execute_hierarchical_rag("pertanyaan uji", variant="vote")
        assert "SOP_Pengisian_IRS.pdf" in res["trace"][1]["detail"]
        assert "majority vote" in res["trace"][1]["detail"]
        assert "Parent Document Tree Expansion (vote)" in res["trace"][1]["step"]


def test_htree_variant_sum_selection():
    """Verify that sum variant selects parent SOP with highest cumulative similarity."""
    engine = RAGEngine()
    
    mock_leaves = [
        _create_mock_leaf("SOP_Izin_Cuti_Akademik.pdf", 1, "Prosedur Cuti A"),
        _create_mock_leaf("SOP_Pengisian_IRS.pdf", 1, "Prosedur IRS B"),
        _create_mock_leaf("SOP_Pengisian_IRS.pdf", 2, "Prosedur IRS C"),
    ]
    
    mock_emb = MagicMock()
    mock_emb.embed_query.side_effect = [
        [1.0, 0.0],  # query
        [0.85, 0.0], # doc 1 -> sim 0.85 (Cuti sum = 0.85)
        [0.60, 0.0], # doc 2 -> sim 0.60 (IRS sum = 1.20)
        [0.60, 0.0], # doc 3 -> sim 0.60
    ]
    
    with patch.object(engine.core.htree_db, "similarity_search", return_value=mock_leaves), \
         patch.object(engine.core, "embedding_function", mock_emb), \
         patch.object(engine.core, "call_llm", return_value="Jawaban"):
        
        res = engine.execute_hierarchical_rag("pertanyaan uji", variant="sum")
        assert "SOP_Pengisian_IRS.pdf" in res["trace"][1]["detail"]
        assert "score sum" in res["trace"][1]["detail"]
        assert "Parent Document Tree Expansion (sum)" in res["trace"][1]["step"]


def test_htree_env_override():
    """Verify that HTREE_VARIANT environment variable controls the selection algorithm."""
    engine = RAGEngine()
    
    mock_leaves = [
        _create_mock_leaf("SOP_Izin_Cuti_Akademik.pdf", 1, "Prosedur Cuti A"),
        _create_mock_leaf("SOP_Pengisian_IRS.pdf", 1, "Prosedur IRS B"),
        _create_mock_leaf("SOP_Pengisian_IRS.pdf", 2, "Prosedur IRS C"),
    ]
    
    mock_emb = MagicMock()
    mock_emb.embed_query.side_effect = [
        [1.0, 0.0],
        [0.9, 0.0],
        [0.7, 0.0],
        [0.6, 0.0],
    ]
    
    with patch.dict(os.environ, {"HTREE_VARIANT": "vote"}), \
         patch.object(engine.core.htree_db, "similarity_search", return_value=mock_leaves), \
         patch.object(engine.core, "embedding_function", mock_emb), \
         patch.object(engine.core, "call_llm", return_value="Jawaban"):
        
        res = engine.execute_hierarchical_rag("pertanyaan uji")
        assert "Parent Document Tree Expansion (vote)" in res["trace"][1]["step"]
