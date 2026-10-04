import pytest
from unittest.mock import MagicMock, patch
from src.graph_sop.hierarchical_chunker import SOPHierarchicalChunker
from src.engine import RAGEngine, RAGCore


def test_hierarchical_chunker_structure_and_counts():
    """Verify that SOPHierarchicalChunker parses exactly 44 procedural steps and 51 leaf chunks across 7 SOP PDFs."""
    chunker = SOPHierarchicalChunker(data_path="data")
    assert len(chunker.document_tree) == 7, f"Expected 7 SOP documents, got {len(chunker.document_tree)}"
    assert len(chunker.leaf_chunks) == 51, f"Expected 51 leaf chunks (7 overview + 44 steps), got {len(chunker.leaf_chunks)}"

    expected_steps = {
        "SOP_Izin_Cuti_Akademik.pdf": 8,
        "SOP_Permohonan_Izin_Aktif_Setelah_Cuti.pdf": 3,
        "SOP_Legalisir_Ijazah_Dan_Transkrip.pdf": 6,
        "SOP_Pengisian_IRS.pdf": 6,
        "SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf": 11,
        "SOP_Pengajuan_Rekomendasi_Beasiswa.pdf": 5,
        "SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf": 5,
    }

    total_steps = 0
    for doc_name, exp_count in expected_steps.items():
        assert doc_name in chunker.document_tree, f"Missing document {doc_name} in tree"
        actual_count = len(chunker.document_tree[doc_name]["steps"])
        assert actual_count == exp_count, f"{doc_name}: expected {exp_count} steps, got {actual_count}"
        total_steps += actual_count

    assert total_steps == 44, f"Expected 44 total steps, got {total_steps}"


def test_step_text_continuation_and_attributes():
    """Verify that step text continuations are joined into step['text'] and only attribute labels enter details."""
    chunker = SOPHierarchicalChunker(data_path="data")
    cuti_node = chunker.document_tree["SOP_Izin_Cuti_Akademik.pdf"]
    step1 = cuti_node["steps"][0]
    
    # Text should contain multi-line sentence joined
    assert "Mahasiswa mengunduh, mengisi, dan menandatangani" in step1["text"]
    assert "SIAP" in step1["text"]

    # Details should isolate attribute fields like Dokumen yang dibutuhkan
    has_attr = any("Dokumen yang dibutuhkan" in d for d in step1["details"])
    assert has_attr, "Expected 'Dokumen yang dibutuhkan' in step details"


def test_htree_leaf_retrieval_and_parent_expansion():
    """Verify that execute_hierarchical_rag searches leaf chunks and expands to parent tree."""
    engine = RAGEngine()
    
    with patch.object(engine.core, "call_llm", return_value="Jawaban alur cuti terstruktur"):
        result = engine.execute_hierarchical_rag("alur cuti akademik", k=3, threshold=0.3)

    assert result["architecture"] == "Hierarchical Tree RAG"
    assert "Jawaban alur cuti terstruktur" in result["answer"]
    assert len(result["trace"]) >= 2
    
    # Check trace steps reflect bottom-up leaf retrieval and top-down parent tree expansion
    step1_trace = result["trace"][0]
    assert "Leaf Chunk Vector Retrieval" in step1_trace["step"]
    
    step2_trace = result["trace"][1]
    assert "Parent Document Tree Expansion" in step2_trace["step"]
    assert "Expanded root node" in step2_trace["detail"]


def test_htree_below_threshold_refusal():
    """Verify that when similarity is below threshold, honest refusal is returned without LLM call."""
    engine = RAGEngine()

    with patch.object(engine.core, "call_llm") as mock_llm:
        result = engine.execute_hierarchical_rag("pertanyaan yang sama sekali tidak relevan", threshold=0.99)
        mock_llm.assert_not_called()

    assert "tidak ditemukan dalam dokumen SOP resmi" in result["answer"]
    assert result["sources"] == []
    assert result["contexts"] == []


def test_htree_v0_ablation_baseline_dispatch():
    """Verify that htree_v0 dispatches to execute_hierarchical_rag_v0 (document chunk baseline)."""
    engine = RAGEngine()

    with patch.object(engine.core, "call_llm", return_value="Jawaban htree v0"):
        res_v0 = engine.query_architecture("Hierarchical Tree RAG v0 (Doc Chunk Baseline)", "alur cuti", threshold=0.3)
        res_v0_short = engine.query_architecture("htree_v0", "alur cuti", threshold=0.3)

    assert res_v0["architecture"] == "Hierarchical Tree RAG v0 (Doc Chunk Baseline)"
    assert res_v0_short["architecture"] == "Hierarchical Tree RAG v0 (Doc Chunk Baseline)"
    assert "v0 Baseline" in res_v0["trace"][0]["step"]

    with patch.object(engine.core, "call_llm", return_value="Jawaban htree new"):
        res_new = engine.query_architecture("Hierarchical Tree RAG", "alur cuti", threshold=0.3)
        res_new_short = engine.query_architecture("htree", "alur cuti", threshold=0.3)

    assert res_new["architecture"] == "Hierarchical Tree RAG"
    assert res_new_short["architecture"] == "Hierarchical Tree RAG"
    assert "Leaf Chunk Vector Retrieval" in res_new["trace"][0]["step"]
