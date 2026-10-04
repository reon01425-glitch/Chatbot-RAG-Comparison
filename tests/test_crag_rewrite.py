import pytest
from unittest.mock import MagicMock
from langchain.schema.document import Document
from src.engine import RAGCore, RAGEngine, REWRITE_PROMPT_TEMPLATE


def make_dummy_doc(content: str, source: str = "SOP_Izin_Cuti_Akademik.pdf", doc_id: str = "doc1"):
    return Document(page_content=content, metadata={"source": source, "id": doc_id})


@pytest.fixture
def crag_engine():
    engine = RAGEngine()
    dummy_doc = make_dummy_doc("Ketentuan cuti akademik FSM Universitas Diponegoro.")
    engine.core.db = MagicMock()
    engine.core.db.similarity_search.return_value = [dummy_doc]
    return engine


def test_crag_correct_grade_no_rewrite(crag_engine):
    """When initial retrieval score >= 0.55 (CORRECT), rewrite should NOT be triggered."""
    # Set similarity to 0.70 (>= 0.55)
    crag_engine.core.embedding_function = MagicMock()
    crag_engine.core.embedding_function.embed_query.side_effect = [
        [1.0, 0.0],  # query_embedding
        [0.7, 0.714] # doc_embedding -> cosine sim ~ 0.70
    ]
    
    call_log = []
    def fake_call_llm(prompt: str) -> str:
        call_log.append(prompt)
        return "Jawaban sintesis langsung tanpa rewrite."
        
    crag_engine.core.call_llm = fake_call_llm
    
    res = crag_engine.execute_crag("Bagaimana prosedur cuti?", k=3)
    
    # Only 1 LLM call: final synthesis (no rewrite call)
    assert len(call_log) == 1
    grade_traces = [t["type"] for t in res["trace"]]
    assert "Grade Decision: [CORRECT]" in grade_traces
    assert "CRAG LLM Query Reformulator" not in grade_traces


def test_crag_incorrect_grade_no_rewrite(crag_engine):
    """When initial retrieval score < 0.35 (INCORRECT), honest refusal is triggered without rewrite."""
    crag_engine.core.embedding_function = MagicMock()
    crag_engine.core.embedding_function.embed_query.side_effect = [
        [1.0, 0.0],  # query_embedding
        [0.1, 0.99]  # doc_embedding -> cosine sim ~ 0.10
    ]
    
    call_log = []
    crag_engine.core.call_llm = lambda p: call_log.append(p) or "dummy"
    
    res = crag_engine.execute_crag("Pertanyaan acak tidak ada di SOP?", k=3)
    
    assert "tidak ditemukan dalam basis dokumen" in res["answer"]
    # 0 LLM calls: no rewrite and no synthesis
    assert len(call_log) == 0
    grade_traces = [t["type"] for t in res["trace"]]
    assert "Grade Decision: [INCORRECT]" in grade_traces


def test_crag_ambiguous_grade_triggers_llm_rewrite(crag_engine):
    """When initial retrieval score is in [0.35, 0.55) (AMBIGUOUS), LLM rewrite is called, cleaned, and used for 2nd search."""
    crag_engine.core.embedding_function = MagicMock()
    # Sequence of embed_query calls:
    # 1. initial query
    # 2. initial doc (sim ~ 0.45 -> AMBIGUOUS)
    # 3. secondary doc (sim ~ 0.80 -> improved)
    crag_engine.core.embedding_function.embed_query.side_effect = [
        [1.0, 0.0],   # initial query_embedding
        [0.45, 0.89], # initial doc_embedding (cosine sim = 0.45)
        [0.80, 0.60]  # secondary doc_embedding (cosine sim = 0.80)
    ]
    
    call_log = []
    def fake_call_llm(prompt: str) -> str:
        call_log.append(prompt)
        if "Anda adalah sistem penulisan ulang kueri" in prompt:
            # Test cleaning: return with prefix and quotes
            return 'Kueri baru: "prosedur resmi pengajuan cuti mahasiswa FSM Undip"'
        return "Jawaban sintesis setelah query rewrite."
        
    crag_engine.core.call_llm = fake_call_llm
    
    res = crag_engine.execute_crag("cuti gimana ya", k=3)
    
    # 2 LLM calls: rewrite + synthesis
    assert len(call_log) == 2
    assert "Anda adalah sistem penulisan ulang kueri" in call_log[0]
    
    # Check that secondary retrieval used the cleaned query (no quotes, no prefix)
    second_search_call = crag_engine.core.db.similarity_search.call_args_list[-1]
    assert second_search_call[0][0] == "prosedur resmi pengajuan cuti mahasiswa FSM Undip"
    
    # Check trace
    trace_details = [t["detail"] for t in res["trace"]]
    assert any("prosedur resmi pengajuan cuti mahasiswa FSM Undip" in d for d in trace_details)
    assert "Jawaban sintesis setelah query rewrite." in res["answer"]


def test_crag_ambiguous_fails_without_silent_fallback_in_eval(crag_engine):
    """During evaluation (disable_extractive_fallback=True), failed rewrite must raise error."""
    crag_engine.core.embedding_function = MagicMock()
    crag_engine.core.embedding_function.embed_query.side_effect = [
        [1.0, 0.0],
        [0.45, 0.89]  # AMBIGUOUS
    ]
    
    crag_engine.core.disable_extractive_fallback = True
    
    # LLM rewrite returns empty string or fails
    crag_engine.core.call_llm = MagicMock(return_value="")
    
    with pytest.raises(RuntimeError) as exc_info:
        crag_engine.execute_crag("cuti gimana ya", k=3)
    assert "CRAG query rewrite via LLM failed" in str(exc_info.value)
    
    crag_engine.core.disable_extractive_fallback = False


def test_crag_ambiguous_heuristic_fallback_in_app(crag_engine):
    """In interactive app mode, failed rewrite can fallback to heuristic template with clear trace note."""
    crag_engine.core.embedding_function = MagicMock()
    crag_engine.core.embedding_function.embed_query.side_effect = [
        [1.0, 0.0],
        [0.45, 0.89], # AMBIGUOUS
        [0.70, 0.71]  # secondary
    ]
    
    crag_engine.core.disable_extractive_fallback = False
    
    # Rewrite fails with exception
    def fake_call_llm(prompt: str) -> str:
        if "penulisan ulang" in prompt:
            raise ConnectionError("Ollama timeout during rewrite")
        return "Jawaban sintesis via fallback."
        
    crag_engine.core.call_llm = fake_call_llm
    
    res = crag_engine.execute_crag("cuti gimana ya", k=3)
    
    # Trace must explicitly record heuristic fallback
    trace_types = [t["type"] for t in res["trace"]]
    assert "CRAG Heuristic Fallback" in trace_types
