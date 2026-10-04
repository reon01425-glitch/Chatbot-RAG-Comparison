import pytest
from unittest.mock import MagicMock, patch
from langchain.schema.document import Document
from src.engine import RAGCore, RAGEngine


def make_dummy_doc(content: str, source: str = "SOP_Izin_Cuti_Akademik.pdf", doc_id: str = "doc1"):
    return Document(page_content=content, metadata={"source": source, "id": doc_id})


@pytest.fixture
def mock_engine():
    """Create a RAGEngine with mocked vector DB and embedding function for fast unit testing."""
    engine = RAGEngine()
    
    # Mock embedding function: returns dummy vector
    engine.core.embedding_function = MagicMock()
    engine.core.embedding_function.embed_query.return_value = [1.0, 0.0, 0.0]
    
    # Mock vector database
    dummy_doc = make_dummy_doc("Langkah 1: Mahasiswa mengunduh formulir cuti akademik melalui SIAP UNDIP.")
    engine.core.db = MagicMock()
    engine.core.db.similarity_search.return_value = [dummy_doc]
    
    return engine


def test_agentic_react_stops_on_final(mock_engine):
    """Test that ReAct loop stops promptly when the LLM outputs 'Final: siap menjawab'."""
    call_log = []
    
    def fake_call_llm(prompt: str) -> str:
        call_log.append(prompt)
        if len(call_log) == 1:
            return "Thought: Saya perlu mencari SOP cuti akademik.\nAction: cari_dokumen_sop\nAction Input: cuti akademik"
        elif len(call_log) == 2:
            return "Thought: Informasi cuti sudah lengkap dari observasi.\nFinal: siap menjawab"
        else:
            return "Jawaban resmi: Prosedur cuti akademik dimulai dengan mengunduh formulir."

    mock_engine.core.call_llm = fake_call_llm
    
    res = mock_engine.execute_agentic_rag("Bagaimana alur cuti akademik?", k=3, threshold=0.3, max_steps=3)
    
    assert "Jawaban resmi" in res["answer"]
    # 2 ReAct steps + 1 synthesis call = 3 total calls to call_llm
    assert len(call_log) == 3
    trace_steps = [t["step"] for t in res["trace"]]
    assert any("Keputusan Selesai" in s for s in trace_steps)


def test_agentic_react_respects_max_steps(mock_engine):
    """Test that ReAct loop strictly terminates when max_steps is reached."""
    call_log = []
    
    def fake_call_llm(prompt: str) -> str:
        call_log.append(prompt)
        if len(call_log) <= 2:
            return "Thought: Saya masih ingin mencari lagi.\nAction: cari_dokumen_sop\nAction Input: syarat cuti"
        else:
            return "Jawaban sintesis setelah mencapai batas langkah."

    mock_engine.core.call_llm = fake_call_llm
    
    res = mock_engine.execute_agentic_rag("Syarat cuti?", k=3, threshold=0.3, max_steps=2)
    
    # Exactly 2 ReAct step calls + 1 synthesis call = 3 calls
    assert len(call_log) == 3
    assert "Jawaban sintesis" in res["answer"]


def test_agentic_react_handles_corrupted_format_gracefully(mock_engine):
    """Test that malformed/unparseable LLM output does not crash the system and stops the loop safely."""
    call_log = []
    
    def fake_call_llm(prompt: str) -> str:
        call_log.append(prompt)
        if len(call_log) == 1:
            return "Halo! Saya adalah chatbot asisten mahasiswa yang ramah. Mau tanya apa?"
        return "Jawaban sintesis."

    mock_engine.core.call_llm = fake_call_llm
    
    res = mock_engine.execute_agentic_rag("Halo apa kabar?", k=3, threshold=0.3, max_steps=3)
    
    # Loop should abort on unparseable format at step 1
    assert len(call_log) == 1
    trace_types = [t.get("type") for t in res["trace"]]
    assert "Parse Error" in trace_types
    # Since no docs were retrieved, best_score is 0.0 < threshold (0.3) -> honest refusal
    assert "tidak menemukan jawaban" in res["answer"]


def test_agentic_react_threshold_refusal(mock_engine):
    """Test that if retrieved chunks score below threshold, an honest refusal template is returned."""
    # Mock db to return empty list or low similarity
    mock_engine.core.db.similarity_search.return_value = []
    call_log = []
    
    def fake_call_llm(prompt: str) -> str:
        call_log.append(prompt)
        return "Thought: Saya cari info.\nAction: cari_dokumen_sop\nAction Input: info tidak relevan"

    mock_engine.core.call_llm = fake_call_llm
    
    res = mock_engine.execute_agentic_rag("Pertanyaan acak di luar SOP?", k=3, threshold=0.3, max_steps=2)
    
    assert "tidak menemukan jawaban" in res["answer"]
    assert res["sources"] == []
    assert res["contexts"] == []
    # No synthesis prompt should be called if below threshold
    assert len(call_log) == 2


def test_agentic_react_all_calls_via_call_llm(mock_engine):
    """Test that all LLM calls during execute_agentic_rag pass strictly through RAGCore.call_llm."""
    spy_calls = []
    
    def spy_call_llm(prompt: str) -> str:
        spy_calls.append(prompt)
        if len(spy_calls) == 1:
            return "Thought: Cari info cuti.\nAction: cari_dokumen_sop\nAction Input: alur cuti"
        elif len(spy_calls) == 2:
            return "Thought: Info lengkap.\nFinal: siap menjawab"
        return "Jawaban terverifikasi."
        
    mock_engine.core.call_llm = spy_call_llm
    
    res = mock_engine.execute_agentic_rag("Pertanyaan alur?", k=3, threshold=0.3, max_steps=3)
    
    # 2 ReAct steps + 1 synthesis call = 3 total calls strictly through call_llm
    assert len(spy_calls) == 3
    assert "Jawaban terverifikasi." in res["answer"]
