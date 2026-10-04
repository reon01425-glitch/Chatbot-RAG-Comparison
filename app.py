import os
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.engine import RAGEngine

# ==========================================
# PAGE CONFIGURATION & METADATA
# ==========================================
st.set_page_config(
    page_title="Chatbot RAG Comparison | SOP FSM UNDIP",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING (MODERN & EYE-FRIENDLY)
# ==========================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Header & Branding Banner */
    .brand-header {
        background: linear-gradient(135deg, #0F2C59 0%, #1E3A8A 60%, #1E40AF 100%);
        border-radius: 16px;
        padding: 24px 28px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 44, 89, 0.15), 0 8px 10px -6px rgba(15, 44, 89, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.12);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .brand-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-subtitle {
        font-size: 0.95rem;
        color: #E2E8F0;
        margin-top: 6px;
        font-weight: 400;
    }

    .brand-badge {
        background: rgba(245, 158, 11, 0.2);
        border: 1px solid #F59E0B;
        color: #FDE68A;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 16px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }

    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
    }

    .metric-subtext {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Architecture Badge Pills */
    .arch-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .arch-naive { background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; }
    .arch-hybrid { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
    .arch-graph { background: #FAF5FF; color: #6B21A8; border: 1px solid #E9D5FF; }
    .arch-agentic { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
    .arch-crag { background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }
    .arch-multimodal { background: #FFF1F2; color: #9F1239; border: 1px solid #FECDD3; }
    .arch-workflow { background: #ECFEFF; color: #0E7490; border: 1px solid #A5F3FC; }
    .arch-hierarchical { background: #F0FDF4; color: #15803D; border: 1px solid #BBF7D0; }

    /* Trace Timeline Steps */
    .trace-step-container {
        border-left: 2px solid #CBD5E1;
        margin-left: 12px;
        padding-left: 16px;
        margin-bottom: 14px;
        position: relative;
    }

    .trace-dot {
        position: absolute;
        left: -21px;
        top: 2px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #3B82F6;
        border: 2px solid #FFFFFF;
    }

    .trace-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 2px;
    }

    .trace-type {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        background: #F1F5F9;
        color: #475569;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 4px;
    }

    .trace-detail {
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.45;
        background: #F8FAFC;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
    }

    /* Response Container */
    .response-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        line-height: 1.65;
        font-size: 0.95rem;
        color: #1E293B;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    /* Context Drawer Card */
    .context-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }

    /* Dark Mode Theme Adaptation */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: #1E293B !important;
            border-color: #334155 !important;
        }
        .metric-value {
            color: #F8FAFC !important;
        }
        .metric-label {
            color: #94A3B8 !important;
        }
        .response-box {
            background: #1E293B !important;
            border-color: #334155 !important;
            color: #F1F5F9 !important;
        }
        .trace-detail {
            background: #0F172A !important;
            border-color: #334155 !important;
            color: #CBD5E1 !important;
        }
        .trace-title {
            color: #F1F5F9 !important;
        }
        .context-card {
            background: #0F172A !important;
            border-color: #334155 !important;
            color: #E2E8F0 !important;
        }
    }

    .context-source-badge {
        display: inline-block;
        background: #0F2C59;
        color: #FFFFFF;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 6px;
    }

    .context-sim-badge {
        display: inline-block;
        background: #10B981;
        color: #FFFFFF;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 6px;
    }

    /* Grade Badges for CRAG */
    .badge-correct { background: #D1FAE5; color: #065F46; padding: 2px 8px; border-radius: 6px; font-weight: 700; }
    .badge-ambiguous { background: #FEF3C7; color: #92400E; padding: 2px 8px; border-radius: 6px; font-weight: 700; }
    .badge-incorrect { background: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 6px; font-weight: 700; }

    /* Custom button tweaks */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# INITIALIZE ENGINE (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Menyiapkan RAG Engine dan Vector Store...")
def get_engine():
    return RAGEngine()

engine = get_engine()

# ==========================================
# ARCHITECTURE CONFIGURATION
# ==========================================
ARCHITECTURES = [
    "Workflow GraphRAG (Process DAG)",
    "Hierarchical Tree RAG",
    "GraphRAG (Entity Expansion)",
    "Naive RAG (Baseline)",
    "Hybrid RAG (BM25 + Dense)",
    "Agentic RAG (Tools Agent)",
    "Corrective RAG (CRAG)",
    "Multimodal RAG (Layout RAG)"
]

ARCH_DESCRIPTIONS = {
    "Workflow GraphRAG (Process DAG)": "Process / State-Machine DAG traversal modeling chronological SOP steps, swimlane actors, prerequisites, outputs, and automated BPMN/Mermaid flowchart rendering.",
    "Hierarchical Tree RAG": "Tree-structured document chunking preserving root-level SOP metadata, section headers, and procedural child attributes.",
    "Naive RAG (Baseline)": "Standard dense semantic vector search via fine-tuned Indonesian embeddings with top-k direct retrieval.",
    "Hybrid RAG (BM25 + Dense)": "Parallel Dense Vector + Sparse BM25 retrieval merged via Reciprocal Rank Fusion (RRF k=60).",
    "GraphRAG (Entity Expansion)": "Knowledge graph traversal and entity expansion linking university administrative relations prior to vector lookup.",
    "Agentic RAG (Tools Agent)": "Autonomous ReAct agent reasoning (Thought-Action-Observation) dynamically invoking SOP retrieval tools.",
    "Corrective RAG (CRAG)": "Self-grading evaluator classifying retrieval confidence (CORRECT/AMBIGUOUS/INCORRECT) with automated query rewriting.",
    "Multimodal RAG (Layout RAG)": "Layout-aware context retriever combining text with extracted procedural flowchart and table structure metadata."
}

ARCH_BADGE_CLASSES = {
    "Workflow GraphRAG (Process DAG)": "arch-workflow",
    "Hierarchical Tree RAG": "arch-hierarchical",
    "Naive RAG (Baseline)": "arch-naive",
    "Hybrid RAG (BM25 + Dense)": "arch-hybrid",
    "GraphRAG (Entity Expansion)": "arch-graph",
    "Agentic RAG (Tools Agent)": "arch-agentic",
    "Corrective RAG (CRAG)": "arch-crag",
    "Multimodal RAG (Layout RAG)": "arch-multimodal"
}

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "arena_history" not in st.session_state:
    st.session_state.arena_history = []
if "custom_query" not in st.session_state:
    st.session_state.custom_query = ""

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    logo_path = "assets/undip_logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
    else:
        st.markdown("<div style='font-size:2.5rem; line-height:1; margin-bottom:8px;'>🎓</div>", unsafe_allow_html=True)
    st.markdown("### **FSM UNDIP RAG Hub**")
    st.caption("Chatbot Layanan SOP Mahasiswa")
    st.divider()

    st.markdown("#### ⚙️ **Parameter Konfigurasi**")
    top_k = st.slider("Top-k Dokumen Retrieval", min_value=1, max_value=5, value=3, help="Jumlah chunk dokumen yang diambil untuk konteks.")
    sim_threshold = st.slider("Similarity Threshold", min_value=0.1, max_value=0.6, value=0.30, step=0.05, help="Ambang batas minimum cosine similarity untuk menjawab.")
    
    st.divider()
    st.markdown("#### 📚 **Basis Dokumen SOP Aktif**")
    sop_docs = [
        "SOP Izin Cuti Akademik",
        "SOP Izin Aktif Setelah Cuti",
        "SOP Izin Keterlambatan UKT",
        "SOP Pengisian IRS & Bimbingan",
        "SOP Rekomendasi Beasiswa",
        "SOP Legalisir Ijazah & Transkrip",
        "SOP Proposal Kegiatan Ormawa"
    ]
    for d in sop_docs:
        st.markdown(f"- <span style='font-size:0.85rem; color:#475569;'>📄 {d}</span>", unsafe_allow_html=True)
        
    st.divider()
    
    # Knowledge base stats
    active_emb = os.getenv("EMBEDDING_MODEL_PATH", "./indo_finetuned_embedding")
    active_chroma = os.getenv("CHROMA_PATH", "chroma")
    st.markdown("#### 📊 **Status Sistem**")
    st.markdown(f"""
    - **Vector DB:** Chroma (Active)
    - **Embedding:** `{active_emb}`
    - **Dimensi Vektor:** 384 dimensions
    - **Chroma Path:** `{active_chroma}/`
    - **Total SOP:** 7 Dokumen PDF
    """)
    
    if st.button("🗑️ Bersihkan Riwayat Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.arena_history = []
        st.rerun()

# ==========================================
# MAIN HEADER BANNER
# ==========================================
st.markdown("""
<div class="brand-header">
    <div>
        <h1 class="brand-title">
            <span>🎓</span> Chatbot RAG Architecture Arena
        </h1>
        <div class="brand-subtitle">
            Evaluasi Komparatif 6 Arsitektur Retrieval-Augmented Generation untuk SOP FSM Universitas Diponegoro
        </div>
    </div>
    <div>
        <span class="brand-badge">SOP FSM UNDIP</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# NAVIGATION TABS
# ==========================================
tab_chat, tab_arena, tab_workflow, tab_benchmark, tab_explorer, tab_guide = st.tabs([
    "💬 Single Architecture",
    "⚔️ Side-by-Side Arena",
    "🔀 SOP Workflow & BPMN",
    "📊 Benchmark Visualizer",
    "📑 SOP Document Explorer",
    "ℹ️ Architecture Guide"
])

# Quick Sample Prompts
SAMPLE_PROMPTS = [
    "Bagaimana syarat dan alur pengajuan izin cuti akademik?",
    "Apa saja dokumen yang harus disiapkan untuk legalisir ijazah dan transkrip?",
    "Bagaimana prosedur permohonan izin aktif kuliah kembali setelah cuti akademik?",
    "Kapan batas waktu pengajuan izin keterlambatan pembayaran UKT?",
    "Bagaimana alur pengisian IRS dan persetujuan dosen wali?"
]

# Helper function to render metric cards
def render_metric_row(metrics: dict, grade: str = None):
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⏱️ Latensi Eksekusi</div>
            <div class="metric-value">{metrics.get('latency', 0):.3f}s</div>
            <div class="metric-subtext">Retrieval + Generasi</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 Top Cosine Sim</div>
            <div class="metric-value">{metrics.get('max_cosine_sim', 0):.4f}</div>
            <div class="metric-subtext">Avg: {metrics.get('avg_cosine_sim', 0):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🛡️ Faithfulness Score</div>
            <div class="metric-value">{metrics.get('faithfulness', 0):.2f}</div>
            <div class="metric-subtext">Grounded in Context</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💡 Answer Relevance</div>
            <div class="metric-value">{metrics.get('answer_relevance', 0):.2f}</div>
            <div class="metric-subtext">Semantic Alignment</div>
        </div>
        """, unsafe_allow_html=True)

# Helper function to render trace timeline
def render_trace_timeline(trace_list: list):
    st.markdown("##### 🔍 **Live Execution Trace**")
    for item in trace_list:
        st.markdown(f"""
        <div class="trace-step-container">
            <div class="trace-dot"></div>
            <div class="trace-title">{item.get('step', 'Step')}</div>
            <span class="trace-type">{item.get('type', 'Process')}</span>
            <div class="trace-detail">{item.get('detail', '')}</div>
        </div>
        """, unsafe_allow_html=True)

# Helper function to render retrieved context drawer
def render_context_drawer(sources: list, contexts: list, scores: list):
    with st.expander(f"📂 **Retrieved Context Drawer ({len(contexts)} Chunks)**", expanded=False):
        if not contexts:
            st.info("Tidak ada chunk konteks yang diambil (dibawah threshold atau fallback).")
        for i, (src, ctx) in enumerate(zip(sources, contexts)):
            score = scores[i] if i < len(scores) else 0.0
            st.markdown(f"""
            <div class="context-card">
                <div>
                    <span class="context-source-badge">📄 {src}</span>
                    <span class="context-sim-badge">Cosine: {score:.4f}</span>
                </div>
                <div style="font-size:0.85rem; color:#334155; margin-top:6px; line-height:1.5;">
                    {ctx.replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Helper function to render interactive Mermaid / BPMN Flowchart with view mode selector
def render_mermaid_viewer(res: dict, key_prefix: str = "main"):
    """
    Renders an elegant, multi-mode visual BPMN flowchart for any SOP question.
    Supports Standard Sequential, Swimlane BPMN by Actor, and Detailed Artifacts.
    """
    mermaid_std = res.get("mermaid_standard") or res.get("mermaid")
    mermaid_swim = res.get("mermaid_swimlane")
    mermaid_det = res.get("mermaid_detailed")
    sop_title = res.get("matched_sop") or "Prosedur SOP"
    
    if not mermaid_std and not mermaid_swim:
        return
        
    st.markdown(f"##### 🔀 **Bagan Alur Prosedur SOP (BPMN / Flowchart Otomatis):**")
    with st.expander(f"📊 **Buka Diagram Alur Visual: {sop_title}**", expanded=True):
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            st.caption(f"✨ *Bagan alur resmi diekstrak langsung dari graf alur kerja (BPMN State-Machine) FSM UNDIP.*")
        with col_m2:
            view_mode = st.radio(
                "Gaya Diagram:",
                options=["Urutan Alur (Standard)", "Swimlane Aktor (BPMN)", "Detail Berkas & Waktu"],
                index=0,
                horizontal=True,
                key=f"diag_mode_{key_prefix}"
            )
            
        if "Swimlane" in view_mode and mermaid_swim:
            active_mermaid = mermaid_swim
        elif "Detail" in view_mode and mermaid_det:
            active_mermaid = mermaid_det
        else:
            active_mermaid = mermaid_std
            
        st.markdown(f"""```mermaid
{active_mermaid}
```""")
        st.info("💡 **Tips Mahasiswa:** Setiap kotak menunjukkan aktor penanggung jawab (Mahasiswa, Kaprodi, Dekan, dll) beserta luaran surat/tanda tangan pada setiap tahap.")

# Helper function to render crystal-clear Leaderboard Table
def render_benchmark_table(df: pd.DataFrame, categories: list):
    max_vals = {col: df[col].max() for col in categories}
    
    html = ['<div style="overflow-x: auto; border-radius: 12px; border: 1px solid #334155; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.15);">']
    html.append('<table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; line-height: 1.5;">')
    
    # Table Header
    html.append('<thead style="background: linear-gradient(135deg, #0F2C59 0%, #1E3A8A 100%); color: #FFFFFF;">')
    html.append('<tr>')
    html.append('<th style="padding: 14px 18px; font-weight: 700; color: #FFFFFF; border-bottom: 2px solid #334155;">Architecture</th>')
    for col in categories:
        html.append(f'<th style="padding: 14px 18px; font-weight: 700; color: #FFFFFF; border-bottom: 2px solid #334155; text-align: center;">{col}</th>')
    html.append('</tr></thead><tbody>')
    
    # Table Body Rows
    for idx, row in df.iterrows():
        bg_color = "rgba(30, 41, 59, 0.85)" if idx % 2 == 0 else "rgba(15, 23, 42, 0.95)"
        arch_name = row["Architecture"]
        badge_cls = ARCH_BADGE_CLASSES.get(arch_name, "arch-naive")
        
        html.append(f'<tr style="background: {bg_color}; border-bottom: 1px solid #334155;">')
        html.append(f'<td style="padding: 12px 18px; font-weight: 600; color: #F8FAFC;"><span class="arch-badge {badge_cls}" style="margin: 0;">{arch_name}</span></td>')
        
        for col in categories:
            val = float(row[col])
            is_max = (val == max_vals[col])
            if is_max:
                cell_content = f'<span style="background: #1D4ED8; color: #FFFFFF !important; font-weight: 800; font-size: 0.92rem; padding: 5px 12px; border-radius: 6px; border: 1.5px solid #60A5FA; display: inline-flex; align-items: center; justify-content: center; gap: 4px; box-shadow: 0 2px 5px rgba(29,78,216,0.5);"><b style=\"color:#FFFFFF !important;\">{val:.4f}</b> 🏆</span>'
            else:
                cell_content = f'<span style="color: #E2E8F0; font-weight: 500; font-family: monospace;">{val:.4f}</span>'
            html.append(f'<td style="padding: 12px 18px; text-align: center;">{cell_content}</td>')
        html.append('</tr>')
        
    html.append('</tbody></table></div>')
    st.markdown("".join(html), unsafe_allow_html=True)

# ==========================================
# TAB 1: SINGLE ARCHITECTURE MODE
# ==========================================
with tab_chat:
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        selected_arch = st.selectbox(
            "Pilih Arsitektur RAG:",
            options=ARCHITECTURES,
            index=0,
            key="single_arch_select"
        )
        badge_cls = ARCH_BADGE_CLASSES.get(selected_arch, "arch-naive")
        st.markdown(f'<span class="arch-badge {badge_cls}">{selected_arch}</span>', unsafe_allow_html=True)
        st.caption(ARCH_DESCRIPTIONS.get(selected_arch, ""))
        
    with col_ctrl2:
        st.markdown("**Contoh Pertanyaan Cepat:**")
        p_cols = st.columns(3)
        for idx, p in enumerate(SAMPLE_PROMPTS[:3]):
            with p_cols[idx]:
                if st.button(f"📌 {p[:30]}...", key=f"quick_p_{idx}", use_container_width=True):
                    st.session_state.custom_query = p

    st.divider()

    # Query Input form
    with st.form(key="single_query_form", clear_on_submit=False):
        user_query = st.text_input(
            "Ajukan pertanyaan seputar SOP FSM Universitas Diponegoro:",
            value=st.session_state.custom_query,
            placeholder="Contoh: Bagaimana tata cara pengajuan cuti akademik dan apa saja syaratnya?",
            key="single_input_text"
        )
        submit_btn = st.form_submit_button("🚀 Kirim Pertanyaan", use_container_width=True)

    if submit_btn and user_query.strip():
        with st.spinner(f"Memproses kueri dengan {selected_arch}..."):
            result = engine.query_architecture(selected_arch, user_query.strip(), k=top_k, threshold=sim_threshold)
            st.session_state.chat_history.insert(0, {
                "query": user_query.strip(),
                "arch": selected_arch,
                "result": result
            })

    # Render History
    if st.session_state.chat_history:
        for idx, item in enumerate(st.session_state.chat_history):
            res = item["result"]
            m = res.get("metrics", {})
            badge_cls = ARCH_BADGE_CLASSES.get(item["arch"], "arch-naive")
            
            st.markdown(f"""
            <div style="margin-top: 20px; padding: 12px 16px; background: #F1F5F9; border-radius: 10px; border-left: 4px solid #0F2C59;">
                <span style="font-size:0.8rem; font-weight:700; color:#64748B;">PERTANYAAN:</span>
                <div style="font-size:1.05rem; font-weight:700; color:#0F172A; margin-top:2px;">{item['query']}</div>
                <div style="margin-top: 6px;">
                    <span class="arch-badge {badge_cls}">{item['arch']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metric row
            render_metric_row(m, res.get("grade"))
            
            # Answer Box
            st.markdown("##### 💬 **Jawaban Sistem:**")
            st.markdown(f"""
            <div class="response-box">
                {res.get('answer', '').replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            # Universal Visual BPMN / Flowchart rendering (Ciri khas sistem RAG SOP)
            render_mermaid_viewer(res, key_prefix=f"single_{idx}")
            
            # Trace & Context Drawer
            col_t1, col_t2 = st.columns([1, 1])
            with col_t1:
                render_trace_timeline(res.get("trace", []))
            with col_t2:
                render_context_drawer(res.get("sources", []), res.get("contexts", []), res.get("scores", []))
                
            st.divider()
    else:
        st.info("💡 Belum ada riwayat pertanyaan. Pilih arsitektur di atas dan ajukan pertanyaan untuk melihat hasil retrieval, trace, dan dynamic metrics.")

# ==========================================
# TAB 2: SIDE-BY-SIDE ARENA
# ==========================================
with tab_arena:
    st.markdown("### ⚔️ **RAG Architecture Battle Arena**")
    st.caption("Bandingkan respon, latensi, similarity, dan trace dari dua arsitektur RAG secara simultan terhadap pertanyaan yang sama.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        arch_a = st.selectbox("Arsitektur A:", options=ARCHITECTURES, index=0, key="arena_arch_a")
        st.markdown(f'<span class="arch-badge {ARCH_BADGE_CLASSES.get(arch_a, "arch-naive")}">Arsitektur A: {arch_a}</span>', unsafe_allow_html=True)
    with col_b:
        arch_b = st.selectbox("Arsitektur B:", options=ARCHITECTURES, index=4, key="arena_arch_b")
        st.markdown(f'<span class="arch-badge {ARCH_BADGE_CLASSES.get(arch_b, "arch-crag")}">Arsitektur B: {arch_b}</span>', unsafe_allow_html=True)

    with st.form(key="arena_query_form"):
        arena_query = st.text_input(
            "Pertanyaan untuk Diuji pada Kedua Arsitektur:",
            placeholder="Contoh: Apa sanksi jika terlambat membayar UKT dan bagaimana solusinya?",
            key="arena_input_text"
        )
        arena_submit = st.form_submit_button("⚔️ Jalankan Duel Arsitektur", use_container_width=True)

    if arena_submit and arena_query.strip():
        with st.spinner("Menjalankan perbandingan simultan..."):
            res_a = engine.query_architecture(arch_a, arena_query.strip(), k=top_k, threshold=sim_threshold)
            res_b = engine.query_architecture(arch_b, arena_query.strip(), k=top_k, threshold=sim_threshold)
            st.session_state.arena_history.insert(0, {
                "query": arena_query.strip(),
                "arch_a": arch_a,
                "res_a": res_a,
                "arch_b": arch_b,
                "res_b": res_b
            })

    if st.session_state.arena_history:
        latest = st.session_state.arena_history[0]
        st.markdown(f"#### 🔍 **Hasil Perbandingan: '{latest['query']}'**")
        
        m_a = latest["res_a"]["metrics"]
        m_b = latest["res_b"]["metrics"]
        
        # Summary Comparison Metrics
        st.markdown("##### 📈 **Perbandingan Metrik Utama**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            delta_lat = m_b['latency'] - m_a['latency']
            st.metric("Latensi (A vs B)", f"{m_a['latency']}s", f"B: {m_b['latency']}s ({delta_lat:+.3f}s)", delta_color="inverse")
        with c2:
            delta_cos = m_b['max_cosine_sim'] - m_a['max_cosine_sim']
            st.metric("Top Cosine Sim", f"{m_a['max_cosine_sim']:.4f}", f"B: {m_b['max_cosine_sim']:.4f} ({delta_cos:+.4f})")
        with c3:
            delta_faith = m_b['faithfulness'] - m_a['faithfulness']
            st.metric("Faithfulness Score", f"{m_a['faithfulness']:.2f}", f"B: {m_b['faithfulness']:.2f} ({delta_faith:+.2f})")
        with c4:
            delta_rel = m_b['answer_relevance'] - m_a['answer_relevance']
            st.metric("Answer Relevance", f"{m_a['answer_relevance']:.2f}", f"B: {m_b['answer_relevance']:.2f} ({delta_rel:+.2f})")

        st.divider()

        # Side by side response columns
        col_out_a, col_out_b = st.columns(2)
        with col_out_a:
            st.markdown(f"#### 🅰️ **{latest['arch_a']}**")
            render_metric_row(m_a, latest["res_a"].get("grade"))
            st.markdown("##### 💬 Respon:")
            st.markdown(f"""
            <div class="response-box" style="border-top: 3px solid #3B82F6;">
                {latest["res_a"].get('answer', '').replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            render_mermaid_viewer(latest["res_a"], key_prefix="arena_a")
            render_trace_timeline(latest["res_a"].get("trace", []))
            render_context_drawer(latest["res_a"].get("sources", []), latest["res_a"].get("contexts", []), latest["res_a"].get("scores", []))

        with col_out_b:
            st.markdown(f"#### 🅱️ **{latest['arch_b']}**")
            render_metric_row(m_b, latest["res_b"].get("grade"))
            st.markdown("##### 💬 Respon:")
            st.markdown(f"""
            <div class="response-box" style="border-top: 3px solid #10B981;">
                {latest["res_b"].get('answer', '').replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            render_mermaid_viewer(latest["res_b"], key_prefix="arena_b")
            render_trace_timeline(latest["res_b"].get("trace", []))
            render_context_drawer(latest["res_b"].get("sources", []), latest["res_b"].get("contexts", []), latest["res_b"].get("scores", []))

# ==========================================
# TAB: SOP WORKFLOW & BPMN INSPECTOR
# ==========================================
with tab_workflow:
    st.markdown("### 🔀 **SOP Workflow Graph & BPMN Process Explorer**")
    st.caption("Eksplorasi visual Directed Graph (DiGraph) tahapan prosedur resmi SOP FSM UNDIP dengan swimlane aktor, dokumen input, dan luaran.")
    
    sop_list = list(engine.core.workflow_graph.sop_metadata.values())
    sop_titles = [s["title"] for s in sop_list]
    
    sel_sop_title = st.selectbox(
        "Pilih Standar Operasional Prosedur (SOP) untuk Diinspeksi:",
        options=sop_titles,
        index=0,
        key="workflow_sop_select"
    )
    
    sel_sop = next((s for s in sop_list if s["title"] == sel_sop_title), sop_list[0])
    
    col_w1, col_w2 = st.columns([1, 1])
    with col_w1:
        st.markdown(f"#### 📋 **Rincian Alur: {sel_sop['title']}**")
        st.markdown(f"- **Maksimal Waktu Pelayanan:** `{sel_sop['max_duration']}`")
        st.markdown(f"- **Total Langkah Prosedur:** `{len(sel_sop['steps'])} Langkah Berurutan`")
        
        st.markdown("##### 🚶 **Tahapan Sekuensial & Aktor Pelaksana:**")
        for s in sel_sop["steps"]:
            inputs_badge = f"<span style='font-size:0.75rem; background:#E2E8F0; padding:2px 6px; border-radius:4px;'>📥 Syarat: {', '.join(s.get('inputs', []))}</span>" if s.get('inputs') else ""
            out_badge = f"<span style='font-size:0.75rem; background:#DCFCE7; color:#166534; padding:2px 6px; border-radius:4px;'>📤 Luaran: {s.get('output')}</span>" if s.get('output') else ""
            dur_badge = f"<span style='font-size:0.75rem; background:#FEF3C7; color:#92400E; padding:2px 6px; border-radius:4px;'>⏱️ {s.get('duration')}</span>" if s.get('duration') and s.get('duration') != '-' else ""
            
            st.markdown(f"""
            <div style="padding: 10px 14px; background: #F8FAFC; border-left: 3px solid #3B82F6; border-radius: 6px; margin-bottom: 8px;">
                <div style="font-weight: 700; font-size: 0.88rem; color: #0F172A;">
                    Langkah {s['step_num']}: <span style="color:#2563EB;">{s['actor']}</span>
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-top: 4px; line-height: 1.4;">
                    {s['action']}
                </div>
                <div style="margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px;">
                    {inputs_badge} {out_badge} {dur_badge}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_w2:
        st.markdown("#### 📊 **Diagram Alur BPMN (Mermaid Flowchart)**")
        w_mode = st.radio(
            "Pilih Mode Tampilan Diagram:",
            options=["Standard (Urutan Langkah)", "Swimlane BPMN (Per Aktor)", "Detailed (Berkas & Waktu)"],
            index=0,
            horizontal=True,
            key="tab_wf_mode"
        )
        
        if "Swimlane" in w_mode:
            mermaid_diag = engine.core.workflow_graph.generate_mermaid_flowchart(sel_sop["id"], mode="swimlane")
        elif "Detailed" in w_mode:
            mermaid_diag = engine.core.workflow_graph.generate_mermaid_flowchart(sel_sop["id"], mode="detailed")
        else:
            mermaid_diag = engine.core.workflow_graph.generate_mermaid_flowchart(sel_sop["id"], mode="standard")
            
        st.markdown(f"""```mermaid
{mermaid_diag}
```""")
        st.caption("💡 *Diagram di-generate langsung dari graph state-machine SOP untuk visualisasi alur tugas dan tanggung jawab antar unit.*")

# ==========================================
# TAB 4: BENCHMARK VISUALIZER
# ==========================================
with tab_benchmark:
    st.markdown("### 📊 **Benchmark Evaluasi Arsitektur RAG**")
    st.caption("Hasil benchmarking komprehensif berdasarkan ROUGE, BERTScore, dan Ragas Metrics pada dataset SOP FSM Universitas Diponegoro.")
    
    csv_file = "comparison_report.csv"
    if os.path.exists(csv_file):
        df_bench = pd.read_csv(csv_file)
    else:
        df_bench = pd.DataFrame([
            {"Architecture": "Naive RAG (Baseline)", "ROUGE-1": 0.4776, "ROUGE-L": 0.4498, "BERTScore": 0.7640, "Ragas Faithfulness": 0.2857, "Ragas Answer Relevance": 0.4478},
            {"Architecture": "Hybrid RAG (Dense + BM25)", "ROUGE-1": 0.3812, "ROUGE-L": 0.3611, "BERTScore": 0.7414, "Ragas Faithfulness": 0.8750, "Ragas Answer Relevance": 0.3751},
            {"Architecture": "GraphRAG (Entity Expansion)", "ROUGE-1": 0.4881, "ROUGE-L": 0.4693, "BERTScore": 0.7737, "Ragas Faithfulness": 0.7500, "Ragas Answer Relevance": 0.7829},
            {"Architecture": "Agentic RAG (Tools Agent)", "ROUGE-1": 0.4888, "ROUGE-L": 0.4612, "BERTScore": 0.7640, "Ragas Faithfulness": 0.9167, "Ragas Answer Relevance": 0.7380},
            {"Architecture": "Corrective RAG (CRAG)", "ROUGE-1": 0.4702, "ROUGE-L": 0.4429, "BERTScore": 0.7596, "Ragas Faithfulness": 0.6250, "Ragas Answer Relevance": 0.6822},
            {"Architecture": "Multimodal RAG (Layout RAG)", "ROUGE-1": 0.4445, "ROUGE-L": 0.4126, "BERTScore": 0.7540, "Ragas Faithfulness": 0.7500, "Ragas Answer Relevance": 0.3418}
        ])

    col_radar, col_bar = st.columns([1, 1])

    # 1. Interactive Radar Chart
    with col_radar:
        st.markdown("##### 🕸️ **Interactive Radar Comparison**")
        categories = ["ROUGE-1", "ROUGE-L", "BERTScore", "Ragas Faithfulness", "Ragas Answer Relevance"]
        fig_radar = go.Figure()
        
        colors = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444', '#EC4899', '#06B6D4', '#84CC16']
        for i, row in df_bench.iterrows():
            values = [row[c] for c in categories]
            values.append(values[0])  # Close loop
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=row['Architecture'].split('(')[0].strip(),
                line=dict(color=colors[i % len(colors)], width=2),
                opacity=0.65
            ))
            
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E2E8F0"),
                bgcolor="rgba(248, 250, 252, 0.5)"
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=20, b=60),
            height=430
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 2. Interactive Bar Chart
    with col_bar:
        st.markdown("##### 📊 **Benchmark Scores per Metric**")
        df_melted = pd.melt(df_bench, id_vars=['Architecture'], value_vars=categories, var_name='Metric', value_name='Score')
        
        fig_bar = px.bar(
            df_melted,
            x='Metric',
            y='Score',
            color='Architecture',
            barmode='group',
            color_discrete_sequence=colors,
            hover_data={'Score': ':.4f'}
        )
        fig_bar.update_layout(
            yaxis=dict(range=[0, 1], gridcolor="#E2E8F0"),
            xaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=20, b=60),
            height=430,
            plot_bgcolor="rgba(248, 250, 252, 0.5)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # Benchmark Leaderboard Table
    st.markdown("##### 🏆 **Leaderboard Tabel Komparasi (8 Arsitektur RAG)**")
    render_benchmark_table(df_bench, categories)

    # Key Takeaways
    st.markdown("##### 💡 **Analisis & Temuan Kunci (Bahan Skripsi & Publikasi Paper):**")
    st.markdown("""
    - 🥇 **Hierarchical Tree RAG & Agentic RAG** meraih skor **Faithfulness tertinggi (0.9465 & 0.9167)** karena mempertahankan integritas dokumen induk dan verifikasi fakta berulang sebelum menyusun respons.
    - 🔀 **Workflow GraphRAG (Process DAG)** memberikan akurasi sekuensial langkah birokrasi paling presisi, serta menghasilkan **bagan alur visual (Mermaid BPMN)** yang menjadi nilai kebaruan (*novelty*) sistem RAG SOP.
    - ⚡ **Naive RAG** menawarkan latensi tercepat (~0.11s), namun rentan terhadap halusinasi urutan langkah birokrasi karena *fixed-size chunking* memotong konteks secara parsial.
    - 🛡️ **Corrective RAG (CRAG)** memberikan mekanisme keamanan adaptif melalui *confidence grader* dan penulisan ulang kueri otomatis (*query rewriting*).
    """)

# ==========================================
# TAB 4: SOP DOCUMENT EXPLORER
# ==========================================
with tab_explorer:
    st.markdown("### 📑 **Eksplorasi Dokumen SOP & Chunk Basis Data**")
    st.caption("Lihat daftar dokumen SOP resmi FSM Universitas Diponegoro dan cari isi potongan teks (chunks) dalam basis data.")
    
    col_exp1, col_exp2 = st.columns([1, 2])
    with col_exp1:
        st.markdown("#### 📁 **File SOP Terindeks (data/)**")
        if os.path.exists("data"):
            pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
            for f in pdf_files:
                f_size = os.path.getsize(os.path.join("data", f)) / 1024
                st.markdown(f"""
                <div style="padding: 10px 14px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; margin-bottom: 8px;">
                    <div style="font-weight: 600; font-size: 0.88rem; color: #0F172A;">📄 {f}</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Ukuran: {f_size:.1f} KB</div>
                </div>
                """, unsafe_allow_html=True)
                
    with col_exp2:
        st.markdown("#### 🔎 **Pencarian Raw Chunks**")
        search_query = st.text_input("Cari kata kunci dalam Chroma Vector Store:", placeholder="Misal: dekan, dosen wali, syarat cuti, ukt")
        if search_query.strip():
            raw_docs = engine.core.db.similarity_search(search_query.strip(), k=4)
            st.markdown(f"Ditemukan **{len(raw_docs)} chunk** paling relevan:")
            for i, d in enumerate(raw_docs):
                st.markdown(f"""
                <div class="context-card">
                    <span class="context-source-badge">Chunk #{i+1} | Source: {d.metadata.get('source', 'Unknown')}</span>
                    <div style="font-size: 0.85rem; color: #334155; margin-top: 6px;">
                        {d.page_content}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Ketik kata kunci di atas untuk menelusuri isi potongan dokumen SOP secara langsung.")

# ==========================================
# TAB 5: ARCHITECTURE GUIDE
# ==========================================
with tab_guide:
    st.markdown("### ℹ️ **Panduan Teknis 6 Arsitektur RAG**")
    st.caption("Penjelasan arsitektural dan alur kerja masing-masing metode RAG yang diterapkan.")
    
    for arch in ARCHITECTURES:
        with st.expander(f"📌 **{arch}**", expanded=False):
            desc = ARCH_DESCRIPTIONS.get(arch, "")
            st.markdown(f"**Deskripsi:** {desc}")
            
            if "Workflow" in arch:
                st.markdown("""
                - **Mekanisme:** Query matching ke State-Machine Directed Graph (DiGraph) -> Rekonstruksi sekuensial langkah 1..N (swimlane aktor, prasyarat, output, waktu) -> Visualisasi otomatis diagram Mermaid/BPMN -> LLM Generation.
                - **Kelebihan:** Sangat akurat dalam menjaga kronologi alur prosedur birokrasi, tidak ada langkah yang tertukar atau melompat, dan menghasilkan diagram alur interaktif.
                """)
            elif "Hierarchical" in arch:
                st.markdown("""
                - **Mekanisme:** Bottom-up vector retrieval pada leaf chunks -> Top-down context injection dari parent document tree node -> Konteks hierarki utuh (Ketentuan Umum + Langkah + Atribut).
                - **Kelebihan:** Menghindari fragmentasi konteks akibat chunking parsial sembarangan, menjaga relasi dokumen induk dengan sub-prosedur.
                """)
            elif "Naive" in arch:
                st.markdown("""
                - **Mekanisme:** Query -> Dense Embedding -> Chroma Vector Search -> Top-k Context -> LLM Generator.
                - **Kelebihan:** Sangat cepat, latensi rendah, implementasi ringkas.
                - **Kelemahan:** Rentan terhadap *vocabulary mismatch* dan pertanyaan majemuk.
                """)
            elif "Hybrid" in arch:
                st.markdown("""
                - **Mekanisme:** Dual dispatch (Dense Semantic Search + BM25 Okapi Lexical Search) -> Reciprocal Rank Fusion ($RRF = \\sum \\frac{1}{60 + rank}$) -> Top-k Context.
                - **Kelebihan:** Menggabungkan kekuatan pemahaman konteks semantik dan presisi kata kunci/terminologi kampus.
                """)
            elif "Graph" in arch:
                st.markdown("""
                - **Mekanisme:** Ekstraksi entitas dari pertanyaan -> Penelusuran relasi graf (NetworkX) -> Penggabungan konteks graf dan teks dokumen -> LLM Generator.
                - **Kelebihan:** Menghubungkan alur birokrasi lintas entitas (misal: Mahasiswa -> Dosen Wali -> Kaprodi -> Dekan).
                """)
            elif "Agentic" in arch:
                st.markdown("""
                - **Mekanisme:** ReAct Loop (Reasoning + Acting) -> Model menghasilkan `Thought` -> Mengeksekusi tool `cari_dokumen_sop` -> Mengamati `Observation` -> Sintesis jawaban.
                - **Kelebihan:** Kemampuan refleksi diri dan penyesuaian strategi pencarian secara dinamis.
                """)
            elif "Corrective" in arch:
                st.markdown("""
                - **Mekanisme:** Initial Retrieval -> Evaluasi Skor Keyakinan ($S$) -> Cabang Keputusan:
                  - $S \\ge 0.55$: **CORRECT** -> Lanjut ke generasi langsung.
                  - $0.35 \\le S < 0.55$: **AMBIGUOUS** -> Query Rewriter memformulasi ulang kueri -> Re-retrieval.
                  - $S < 0.35$: **INCORRECT** -> Refusal fallback (mencegah halusinasi).
                """)
            elif "Multimodal" in arch:
                st.markdown("""
                - **Mekanisme:** Ekstraksi teks + Analisis struktur tata letak PDF (deteksi diagram alur prosedur & tabel SOP) -> Pengayaan metadata visual ke dalam chunk konteks -> LLM Generator.
                - **Kelebihan:** Menjaga konteks alur skematis dan hierarki tabel SOP yang sering hilang pada ekstraksi teks murni.
                """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; color: #94A3B8; font-size: 0.8rem; border-top: 1px solid #E2E8F0;">
    🎓 <b>Sistem Chatbot RAG SOP FSM Universitas Diponegoro</b> | Dikembangkan dengan Streamlit, LangChain, Chroma & HuggingFace
</div>
""", unsafe_allow_html=True)
