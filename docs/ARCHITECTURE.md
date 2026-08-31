# 🏛️ Technical Architecture & Design Document (RAG Comparison)

Dokumen ini menjelaskan secara mendalam desain teknis, diagram alur data (*dataflow*), algoritma retrieval, dan mekanisme inferensi dari **6 Arsitektur Retrieval-Augmented Generation (RAG)** yang diimplementasikan dalam project ini.

---

## 📐 High-Level System Overview

Sistem Chatbot RAG ini dirancang untuk menjawab pertanyaan seputar Standar Operasional Prosedur (SOP) resmi Fakultas Sains dan Matematika Universitas Diponegoro (FSM UNDIP). Arsitektur terdiri dari 4 lapisan utama:

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data & Preprocessing Layer"]
        PDF["Dokumen SOP FSM UNDIP (PDF)"] --> Parser["PyPDF Directory Loader & Text Splitter"]
        Parser --> Chunks["Document Chunks (chunk_size=1700, overlap=100)"]
        Chunks --> FineTune["Indonesian Fine-Tuned Embedding Model"]
        FineTune --> ChromaDB[("Chroma Vector Database")]
        Chunks --> BM25Corpus["BM25 Inverted Index"]
        Chunks --> GraphBuilder["NetworkX Knowledge Graph"]
        PDF --> LayoutExtractor["Multimodal Layout Extractor"]
    end

    subgraph User_Layer ["2. Interaction Layer"]
        UserQuery(["Pertanyaan Pengguna"]) --> StreamlitApp["Streamlit Web UI / CLI / Evaluator"]
        StreamlitApp --> Engine["RAGEngine (src/engine.py)"]
    end

    subgraph Architecture_Layer ["3. RAG Architectures"]
        Engine --> ARCH1["1. Naive RAG (Baseline)"]
        Engine --> ARCH2["2. Hybrid RAG (Dense + BM25)"]
        Engine --> ARCH3["3. GraphRAG (Entity Expansion)"]
        Engine --> ARCH4["4. Agentic RAG (ReAct Tool Agent)"]
        Engine --> ARCH5["5. Corrective RAG (CRAG)"]
        Engine --> ARCH6["6. Multimodal RAG (Layout Enriched)"]
    end

    subgraph Generation_Layer ["4. Inference & Generation Layer"]
        ARCH1 & ARCH2 & ARCH3 & ARCH4 & ARCH5 & ARCH6 --> ContextFusion["Context Assembly & Prompt Template"]
        ContextFusion --> LLM["Ollama LLM (llama3.1:8b) / Smart Synthesis Engine"]
        LLM --> FinalAnswer["Jawaban Asisten Mahasiswa + Metrik Live"]
    end
```

---

## 🔬 Rincian Teknis 6 Arsitektur RAG

### 1. Naive RAG (Baseline)
* **File Sumber**: `src/architectures/naive_rag.py` | `RAGEngine.execute_naive_rag`

#### Mekanisme Kerja:
1. **Embedding Query**: Kueri teks diubah menjadi vektor representasi 384-dimensi menggunakan model lokal `indo_finetuned_embedding`.
2. **Dense Similarity Search**: Menghitung *cosine similarity* terhadap seluruh dokumen di Chroma DB untuk mengambil top-$k$ chunk ($k=3$).
3. **Threshold Check**: Jika skor kemiripan tertinggi $< 0.30$, sistem secara otomatis menolak menjawab (*fallback refusal*) untuk mencegah halusinasi.
4. **LLM Generation**: Menyusun *prompt context* dari chunk yang lolos dan menghasilkan respons melalui `llama3.1:8b`.

```mermaid
flowchart LR
    Q[Query] --> Emb[Embedding]
    Emb --> Chroma[(Chroma DB)]
    Chroma --> Filter{Score >= 0.3?}
    Filter -- Ya --> Prompt[Format Prompt]
    Filter -- Tidak --> Refusal[Honest Refusal]
    Prompt --> LLM[Llama 3.1:8b] --> Answer[Output]
```

---

### 2. Hybrid RAG (Dense + Sparse BM25 + Reciprocal Rank Fusion)
* **File Sumber**: `src/architectures/hybrid_rag.py` | `RAGEngine.execute_hybrid_rag`

#### Mekanisme Kerja:
Mengatasi keterbatasan pencarian semantik murni pada kueri yang membutuhkan pencocokan kata kunci eksak (misalnya nomor SOP, singkatan UKT/IRS, atau istilah spesifik).

1. **Dual Parallel Retrieval**:
   - **Dense Path**: Pencarian vektor semantik via Chroma DB ($k=5$).
   - **Sparse Path**: Pencarian berbasis leksikal menggunakan algoritma BM25Okapi ($k=5$).
2. **Reciprocal Rank Fusion (RRF)**:
   Menggabungkan kedua daftar peringkat dokumen menggunakan formula RRF:
   $$RRF\_Score(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{k_{rrf} + \text{rank}_m(d)} \quad (\text{dengan } k_{rrf} = 60)$$
3. **Scoring & Context Assembly**: Mengambil top-$3$ chunk hasil peringkat fusi RRF dan mengurutkannya berdasarkan kemiripan kosinus sebelum dikirim ke LLM.

```mermaid
flowchart TD
    Q[Query Pengguna] --> Dense[Dense Vector Search k=5]
    Q --> Sparse[BM25 Lexical Search k=5]
    Dense --> RRF["Reciprocal Rank Fusion (RRF Constant k=60)"]
    Sparse --> RRF
    RRF --> TopK[Top-3 Fused Chunks]
    TopK --> LLM[Llama 3.1:8b] --> Answer[Output Jawaban]
```

---

### 3. GraphRAG (Knowledge Graph Entity Expansion)
* **File Sumber**: `src/architectures/graph_rag.py` | `RAGEngine.execute_graph_rag`

#### Mekanisme Kerja:
Memanfaatkan struktur grafik pengetahuan (*Knowledge Graph*) berbasis **NetworkX** yang memetakan entitas kampus dan hubungan antar pihak/prosedur SOP.

1. **Entity Linking**: Mendeteksi node entitas dalam kueri (misal: *cuti akademik*, *legalisir*, *beasiswa*, *IRS*, *UKT*).
2. **Graph Traversal & Query Expansion**: Menelusuri tetangga (*1-hop neighbors*) pada graf relasi. Kueri asli diperluas dengan istilah relasional terkait (contoh: kueri *cuti akademik* diperluas dengan *izin cuti, aktif kembali, dekan, ketua program studi*).
3. **Graph Context Injection**: Struktur relasi yang ditemukan diformat sebagai *header context* eksplisit untuk membimbing pemahaman relasional LLM.

```mermaid
flowchart LR
    Q[Query Pengguna] --> EntMatch[Entity Extraction]
    EntMatch --> Graph[(NetworkX SOP Graph)]
    Graph --> Expand[Expanded Query + Relations]
    Expand --> Chroma[(Chroma DB Search)]
    Chroma --> Fusion[Graph Header + Document Chunks]
    Fusion --> LLM[Llama 3.1:8b] --> Answer[Jawaban Komprehensif]
```

---

### 4. Agentic RAG (LangChain ReAct Tools Agent)
* **File Sumber**: `src/architectures/agentic_rag.py` | `RAGEngine.execute_agentic_rag`

#### Mekanisme Kerja:
Menggunakan paradigma **ReAct (Reasoning + Acting)** yang memungkinkan model LLM berpikir secara otonom untuk menentukan kapan dan bagaimana memanggil alat retrieval.

1. **Reasoning (Thought)**: Model menganalisis kebutuhan kueri pengguna.
2. **Action**: Mengeksekusi tool `cari_dokumen_sop(query)`.
3. **Observation**: Mengamati hasil dokumen yang diperoleh dan mengevaluasi kecukupan informasinya.
4. **Synthesis (Final Answer)**: Menyusun jawaban terverifikasi berdasarkan observasi aktual.

```mermaid
flowchart TD
    Input[Query Pengguna] --> Agent["LangChain ReAct Agent"]
    Agent --> Thought1["Thought: Memerlukan pencarian SOP"]
    Thought1 --> Action["Action: cari_dokumen_sop(query)"]
    Action --> ToolExec["Eksekusi Vector Search Chroma"]
    ToolExec --> Obs["Observation: Cuplikan Dokumen SOP Ditemukan"]
    Obs --> Thought2["Thought: Informasi terverifikasi"]
    Thought2 --> FinalOutput["Final Output Response"]
```

---

### 5. Corrective RAG / CRAG (Self-Correction & Query Rewriting)
* **File Sumber**: `src/architectures/corrective_rag.py` | `RAGEngine.execute_crag`

#### Mekanisme Kerja:
Menambahkan layer evaluator (*grader*) untuk menilai relevansi retrieval awal dan secara adaptif menentukan langkah koreksi:

* **Threshold Batas**:
  - Upper Threshold = $0.55$
  - Lower Threshold = $0.35$

* **Alur Keputusan**:
  1. **`CORRECT`** ($\text{Score} \ge 0.55$): Konteks sangat relevan, langsung dilanjutkan ke proses generasi.
  2. **`AMBIGUOUS`** ($0.35 \le \text{Score} < 0.55$): Konteks kurang meyakinkan; sistem mengaktifkan modul **Query Rewriter** untuk mereformulasi kueri dengan kata kunci yang lebih terarah, lalu melakukan pencarian ulang (*secondary retrieval*).
  3. **`INCORRECT`** ($\text{Score} < 0.35$): Dokumen tidak relevan sama sekali; sistem memicu *honest refusal* untuk menghentikan halusinasi.

```mermaid
flowchart TD
    Q[Query Asli] --> Search1[Pencarian Tahap 1]
    Search1 --> Grader{"Confidence Grader"}
    Grader -- ">= 0.55 (CORRECT)" --> LLM[Generasi Jawaban Langsung]
    Grader -- "0.35 - 0.55 (AMBIGUOUS)" --> Rewriter["Query Rewriter (Reformulasi)"]
    Rewriter --> Search2[Pencarian Ulang Tahap 2]
    Search2 --> LLM
    Grader -- "< 0.35 (INCORRECT)" --> Refusal["Fallback Honest Refusal"]
```

---

### 6. Multimodal RAG (PDF Layout & Figure Descriptor Enrichment)
* **File Sumber**: `src/architectures/multimodal_rag.py` | `RAGEngine.execute_multimodal_rag`

#### Mekanisme Kerja:
SOP universitas sering kali memuat diagram alur (*flowcharts*), tabel alur kerja, dan bagan persetujuan. Multimodal RAG mengekstrak representasi struktural tersebut:

1. **Layout & Visual Indexing**: Selama inisialisasi, dokumen PDF dipindai untuk mendeteksi jumlah diagram alur, gambar kerja, dan representasi tabel data per halaman.
2. **Context Enrichment**: Setiap teks chunk yang diretrieve diinjeksi dengan metadata layout visual dari halaman aslinya (misal: `[INFORMASI VISUAL & LAYOUT: Mengandung 2 diagram alur kerja dan tabel persetujuan]`).
3. **Grounded Generation**: LLM memanfaatkan sinyal tata letak visual bersama teks untuk menghasilkan jawaban dengan pemahaman urutan alur SOP.

---

## 📊 Metrik Evaluasi Kuantitatif & Kualitatif

Pengujian dilakukan menggunakan dataset sintetis ground-truth SOP FSM UNDIP dengan kombinasi metrik:

| Kategori | Metrik | Deskripsi |
| :--- | :--- | :--- |
| **Token Overlap** | **ROUGE-1** | Overlap unigram antara jawaban model dan ground truth. |
| **Token Overlap** | **ROUGE-L** | Longest Common Subsequence (LCS) untuk konsistensi struktur kalimat. |
| **Semantic Similarity** | **BERTScore (F1)** | Kemiripan representasi semantik token-level berbasis transformer. |
| **RAG Evaluation** | **Ragas Faithfulness** | Proporsi klaim faktual dalam jawaban yang didukung oleh konteks retrieval (meminimalisir halusinasi). |
| **RAG Evaluation** | **Ragas Answer Relevance** | Tingkat kesesuaian dan kelengkapan jawaban terhadap kueri pengguna. |
