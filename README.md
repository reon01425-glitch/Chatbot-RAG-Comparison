# 🎓 Chatbot RAG – Benchmark 5 Advanced RAG Architectures (SOP FSM UNDIP)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C.svg?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20(Llama--3.1--8B)-black.svg?logo=ollama&logoColor=white)](https://ollama.com/)
[![Chroma DB](https://img.shields.io/badge/VectorDB-Chroma-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Project ini merupakan implementasi dan evaluasi komparatif dari **5 Arsitektur Advanced RAG** beserta **Baseline Naive RAG** untuk menjawab pertanyaan seputar Standar Operasional Prosedur (SOP) resmi Fakultas Sains dan Matematika Universitas Diponegoro (FSM UNDIP).

Sistem ini membandingkan kinerja berbagai metode *Retrieval-Augmented Generation* menggunakan metrik kuantitatif NLP (**ROUGE-1, ROUGE-L, BERTScore**) serta evaluasi berbasis RAG (**Ragas Faithfulness** & **Ragas Answer Relevance**), lengkap dengan antarmuka web interaktif berbasis **Streamlit**.

---

## 🎯 About The Project

Chatbot RAG ini membaca dokumen resmi kampus (SOP format PDF pada folder `data/`), melakukan ekstraksi konteks dan embedding berbasis model lokal berbahasa Indonesia, serta memproses kueri pengguna menggunakan 6 variasi arsitektur RAG yang dapat dibandingkan secara *side-by-side*.

```
                ┌───────────────────────────────────────────────────────────┐
                │             Kueri Mahasiswa seputar SOP Kampus             │
                └─────────────────────────────┬─────────────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │      Pilihan Arsitektur RAG (src/architectures/)  │
                    └─────────────────────────┬─────────────────────────┘
         ┌───────────────┬────────────────────┼───────────────────┬───────────────┐
         │               │                    │                   │               │
         ▼               ▼                    ▼                   ▼               ▼
   [ Naive RAG ]  [ Hybrid RAG ]       [ GraphRAG ]        [ Agentic RAG ]     [ CRAG ]
    Dense Vector   Dense + BM25     Knowledge Graph Ent.   ReAct Tool Agent   Self-Correction
    (Chroma DB)    (RRF Ranking)    (NetworkX Relations)  (cari_dokumen_sop) (Query Rewrite)
         │               │                    │                   │               │
         └───────────────┴────────────────────┼───────────────────┴───────────────┘
                                              │
                                              ▼
                    ┌───────────────────────────────────────────────────┐
                    │      Ekstraksi Layout & Figur (Multimodal RAG)    │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                    ┌───────────────────────────────────────────────────┐
                    │      Sintesis Jawaban LLM Lokal (Llama 3.1 8B)    │
                    └───────────────────────────────────────────────────┘
```

---

## 🏛️ 6 Arsitektur RAG yang Diuji

| No | Arsitektur | File Sumber | Deskripsi Singkat |
|:---:|:---|:---|:---|
| 1 | **Naive RAG (Baseline)** | `src/architectures/naive_rag.py` | Pendekatan standar pencarian vektor kosinus tunggal pada Chroma DB. |
| 2 | **Hybrid RAG (Dense + BM25)** | `src/architectures/hybrid_rag.py` | Penggabungan pencarian semantik (Chroma) dan leksikal (BM25) dengan algoritma **Reciprocal Rank Fusion (RRF)**. |
| 3 | **GraphRAG (Entity Expansion)** | `src/architectures/graph_rag.py` | Ekspansi kueri semantik berbasis grafik relasi entitas kampus yang dibangun menggunakan **NetworkX**. |
| 4 | **Agentic RAG (Tools Agent)** | `src/architectures/agentic_rag.py` | Agent pintar berbasis **LangChain ReAct** yang secara dinamis memanggil alat pencarian dokumen (`cari_dokumen_sop`). |
| 5 | **Corrective RAG (CRAG)** | `src/architectures/corrective_rag.py` | Alur *self-correction* dengan grader skor kemiripan dan *query rewriter* otomatis untuk kueri ambigu. |
| 6 | **Multimodal RAG (Layout RAG)** | `src/architectures/multimodal_rag.py` | Ekstraksi deskriptor tata letak visual PDF (tabel data, diagram alur, gambar kerja) untuk memperkaya konteks. |

> 📖 *Dokumentasi teknis lengkap dan alur diagram per arsitektur dapat dibaca di [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).*

---

## 📊 Hasil Evaluasi Komparatif Benchmark

Evaluasi dijalankan menggunakan LLM lokal **`llama3.1:8b`** via **Ollama** dengan dataset uji SOP FSM UNDIP:

| Arsitektur RAG | ROUGE-1 | ROUGE-L | BERTScore | Ragas Faithfulness | Ragas Answer Relevance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Agentic RAG (Tools Agent)** | **0.4888** | 0.4612 | 0.7640 | **0.9167** | 0.7380 |
| 🥈 **GraphRAG (Entity Expansion)** | 0.4881 | **0.4693** | **0.7737** | 0.7500 | **0.7829** |
| 🥉 **Corrective RAG (CRAG)** | 0.4702 | 0.4429 | 0.7596 | 0.6250 | 0.6822 |
| 🏅 **Hybrid RAG (Dense + BM25)** | 0.3812 | 0.3611 | 0.7414 | 0.8750 | 0.3751 |
| 🏅 **Multimodal RAG (Layout)** | 0.4445 | 0.4126 | 0.7540 | 0.7500 | 0.3418 |
| ⚠️ **Naive RAG (Baseline)** | 0.4776 | 0.4498 | 0.7640 | 0.2857 | 0.4478 |

### 🔍 Analisis Temuan Utama:
- **Akurasi Fakta Tertinggi (Faithfulness)**: **Agentic RAG (0.9167)**  
  Penggunaan *ReAct tool agent* memungkinkan LLM melakukan verifikasi fakta berulang ke basis data dokumen sebelum menyusun jawaban, meminimalisir halusinasi.
- **Relevansi Jawaban Tertinggi**: **GraphRAG (0.7829)**  
  Ekstensi entitas grafik membantu LLM memahami relasi hierarki kampus (misal: keterkaitan cuti akademik dengan persetujuan ketua prodi dan dekan).
- **Risiko Halusinasi Naive RAG**:  
  Skor **Faithfulness Naive RAG paling rendah (0.2857)**, menunjukkan risiko tinggi menghasilkan informasi fiktif tanpa mekanisme verifikasi atau re-ranking.

---

## 🚀 Panduan Setup di Laptop Baru (Step-by-Step)

Ikuti panduan ini untuk mengklon, menginstall, dan menjalankan project secara langsung di laptop baru:

### 📋 1. Prasyarat Sistem
- **Sistem Operasi**: Windows 10/11, macOS, atau Linux
- **Python**: Versi `3.11` atau `3.12` (disarankan)
- **Git**: Terinstall di komputer
- **Ollama**: Terinstall untuk menjalankan LLM lokal ([Download Ollama](https://ollama.com))

---

### 📥 2. Clone Repository
Buka Terminal / Command Prompt / PowerShell:

```bash
git clone https://github.com/reon01425-glitch/Chatbot-RAG-Comparison.git
cd Chatbot-RAG-Comparison
```

---

### 🐍 3. Buat & Aktifkan Virtual Environment

#### Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(Jika muncul error execution policy di PowerShell, jalankan: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` lalu aktifkan kembali).*

#### Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 📦 4. Install Dependencies
Pastikan virtual environment telah aktif, lalu jalankan:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 🦙 5. Setup LLM Lokal (Ollama)
1. Jalankan service Ollama di background (atau buka aplikasi desktop Ollama):
   ```bash
   ollama serve
   ```
2. Buka terminal baru dan unduh model **Llama 3.1 (8B)**:
   ```bash
   ollama pull llama3.1:8b
   ```

---

### ⚙️ 6. Setup Environment File (`.env`)
Salin file konfigurasi contoh:

```bash
# Windows PowerShell:
Copy-Item .env.example .env

# Linux / macOS:
cp .env.example .env
```

*(Opsional)* Anda dapat menambahkan `GITHUB_TOKEN` atau `GOOGLE_API_KEY` di dalam `.env` jika diperlukan.

---

### 🧠 7. Fine-tuning Embeddings & Membangun Vector Database
Jalankan script pembuatan model embedding lokal dan ekstraksi dokumen PDF ke Chroma DB:

```bash
# 1. Fine-tuning model embedding bahasa Indonesia (menghasilkan ./indo_finetuned_embedding/)
python finetune_embeddings.py

# 2. Ekstrak seluruh PDF di folder data/ dan bangun basis data vektor Chroma
python embeddings.py
```

---

### 💻 8. Menjalankan Aplikasi

#### 🌐 A. Menjalankan Interactive Web Dashboard (Streamlit UI):
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:8501`. Dashboard ini menyediakan:
- Chatbot interaktif dengan pilihan 6 arsitektur RAG.
- Live *Execution Trace* (ReAct step, similarity score, latency, and context inspection).
- Visualisasi radar chart & perbandingan metrik *side-by-side*.

#### 🔍 B. Menguji Kueri via CLI:
```bash
# Menguji Naive RAG Baseline:
python src/architectures/naive_rag.py

# Menguji Hybrid RAG:
python src/architectures/hybrid_rag.py

# Menguji Agentic RAG:
python src/architectures/agentic_rag.py

# Query custom via CLI:
python query_data.py "Bagaimana prosedur pengajuan cuti akademik?"
```

#### 📈 C. Menjalankan Benchmark Evaluasi Lengkap:
```bash
python evaluate_all.py
```
Hasil evaluasi komparatif akan ditampilkan di konsol dan disimpan otomatis ke dalam file `comparison_report.csv`.

---

## 📂 Struktur Direktori Project

```
Chatbot-RAG-Comparison/
├── data/                       # Dokumen PDF resmi SOP FSM UNDIP
│   ├── SOP_Izin_Cuti_Akademik.pdf
│   ├── SOP_Legalisir_Ijazah_Dan_Transkrip.pdf
│   ├── SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf
│   ├── SOP_Pengajuan_Rekomendasi_Beasiswa.pdf
│   ├── SOP_Pengisian_IRS.pdf
│   ├── SOP_Permohonan_Izin_Aktif_Setelah_Cuti.pdf
│   └── SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf
├── datasets/                   # Dataset synthetic QA ground truth untuk evaluasi
├── docs/
│   └── ARCHITECTURE.md         # Dokumentasi detail teknis & diagram arsitektur
├── assets/                     # Aset visual & logo
│   └── undip_logo.png
├── chroma/                     # Basis data vektor Chroma (persist)
├── indo_finetuned_embedding/   # Model embedding hasil fine-tuning lokal
├── src/
│   ├── __init__.py
│   ├── engine.py               # RAG Core Engine & Unified Execution Manager
│   └── architectures/
│       ├── __init__.py
│       ├── naive_rag.py        # 1. Baseline Naive RAG
│       ├── hybrid_rag.py       # 2. Dense + BM25 Sparse Search + RRF
│       ├── graph_rag.py        # 3. Entity Graph Expansion (NetworkX)
│       ├── agentic_rag.py      # 4. LangChain Tools Agent (ReAct)
│       ├── corrective_rag.py   # 5. Corrective RAG (CRAG)
│       └── multimodal_rag.py   # 6. PDF Layout & Figure Descriptor RAG
│
├── app.py                      # Interactive Streamlit Web UI Dashboard
├── finetune_embeddings.py     # Script fine-tuning sentence-transformers
├── embeddings.py               # Script pembentukan Chroma Vector DB
├── generate_qa.py              # Generator dataset Q&A sintetis
├── query_data.py               # Entrypoint query CLI sederhana
├── evaluate_all.py             # Runner evaluasi komparatif multi-arsitektur
├── comparison_report.csv       # Laporan spreadsheet hasil benchmark
├── requirements.txt            # Daftar dependensi Python
├── .env.example                # Template konfigurasi environment
├── LICENSE                     # MIT License
└── README.md                   # Dokumentasi panduan project
```

---

## 📌 Troubleshooting & FAQ

- **`ConnectionRefusedError / Ollama server not reachable`**:  
  Pastikan Ollama sudah aktif di background dengan menjalankan perintah `ollama serve`.
- **`Model llama3.1:8b not found`**:  
  Jalankan `ollama pull llama3.1:8b` di terminal untuk mengunduh bobot model.
- **`ModuleNotFoundError: No module named 'src'`**:  
  Jalankan script dengan menyetel `PYTHONPATH`:
  ```bash
  # PowerShell Windows:
  $env:PYTHONPATH="."; python evaluate_all.py

  # Linux/macOS:
  PYTHONPATH=. python evaluate_all.py
  ```
- **Port Streamlit 8501 sudah terpakai**:  
  Jalankan Streamlit di port lain: `streamlit run app.py --server.port 8502`.

---

## 👤 Author & Contributor

**Alfonso Clement S**  
- Email: sutancs42@gmail.com  
- GitHub: [reon01425-glitch](https://github.com/reon01425-glitch) / [PIP-Bravo](https://github.com/PIP-Bravo)