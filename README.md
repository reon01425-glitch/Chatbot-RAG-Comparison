# 📚 Chatbot RAG – Benchmark 5 Advanced RAG Architectures (SOP FSM UNDIP)

Project ini merupakan implementasi dan evaluasi komparatif dari **5 Arsitektur Advanced RAG** beserta **Baseline Naive RAG** untuk menjawab pertanyaan seputar Standar Operasional Prosedur (SOP) Fakultas Sains dan Matematika Universitas Diponegoro (FSM UNDIP).

Sistem ini membandingkan kinerja berbagai metode Retrieval-Augmented Generation menggunakan metric kuantitatif NLP (**ROUGE-1, ROUGE-L, BERTScore**) serta evaluasi berbasis RAG (**Ragas Faithfulness** & **Ragas Answer Relevance**).

---

## 🎯 About The Project

Chatbot RAG ini dirancang untuk membaca dokumen resmi kampus (SOP PDF pada folder `data/`), melakukan ekstraksi konteks dan embedding, serta memproses kueri pengguna menggunakan arsitektur RAG yang disesuaikan.

### 🏛️ 6 Arsitektur RAG yang Diuji:

1. **Naive RAG (Baseline)** (`src/architectures/naive_rag.py`):
   - Pendekatan standar menggunakan pencarian vektor kosinus tunggal pada Chroma DB.
2. **Hybrid RAG (Dense + BM25)** (`src/architectures/hybrid_rag.py`):
   - Penggabungan pencarian semantik (Chroma Dense Vector) dan pencarian kata kunci (BM25 Sparse Search) menggunakan algoritma **Reciprocal Rank Fusion (RRF)**.
3. **GraphRAG (Entity Expansion)** (`src/architectures/graph_rag.py`):
   - Ekspansi kueri semantik berbasis grafik relasi entitas kampus yang dibangun dengan **NetworkX** (misal: menghubungkan *cuti akademik* $\rightarrow$ *dekan* $\rightarrow$ *ketua prodi*).
4. **Agentic RAG (Tools Agent)** (`src/architectures/agentic_rag.py`):
   - Agentik pintar berbasis **LangChain ReAct Agent** yang secara dinamis memilih dan mengeksekusi alat pencarian dokumen (`cari_dokumen_sop`).
5. **Corrective RAG / CRAG** (`src/architectures/corrective_rag.py`):
   - Alur keputusan *self-correction* dengan penilai skor kemiripan (grader) dan *query rewriter* otomatis untuk penulisan ulang kueri ambigu.
6. **Multimodal RAG (Layout RAG)** (`src/architectures/multimodal_rag.py`):
   - Ekstraksi deskriptor tata letak visual PDF (tabel data, diagram alur, gambar kerja) untuk memperkaya konteks pencarian teks.

---

## 📊 Hasil Evaluasi Komparatif Benchmark

Evaluasi dijalankan menggunakan LLM lokal **`llama3.1:8b`** via **Ollama** dengan evaluasi Ragas secara sekuensial pada dataset pertanyaan SOP FSM UNDIP.

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
  *Mengapa?* Penggunaan ReAct tools agent memungkinkan LLM secara aktif melakukan verifikasi fakta berulang ke dalam basis data dokumen, sehingga meminimalisir halusinasi.
- **Relevansi Jawaban Tertinggi**: **GraphRAG (0.7829)**  
  *Mengapa?* Ekstensi entitas grafik membantu LLM memahami hubungan struktural antar pihak kampus (misal: syarat rekomendasi cuti oleh dosen wali), sehingga jawaban lebih komprehensif.
- **Risiko Halusinasi Naive RAG**:  
  Meskipun Naive RAG memiliki nilai token overlap (ROUGE) yang baik, skor **Faithfulness-nya paling rendah (0.2857)**, menunjukkan risiko tinggi menghasilkan informasi palsu tanpa mekanisme verifikasi.

---

## 🚀 Panduan Setup dari 0 (How to Start from Scratch)

Ikuti langkah-langkah berikut untuk menginstall dan menjalankan seluruh sistem dari 0 di perangkat baru.

### 📋 Prasyarat Sistem
- **OS**: Windows / Linux / macOS
- **Python**: Versi `3.11` atau `3.12`
- **Git**: Terinstall di komputer
- **Ollama**: Terinstall untuk menjalankan LLM lokal ([https://ollama.com](https://ollama.com))

---

### Langkah 1: Clone Repository & Masuk ke Folder

Buka Terminal / Command Prompt / PowerShell:

```bash
git clone https://github.com/PIP-Bravo/Chatbot-RAG.git
cd Chatbot-RAG
```

---

### Langkah 2: Buat & Aktifkan Virtual Environment

#### Menggunakan `venv` (Standar Python):
```bash
# Membuat environment
python -m venv .venv

# Mengaktifkan di Windows PowerShell:
.venv\Scripts\Activate.ps1

# Mengaktifkan di Linux/macOS:
source .venv/bin/activate
```

#### Atau Menggunakan Conda:
```bash
conda create -n chatbot_rag python=3.12 -y
conda activate chatbot_rag
```

---

### Langkah 3: Install Dependencies

Pastikan virtual environment sudah aktif, lalu jalankan:

```bash
pip install -r requirements.txt
```

---

### Langkah 4: Setup LLM Lokal (Ollama)

1. Buka aplikasi **Ollama** atau jalankan server Ollama:
   ```bash
   ollama serve
   ```
2. Di terminal lain, unduh model **Llama 3.1 (8B)**:
   ```bash
   ollama pull llama3.1:8b
   ```

---

### Langkah 5: Setup Environment File (`.env`)

Buat file bernama `.env` di direktori utama project:

```env
GITHUB_TOKEN="your_github_personal_access_token_here"
```

---

### Langkah 6: Fine-tuning Embedding & Membangun Vector Database

1. **Jalankan Fine-tuning Embedding Bahasa Indonesia**:
   ```bash
   python finetune_embeddings.py
   ```
   *Proses ini membuat model embedding lokal pada folder `./indo_finetuned_embedding/`.*

2. **Ekstrak & Simpan Dokumen SOP ke Chroma DB**:
   ```bash
   python embeddings.py
   ```
   *Proses ini membaca seluruh PDF di folder `data/` dan menyimpannya ke folder `chroma/`.*

---

### Langkah 7: Menjalankan Pengujian Query

Anda dapat menguji individual RAG architecture:

#### Menjalankan Naive RAG:
```bash
python src/architectures/naive_rag.py
```

#### Menjalankan Hybrid RAG:
```bash
python src/architectures/hybrid_rag.py
```

#### Menjalankan Agentic RAG:
```bash
python src/architectures/agentic_rag.py
```

#### Menjalankan Query Baseline Asli:
```bash
python query_data.py "Bagaimana prosedur pengajuan cuti akademik?"
```

---

### Langkah 8: Menjalankan Benchmark Evaluasi Semua Arsitektur

Untuk menjalankan pengujian lengkap ROUGE, BERTScore, dan Ragas pada 6 arsitektur:

```bash
python evaluate_all.py
```

Hasil pengujian akan otomatis ditampilkan di terminal dan disimpan dalam file `comparison_report.csv`.

---

## 📂 Struktur Direktori Project

```
Chatbot-RAG/
├── data/                       # Dokumen PDF resmi SOP FSM UNDIP
├── datasets/                   # Dataset synthetic QA ground truth
├── chroma/                     # Basis data vektor Chroma (persist)
├── indo_finetuned_embedding/   # Model embedding hasil fine-tuning lokal
├── src/
│   ├── __init__.py
│   └── architectures/
│       ├── naive_rag.py        # 1. Baseline Naive RAG
│       ├── hybrid_rag.py       # 2. Dense + BM25 Sparse Search + RRF
│       ├── graph_rag.py        # 3. Entity Graph Expansion (NetworkX)
│       ├── agentic_rag.py      # 4. LangChain Tools Agent (ReAct)
│       ├── corrective_rag.py   # 5. Corrective RAG (CRAG)
│       └── multimodal_rag.py   # 6. PDF Layout & Figure Descriptor RAG
│
├── finetune_embeddings.py     # Script fine-tuning sentence-transformers
├── embeddings.py               # Script pembangunan Chroma Vector DB
├── generate_qa.py              # Generator dataset Q&A sintetis
├── query_data.py               # Entrypoint query CLI sederhana
├── evaluate_all.py             # Runner evaluasi komparatif multi-arsitektur
├── comparison_report.csv       # Laporan spreadsheet hasil benchmark
├── requirements.txt            # Dependency daftar package
└── README.md                   # Dokumentasi panduan project
```

---

## 🧠 Knowledge Base Dokumen SOP

Dokumen yang digunakan sebagai basis data pengetahuan meliputi:
- SOP Pengisian IRS
- SOP Permohonan Izin Aktif Kuliah Setelah Cuti
- SOP Permohonan Izin Cuti Akademik
- SOP Permohonan Izin Keterlambatan Pembayaran UKT
- SOP Legalisir Ijazah dan Transkrip
- SOP Pengajuan Beasiswa
- SOP Pengajuan Proposal Kegiatan Organisasi

*Hak Cipta Dokumen: Fakultas Sains dan Matematika, Universitas Diponegoro.*

---

## 📌 Troubleshooting & FAQ

- **`Error: ConnectionRefusedError / Ollama server not reachable`**:  
  Pastikan Ollama sudah berjalan di background dengan mengetikkan `ollama serve`.
- **`Model llama3.1:8b not found`**:  
  Jalankan `ollama pull llama3.1:8b` untuk mengunduh model.
- **`ModuleNotFoundError: No module named 'src'`**:  
  Jalankan script dengan menambahkan PYTHONPATH:
  ```bash
  # PowerShell Windows:
  $env:PYTHONPATH="."; python evaluate_all.py

  # Linux/macOS:
  PYTHONPATH=. python evaluate_all.py
  ```

---

## 👤 Author & Contributor

**Alfonso Clement S**  
- Email: sutancs42@gmail.com  
- GitHub: [https://github.com/PIP-Bravo](https://github.com/PIP-Bravo)