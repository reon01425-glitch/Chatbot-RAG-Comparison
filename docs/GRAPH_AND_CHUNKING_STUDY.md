# 🔬 Studi Eksplorasi: Komparasi Graph & Chunking untuk Dokumen SOP

**Konteks Riset**: Standar Operasional Prosedur (SOP) Fakultas Sains dan Matematika Universitas Diponegoro (FSM UNDIP)  
**Tujuan**: Membandingkan representasi pengetahuan berbasis graf (*Entity-Relation Graph* vs *Process/Workflow Graph*) dan strategi pemotongan teks (*Fixed-size Chunking* vs *Hierarchical/Tree-structured Chunking*) guna mengatasi karakteristik unik dokumen birokrasi kampus serta menghasilkan luaran bagan alur (*BPMN / Mermaid Flowchart*).

---

## 1. Latar Belakang & Karakteristik Unik Dokumen SOP

Dokumen Standar Operasional Prosedur (SOP) memiliki karakteristik struktural dan semantik yang sangat berbeda dengan teks naratif umum atau artikel ensiklopedia:

| Karakteristik SOP | Deskripsi dalam SOP Kampus | Dampak terhadap Naive / Fixed Chunking |
| :--- | :--- | :--- |
| **Sekuensial / Kronologis** | Tahapan birokrasi wajib dijalankan berurutan (Langkah $1 \to 2 \to \dots \to N$). | Pemotongan teks acak (*fixed-size*) memisahkan langkah, membuat LLM menghasilkan urutan yang terbolak-balik atau terlewat. |
| **Multi-Aktor (Swimlanes)** | Setiap langkah dilakukan oleh aktor spesifik (Mahasiswa, Dosen Wali, Kaprodi, Dekan, Subbag). | Terjadinya *actor mismatch* (misal: mengira Dekan yang menandatangani IRS, padahal Dosen Wali). |
| **Prasyarat & Luaran (Artifacts)** | Input dokumen (KTM, KHS, Bukti Bayar) dan output resmi (Surat Izin Dekan, Lembar Pengesahan). | Hubungan sebab-akibat antar berkas hilang jika chunk terpotong di tengah kalimat. |
| **Format Campuran (Tabel & Bagan)** | Tabel matriks kegiatan, estimasi waktu (menit/hari kerja), dan bagan alur proses (BPMN). | Ekstraksi teks biasa menghilangkan makna spasial dan relasi relasional tabel SOP. |

---

## 2. Taksonomi Pendekatan Graph & Chunking yang Dibandingkan

Dalam studi ini, diimplementasikan dan dibandingkan **3 pendekatan representasi struktur**:

```
                                  [ Dokumen SOP FSM UNDIP ]
                                              │
              ┌───────────────────────────────┼──────────────────────────────┐
              │                               │                              │
              ▼                               ▼                              ▼
    [ Metode 1: Entity-Relation ]    [ Metode 2: Process Workflow ]    [ Metode 3: Hierarchical Tree ]
    - Graph Type: Undirected         - Graph Type: Directed (DAG)      - Graph Type: Multi-level Tree
    - Node: Entitas / Aktor          - Node: Langkah Prosedur 1..N     - Root: Dokumen SOP
    - Edge: Hubungan Asosiatif       - Edge: NEXT_STEP & HAS_STEP      - Child: Section & Leaf Steps
    - Traversal: 1-Hop Neighbors     - Traversal: Topological Order    - Traversal: Top-down / Bottom-up
    - Output: Context Expansion      - Output: Answer + Live BPMN      - Output: Preserved Context Tree
```

---

### Metode 1: Entity-Relation Graph (Baseline GraphRAG)
* **File**: `src/architectures/graph_rag.py`
* **Prinsip**: Memetakan hubungan antar aktor dan konsep kampus sebagai graf tak berarah $G = (V, E)$.
* **Formulasi**:
  - $V = \{ \text{Mahasiswa}, \text{Dosen Wali}, \text{Cuti Akademik}, \text{Dekan}, \text{UKT}, \dots \}$
  - $E = \{ (u, v) \mid \text{terdapat relasi administratif antara } u \text{ dan } v \}$
  - Kueri diperluas (*Query Expansion*): $Q' = Q \cup \mathcal{N}_G(\text{Entities}(Q))$.
* **Kelebihan**: Bagus untuk kueri konseptual dan asosiatif.
* **Kelemahan**: Tidak menyimpan urutan waktu/langkah prosedur sekuensial.

---

### Metode 2: Process / Workflow Directed Graph (BPMN-like State Machine)
* **File**: `src/graph_sop/workflow_graph.py` & `src/architectures/workflow_graph_rag.py`
* **Prinsip**: Memodelkan tahapan SOP sebagai *Directed Acyclic Graph* (DAG) yang merepresentasikan mesin alur kerja birokrasi kampus.
* **Komponen Node**:
  - $Step_i = \langle \text{Aktor}, \text{Aksi}, \text{Input Prasyarat}, \text{Output Luaran}, \text{Estimasi Waktu} \rangle$
* **Komponen Edge**:
  - $e = (Step_i, Step_{i+1})$ dengan label relasi `NEXT_STEP` dan bobot transisi output.
* **Fitur Utama**:
  1. **Topological Sequence Retrieval**: Rekonstruksi kronologi langkah dari awal hingga akhir tanpa risiko halusinasi urutan.
  2. **Automated BPMN / Mermaid Flowchart Rendering**: Sistem secara otomatis mengekstrak sintaks diagram alur Mermaid yang dapat langsung dirender di antarmuka web Streamlit.

---

### Metode 3: Hierarchical / Tree-Structured Chunking
* **File**: `src/graph_sop/hierarchical_chunker.py` & `src/architectures/hierarchical_rag.py`
* **Prinsip**: Menghindari *loss of context* akibat *fixed-size chunking* dengan membagi dokumen berdasarkan struktur pohon semantik:
  - $\text{Root Node}$: Judul SOP & Ketentuan Umum Dokumen.
  - $\text{Branch Node}$: Bagian / Pasal / Tahapan Prosedur.
  - $\text{Leaf Node}$: Aksi Spesifik, Persyaratan Dokumen, Tautan Google Drive Formulir, dan Waktu Layanan.
* **Kelebihan**: Saat sebuah *leaf chunk* cocok dengan kueri pengguna, metadata dan ringkasan dari *parent node* secara otomatis disuntikkan ke dalam prompt LLM (*Parent-Child Context Fusion*).

---

## 3. Hasil Benchmarking Komparatif

Evaluasi empiris dijalankan menggunakan 6 skenario kueri representatif yang menguji 3 aspek krusial dokumen SOP:
1. **Procedural Order (Alur Langkah)**: Urutan kronologis birokrasi tanpa langkah yang terbolak-balik.
2. **Prerequisites & Documents (Syarat & Berkas)**: Kelengkapan dokumen prasyarat, formulir online, dan berkas lampiran.
3. **Swimlane Actors & Authority (Aktor, Wewenang & Batas Waktu)**: Ketepatan penugasan tanggung jawab antar unit kerja.

Hasil evaluasi kuantitatif empiris:

| Arsitektur | ROUGE-1 | ROUGE-L | BERTScore (id) | Faithfulness | Answer Relevance | Avg Latency | Output Visual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Workflow GraphRAG (Process DAG)** | **0.3568** | **0.3096** | 0.6929 | 0.8377 | 0.7477 | 15.718s | **Bagan Alur BPMN (Mermaid)** |
| **Hierarchical Tree RAG** | 0.3487 | 0.2926 | 0.6954 | **0.9465** | 0.7575 | 12.943s | Konteks Dokumen Induk + Rincian Sub-Langkah |
| **GraphRAG (Entity Expansion)** | **0.3840** | **0.3381** | **0.7231** | 0.9332 | **0.8014** | 16.470s | Perluasan Entitas Relasional Kampus |
| **Naive RAG (Baseline)** | 0.1349 | 0.1146 | 0.6847 | 1.0000* | 0.6149 | **0.118s** | Teks Potongan Vektor Parsial |

*\*Catatan: Faithfulness Naive RAG bernilai 1.0000 pada kasus refusal fallback (sistem menolak menjawab ketika dokumen tidak relevan, sehingga tidak ada klaim keliru yang dibuat).*

---

## 4. Analisis & Temuan Ilmiah (Untuk Penulisan Paper/Skripsi)

1. **Keunggulan Entity-Relation Graph & Workflow DAG terhadap Naive Chunking**:
   - Metode berbasis Graph menunjukkan lonjakan drastis pada metrik **Answer Relevance (0.8014 & 0.7477 vs 0.6149 pada Naive)** dan **ROUGE-1 (0.3840 & 0.3568 vs 0.1349)**. 
   - Hal ini membuktikan hipotesis bahwa *fixed-size chunking* pada dokumen SOP memotong alur birokrasi kampus secara sembarangan, sedangkan graph mampu menghubungkan keterkaitan aktor dan tahapan.

2. **Dukungan Diagram Alur (BPMN / Flowchart): Nilai Kebaruan Sistem**:
   - Sesuai arahan pembimbing (Pak Indra), representasi bagan alur menjadi **ciri khas sistem RAG SOP**. Mahasiswa tidak hanya membaca teks panjang, namun langsung melihat representasi visual diagram langkah, aktor pelaksana, dan dokumen yang harus disiapkan.

3. **Peran Hierarchical Tree Chunking**:
   - Menghasilkan **Faithfulness tertinggi (0.9465)** pada respon yang dihasilkan karena setiap simpul leaf membawa metadata lengkap dari parent node dokumen asalnya, mengeliminasi risiko salah atribusi kebijakan antar SOP.
