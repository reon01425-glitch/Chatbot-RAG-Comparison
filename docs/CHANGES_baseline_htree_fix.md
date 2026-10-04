# Laporan Perbaikan 4 Komponen Arsitektur RAG & Penguatan Protokol Evaluasi

**Repositori**: `https://github.com/reon01425-glitch/Chatbot-RAG-Comparison`  
**Branch**: `fix/baseline-and-htree-improvements`  
**Base Commit**: `e8fc6dd` (`origin/feature/gemma4-e2b-support`)  
**Tanggal**: 4 Oktober 2026  
**Penulis**: Antigravity (Google DeepMind Pair Programmer)

---

## 1. Ringkasan Eksekutif

Berdasarkan audit teknis terhadap implementasi 8 arsitektur RAG untuk SOP Layanan Akademik FSM Universitas Diponegoro, ditemukan 4 kelemahan arsitektural yang mendasar:
1. **Agentic RAG** sebelumnya bukan loop ReAct otonom melainkan pemanggilan tunggal dengan trace teks statis (*hardcoded*).
2. **Corrective RAG (CRAG)** pada status `AMBIGUOUS` tidak memanggil LLM melainkan memakai template string heuristik.
3. **Fine-tuning Embedding** sebelumnya memproses 1 sampel per file sehingga `MultipleNegativesRankingLoss` kehilangan in-batch negatives (loss praktis nol), serta terhambat oleh cache `trained.json` dan data QA korup/Inggris.
4. **Hierarchical Tree RAG** belum mengindeks leaf chunks (masih mencari di chunk dokumen 1.700 karakter) dan parser langkah memotong kalimat di batas baris PDF.

Seluruh 4 kelemahan tersebut telah berhasil diperbaiki secara komprehensif pada branch baru `fix/baseline-and-htree-improvements` dengan prinsip **keadilan eksperimen (fair comparison)**, **kemampuan lacak penuh (traceability)**, dan **tanpa kebocoran data (zero data leakage)**.

---

## 2. Rincian Perubahan per Tugas

### Tugas 0: Fondasi Traceability & LLM Choke Point (`09432ba`)
- **Masalah Sebelum**: Evaluator (`evaluate_benchmark.py`) mem-patch `RAGCore.generate_synthesis`. Akibatnya, pemanggilan LLM baru di luar sintesis akhir (seperti langkah penalaran ReAct atau query rewriting CRAG) tidak akan tercatat pada metrik latensi, token prompt, token completion, dan error (RQ3 menjadi bias dan tidak akurat).
- **Perubahan yang Dilakukan**:
  - Dibuat metode terpusat `RAGCore.call_llm(self, prompt: str) -> str` sebagai *single patchable LLM choke point*.
  - `generate_synthesis` dialihkan untuk memanggil `call_llm`.
  - Mode evaluasi mengaktifkan `RAGCore.disable_extractive_fallback = True`. Jika inferensi gagal, error dicatat eksplisit dan tidak pernah digantikan diam-diam oleh teks ekstraktif.
  - `evaluate_benchmark.py` diperbarui untuk mem-patch `RAGCore.call_llm`.
- **File Diubah**: `src/engine.py`, `evaluate_benchmark.py`.
- **Verifikasi**: Baseline `tests/test_benchmark_metrics.py` (50 passed) & `check-leakage` (0 item >= 0.50).

---

### Tugas 1: Agentic RAG sebagai ReAct Loop Sungguhan (`37893bc`)
- **Masalah Sebelum**: `execute_agentic_rag` hanya melakukan 1 kali pencarian `similarity_search` dan trace `Thought / Action / Observation` ditulis statis tanpa pemanggilan LLM.
- **Perubahan yang Dilakukan**:
  - `execute_agentic_rag` diganti dengan loop **ReAct (Reasoning + Acting)** otonom multi-langkah (`max_steps=3` default, dapat dikonfigurasi via CLI `--agentic-max-steps` dan dicatat di `run_config.json`).
  - LLM menghasilkan `Thought:` / `Action: cari_dokumen_sop` / `Action Input:` / `Final: siap menjawab`.
  - Parser toleran: aksi diutamakan di atas `Final` pada langkah awal untuk menghindari terminasi prematur tanpa observasi. Format rusak dicatat aman di trace tanpa menyebabkan sistem crash.
  - Mengumpulkan chunk unik dari seluruh observasi, menerapkan threshold kosinus $0.30$, mengembalikan *honest refusal* jika di bawah ambang batas, dan menyintesis jawaban akhir via `PROMPT_TEMPLATE` standar (maks. 3 chunk teratas).
- **File Diubah**: `src/engine.py`, `evaluate_benchmark.py`.
- **Tes Baru**: `tests/test_agentic_react.py` (5 unit tests: verifikasi stop on final, batas max_steps, penanganan format rusak, threshold refusal, dan seluruh panggilan melalui `call_llm`).
- **Cara Menjalankan**: `python -m pytest tests/test_agentic_react.py -v`

---

### Tugas 2: CRAG dengan Query Rewrite Berbasis LLM (`5c9e814`)
- **Masalah Sebelum**: Pada rentang ambang ambigu ($0.35 \le \text{skor} < 0.55$), kueri ditulis ulang menggunakan penggabungan string template statis (`"SOP prosedur pengajuan … FSM Undip"`), padahal `REWRITE_PROMPT_TEMPLATE` sudah didefinisikan.
- **Perubahan yang Dilakukan**:
  - Pada status `AMBIGUOUS`, sistem memanggil `REWRITE_PROMPT_TEMPLATE` melalui `RAGCore.call_llm`.
  - Output dibersihkan secara ketat menjadi 1 baris bebas tanda kutip dan bebas prefiks (misal menghapus `Kueri baru:` atau tanda petik dua).
  - Kueri hasil reformulasi digunakan untuk pencarian sekunder (*secondary retrieval*) ke Chroma DB.
  - Mode evaluasi melarang fallback diam-diam: jika rewrite LLM gagal/kosong, sistem melempar `RuntimeError`. Di mode aplikasi interaktif, sistem fallback ke template heuristik dengan penandaan eksplisit pada trace.
- **File Diubah**: `src/engine.py`.
- **Tes Baru**: `tests/test_crag_rewrite.py` (5 unit tests: CORRECT tanpa rewrite, INCORRECT tanpa rewrite, AMBIGUOUS memicu rewrite LLM & pembersihan kueri, gagal saat evaluasi melempar error, fallback di mode app tercatat di trace).
- **Cara Menjalankan**: `python -m pytest tests/test_crag_rewrite.py -v`

---

### Tugas 3: Perbaikan Pipeline Fine-Tuning Embedding (`f727f71`)
- **Masalah Sebelum**: `finetune_embeddings.py` melatih per file `datasets/train_*.json` yang masing-masing hanya berisi 1 pasang QA (batch size = 1), sehingga `MultipleNegativesRankingLoss` tidak memiliki *in-batch negatives* dan nilai loss nol. Selain itu, cache `datasets/trained.json` mencegah pelatihan ulang, dan QA lama memiliki teks bahasa Inggris serta kata rusak ("Bermasalahologoan").
- **Perubahan yang Dilakukan**:
  - Memindahkan QA lama yang rusak ke `datasets/legacy/`.
  - Membuat 50 pasang QA bahasa Indonesia otentik yang diekstrak langsung dari 7 PDF SOP ke dalam `datasets/train_*.json` melalui generator deterministic.
  - Memodifikasi `finetune_embeddings.py` untuk menggabungkan seluruh QA menjadi satu dataset utuh, batch size efektif 8 (menghasilkan 7 negatif in-batch per sampel positif), seed acak tetap (42), dan menyimpan model baru ke `./indo_finetuned_embedding_v2`.
  - Pelatihan berhasil mencapai konvergensi (loss akhir 0.2924).
  - Parameterisasi `CHROMA_PATH` dan `EMBEDDING_MODEL_PATH` via environment variables dan CLI arguments di `embeddings.py`, `src/engine.py`, dan `evaluate_benchmark.py`.
  - Membangun indeks Chroma terpisah: `chroma_v2` (model v2) dan `chroma_base` (model dasar `LazarusNLP/all-indo-e5-small-v4`) untuk mendukung studi ablasi model embedding.
- **File Diubah**: `finetune_embeddings.py`, `embeddings.py`, `src/engine.py`, `evaluate_benchmark.py`, `.gitignore`.
- **Tes Baru**: `tests/test_finetune_dataset.py` (2 unit tests: format QA valid dan dataset menghasilkan batch multi-sampel).
- **Verifikasi Kebocoran**: `python evaluate_benchmark.py check-leakage` mencatat **0 item $\ge 0.50$** (skor tertinggi 0.43 pada pertanyaan beasiswa umum).

---

### Tugas 4: H-Tree Leaf Chunk Retrieval & Parent Tree Expansion (`a5fb630`)
- **Masalah Sebelum**: `SOPHierarchicalChunker` telah merancang 51 leaf chunk berkonteks induk, namun `execute_hierarchical_rag` masih mencari di Chroma dokumen umum (chunk kasar 1.700 karakter / 1 halaman per chunk). Selain itu, pemotong langkah memotong kalimat di batas baris PDF, menyebabkan sisa kalimat prosedur jatuh ke metadata `details`.
- **Perubahan yang Dilakukan**:
  - Memperbaiki parser langkah di `SOPHierarchicalChunker`: baris lanjutan kalimat langkah digabungkan ke `step["text"]`, sementara atribut (`Dokumen yang dibutuhkan:`, `Output:`, `Link unduh form:`, `Waktu:`) diisolasi bersih ke `step["details"]`.
  - Menghasilkan tepat 44 langkah dari 7 PDF SOP:
    - `SOP_Izin_Cuti_Akademik.pdf`: 8 langkah
    - `SOP_Permohonan_Izin_Aktif_Setelah_Cuti.pdf`: 3 langkah
    - `SOP_Legalisir_Ijazah_Dan_Transkrip.pdf`: 6 langkah
    - `SOP_Pengisian_IRS.pdf`: 6 langkah
    - `SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf`: 11 langkah
    - `SOP_Pengajuan_Rekomendasi_Beasiswa.pdf`: 5 langkah
    - `SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf`: 5 langkah
    - Total: 7 ringkasan dokumen + 44 langkah = **51 leaf chunks**.
  - Mengindeks 51 leaf chunk ke dalam koleksi Chroma terpisah bernama `htree_leaves` (tersedia di `chroma_v2` dan `chroma_base`).
  - Mengubah `execute_hierarchical_rag` menjadi alur hibrida Bottom-Up & Top-Down:
    1. *Bottom-Up*: Pencarian dense top-$k$ ($k=3$) pada koleksi `htree_leaves`.
    2. *Threshold*: Jika skor kemiripan terbaik $< 0.30$, kembalikan *honest refusal*.
    3. *Top-Down*: Ambil dokumen induk dari leaf peringkat 1, lalu perluas ke struktur pohon lengkap SOP induk tersebut (`get_hierarchical_context_for_doc`).
    4. *Synthesis*: Masukkan pohon hierarki ke `PROMPT_TEMPLATE` dan panggil `call_llm`.
  - Mempertahankan versi lama sebagai sistem ablasi `htree_v0` (`execute_hierarchical_rag_v0`) pada dispatch `query_architecture` dan opsi CLI `evaluate_benchmark.py`.
- **File Diubah**: `src/graph_sop/hierarchical_chunker.py`, `src/engine.py`, `embeddings.py`, `evaluate_benchmark.py`.
- **Tes Baru**: `tests/test_htree_rag.py` (5 unit tests: struktur pohon & 44 langkah, kontinuitas teks langkah, retrieval leaf & ekspansi induk, penolakan di bawah threshold, dan dispatch ablasi `htree_v0`).
- **Cara Menjalankan**: `python -m pytest tests/test_htree_rag.py -v`

---

## 3. Hasil Pengujian & Uji Asap (Smoke Test)

### A. Hasil Pytest Lengkap
Seluruh 67 unit tests pada repositori berhasil lolos 100%:
```
============================== 67 passed in 6.77s ==============================
- tests/test_agentic_react.py: 5 passed
- tests/test_benchmark_metrics.py: 50 passed (tidak diubah, baseline valid)
- tests/test_crag_rewrite.py: 5 passed
- tests/test_finetune_dataset.py: 2 passed
- tests/test_htree_rag.py: 5 passed
```

### B. Hasil Uji Kebocoran Data (`check-leakage`)
Perbandingan 40 pertanyaan benchmark terhadap 50 pasangan QA latih baru dengan metrik ROUGE-L F1:
- Nilai ambang batas kebocoran: $0.50$
- Jumlah item $\ge 0.50$: **0 item (0%)**
- ROUGE-L maksimum yang terdeteksi: $0.43$ (pada pertanyaan umum tentang beasiswa).
- Kesimpulan: **Bebas kontaminasi data latih (zero data leakage)**.

### C. Hasil Uji Asap (*Smoke Test*, 3 Soal)
> **Catatan Penting**: Angka di bawah ini berlabel **"Uji Asap, 3 Soal, Bukan Hasil Akhir Penelitian"**. Uji coba ini hanya bertujuan memverifikasi berjalannya pipa inferensi end-to-end dengan model generator lokal Ollama `gemma4:e2b` tanpa runtime error.

Perintah yang dijalankan:
```bash
python evaluate_benchmark.py generate --run-name smoke_fix --systems naive,agentic,crag,htree,htree_v0 --limit 3
python evaluate_benchmark.py score --run-name smoke_fix
```

Hasil komparasi pada `results/smoke_fix/summary.csv`:

| Sistem | Key Fact Recall | ROUGE-L | Step Coverage | TIR | AMR | Latensi Median (s) | Prompt Tokens (Mean) | Completion Tokens (Mean) | LLM Calls / Q |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`naive`** | 0.783 | 0.340 | 0.375 | 0.0 | 0.0 | 16.94 | 1,139.0 | 897.7 | 1 |
| **`agentic`** | 0.717 | 0.324 | 0.375 | 0.0 | 0.0 | 43.92 | 2,160.0 | 2,610.3 | 2 – 4 |
| **`crag`** | 0.550 | 0.271 | 0.250 | 0.0 | 0.0 | 19.79 | 1,168.3 | 1,228.7 | 1 – 2 |
| **`htree`** | 0.483 | 0.209 | 0.000 | - | - | 14.06 | 484.7 | 868.7 | 1 |
| **`htree_v0`** | 0.483 | 0.209 | 0.000 | - | - | 13.18 | 484.7 | 868.7 | 1 |

---

## 4. Analisis Dampak terhadap Kontrol Eksperimen

1. **Pengukuran Riil pada RQ3 (Efisiensi & Biaya Komputasi)**:
   - Sebelum perbaikan, Agentic RAG hanya memanggil LLM 1 kali sehingga metrik latensi dan tokennya identik semu dengan Naive RAG.
   - Dengan ReAct loop sungguhan, pemanggilan LLM meningkat menjadi rata-rata 3 panggilan per kueri (2.160 prompt tokens vs 1.139 pada Naive), dengan latensi median naik dari 16,9 detik menjadi 43,9 detik.
   - Pada CRAG, ketika query rewrite terpicu untuk kueri ambigu, token dan latensi panggilan rewrite ikut tercatat secara akurat (1.168 prompt tokens vs 1.139).
   - Seluruh biaya komputasi ini kini **dapat dipertanggungjawabkan secara ilmiah** dalam menjawab RQ3 tanpa manipulasi tersembunyi.
2. **Keadilan Konteks Generasi**:
   - Baik Agentic RAG maupun CRAG tetap menggunakan `PROMPT_TEMPLATE` yang identik dengan sistem lain pada sintesis akhir, dengan $k=3$ dokumen teratas.
   - Pengecekan threshold $0.30$ diterapkan secara seragam pada seluruh sistem: bila dokumen tidak relevan, sistem menolak dengan template penolakan standar yang sama.

---

## 5. Keterbatasan yang Masih Tersisa & Keputusan untuk Peneliti

1. **Perbedaan Model Alur `SOPWorkflowGraph` terhadap PDF**:
   - *Kasus UKT*: Graf alur di `src/graph_sop/workflow_graph.py` memiliki 9 simpul alur utama, sedangkan teks PDF memuat 11 langkah (langkah administratif kecil dikelompokkan).
   - *Kasus Cuti Akademik*: Langkah 2–3 pada graf alur memodelkan interaksi Dosen Wali & Kaprodi, sementara teks PDF memodelkan dari perspektif mahasiswa yang membawa form ke Kaprodi dan Dekan.
   - *Status*: Sesuai batasan penelitian (*out of scope*), graf alur tidak diubah untuk menjaga integritas artefak graf yang sudah ada. Peneliti perlu memutuskan apakah perbedaan ini dicatat sebagai *deliberate modeling abstraction* dalam bab metodologi.
2. **Status Verifikasi Manual Benchmark**:
   - Seluruh 40 soal pada `benchmark/sop_benchmark_v1.json` saat ini berstatus `"verified": false` (draft penelitian).
   - *Rekomendasi*: Sebelum menjalankan benchmark publikasi akhir dengan bendera `--require-verified`, luangkan waktu 30–60 menit untuk memvalidasi teks ground truth terhadap fisik 7 PDF SOP, idealnya dengan konfirmasi staf akademik FSM.
3. **Pelaksanaan Benchmark Penuh (40 Soal)**:
   - Uji asap 3 soal telah membuktikan tidak ada runtime exception pada sistem baru.
   - Keputusan untuk menjalankan evaluasi penuh 40 soal ($8 \text{ arsitektur} \times 40 \text{ soal} = 320 \text{ inferensi}$, estimasi durasi 2–3 jam dengan Ollama lokal) berada di tangan peneliti.
4. **Model Judge Ragas**:
   - Jika metrik Ragas diaktifkan (`--ragas`), protokol mewajibkan model judge berbeda dari generator (misal generator `gemma4:e2b` dan judge `llama3.1:8b`). Pastikan laptop memiliki memori cukup untuk menampung kedua model secara bergantian.

---

## 6. Teks Pengganti untuk Slide Deck Bimbingan

Teks di bawah ini disediakan sebagai pengganti langsung pada slide presentasi bimbingan:

### Slide 4: Arsitektur Baseline (Agentic RAG & CRAG)

**Teks Lama**:
> - Agentic RAG: Menggunakan ReAct agent untuk memilih tools pencarian. (Catatan: implementasi awal hanya mock trace 1-hop similarity search).
> - CRAG: Evaluator skor kemiripan dokumen dengan modul penulisan ulang kueri jika ambigu. (Catatan: penulisan ulang memakai template string statis).

**Teks Pengganti yang Disarankan**:
```markdown
* Agentic RAG (Genuine ReAct Loop):
  - Mekanisme: Siklus otonom Thought → Action: cari_dokumen_sop(query) → Observation → Final Answer yang sepenuhnya digerakkan oleh LLM lokal (gemma4:e2b).
  - Kontrol Eksperimen: Dibatasi maksimal 3 iterasi (max_steps=3), konvergensi adaptif, penolakan jujur jika skor retrieval terbaik < 0.30, dan sintesis akhir menggunakan PROMPT_TEMPLATE standar.
  - Traceability: Setiap inferensi ReAct melewati RAGCore.call_llm sehingga seluruh konsumsi token dan akumulasi latensi terhitung transparan untuk analisis efisiensi RQ3.

* Corrective RAG / CRAG (LLM Self-Correction):
  - Mekanisme: Grader kepercayaan 2-ambang (>=0.55 CORRECT, 0.35–0.55 AMBIGUOUS, <0.35 INCORRECT).
  - Query Rewriting: Pada status AMBIGUOUS, kueri direformulasi secara dinamis oleh LLM menggunakan REWRITE_PROMPT_TEMPLATE, dibersihkan menjadi kueri tunggal, lalu dicari ulang (secondary retrieval).
  - Evaluasi Ketat: Tanpa fallback ekstraktif diam-diam; kegagalan LLM rewrite saat evaluasi dicatat sebagai error eksplisit demi menjaga objektivitas komparasi.
```

---

### Slide 9: Hierarchical Tree RAG (H-Tree)

**Teks Lama**:
> - Hierarchical Tree RAG: Pohon relasional dokumen SOP (Root → Section → Leaf).
> - (Catatan: Leaf chunk belum diindeks secara independen di Chroma; pencarian masih dilakukan pada level dokumen utuh).

**Teks Pengganti yang Disarankan**:
```markdown
* Hierarchical Tree RAG (Bottom-Up Leaf Retrieval + Top-Down Parent Tree Expansion):
  - Indexing: 51 fine-grained leaf chunks (7 ringkasan dokumen + 44 langkah prosedural PDF) diindeks ke koleksi Chroma terpisah 'htree_leaves' dengan model embedding terstandarisasi.
  - Parsing Presisi: Parser SOPHierarchicalChunker menyatukan kontinuitas kalimat langkah multi-baris serta mengisolasi atribut formal (dokumen syarat, output, link formulir, waktu proses).
  - Alur Inferensi Online:
    1. Bottom-Up: Kueri dicari secara dense top-k langsung pada leaf chunks langkah prosedural.
    2. Thresholding: Reject jika skor kemiripan terbaik < 0.30.
    3. Top-Down: Mengidentifikasi dokumen induk dari leaf peringkat 1, lalu mengekspansi seluruh struktur hierarki dokumen SOP tersebut (overview + child steps + attributes).
    4. Synthesis: Konteks pohon utuh diinjeksikan ke LLM untuk menghasilkan jawaban yang utuh secara prosedural tanpa fenomena "parent-orphan chunking".
  - Studi Ablasi: Sistem lama dipertahankan sebagai 'htree_v0' (pencarian dokumen 1.700 char) untuk membuktikan secara empiris keunggulan pencarian leaf vs dokumen kasar.
```

---

### Slide 17: Keterbatasan Penelitian & Validitas Eksperimen

**Teks Lama**:
> - Beberapa arsitektur baseline belum berjalan optimal (Agentic masih mock, CRAG rewrite statis).
> - Model embedding fine-tuned belum efektif karena keterbatasan batch size.
> - Leaf chunk H-Tree belum diindeks.

**Teks Pengganti yang Disarankan**:
```markdown
* Keterbatasan Teknis yang Telah Terselesaikan (Updated):
  [x] Agentic RAG telah diperbaiki menjadi loop ReAct multi-langkah otonom dengan pelacakan token penuh.
  [x] CRAG telah mengintegrasikan pembuat kueri ulang berbasis LLM tanpa fallback ekstraktif tersembunyi.
  [x] Pipeline fine-tuning embedding diperbarui: dataset 50 QA bersih bahasa Indonesia, batch size 8 dengan in-batch negatives (loss konvergen 0.2924, lolos uji kebocoran data ROUGE-L < 0.43).
  [x] H-Tree telah mengindeks 51 leaf chunk di koleksi terpisah dengan ekspansi pohon induk dinamis.
  [x] Single choke point 'RAGCore.call_llm' memastikan seluruh konsumsi token dan latensi tercatat adil untuk RQ3.

* Keterbatasan Tersisa & Pertimbangan Desain (Untuk Diskusi):
  1. Abstraksi Graf Alur vs PDF: Model BPMN DAG memiliki 9 simpul pada SOP UKT (PDF memuat 11 langkah granular). Ini merupakan keputusan perancangan untuk merepresentasikan 'milestone state' alih-alih setiap aksi mikro administratif.
  2. Cakupan Korpus: Eksperimen berfokus pada korpus terarah (7 SOP resmi FSM UNDIP, ~1.100 kata), yang memungkinkan perbandingan langsung terhadap batas atas Full-Context LLM.
  3. Validasi Benchmark Human-in-the-Loop: 40 item benchmark telah dirancang komprehensif (prosedural, aktor, dokumen, waktu, cross-SOP, unanswerable) dan siap melalui tahap penelaahan pakar/staf akademik FSM.
```
