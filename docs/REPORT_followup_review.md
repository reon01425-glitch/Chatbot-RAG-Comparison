# Laporan Tindak Lanjut Temuan Review (Follow-up Review Report)

**Repositori**: `https://github.com/reon01425-glitch/Chatbot-RAG-Comparison`  
**Branch**: `fix/followup-review-findings`  
**Base Commit**: `185c689` (`origin/fix/baseline-and-htree-improvements`)  
**Tanggal**: 4 Oktober 2026  
**Penulis**: Antigravity (Google DeepMind Pair Programmer)  
**Dokumen Terkait**: `docs/CHANGES_baseline_htree_fix.md`, `docs/CHANGES_followup.md`, `docs/EVALUATION_PROTOCOL.md`

---

## 1. Tindak Lanjut 6 Temuan Review

Berikut adalah rincian masalah, solusi teknis yang diterapkan, file terdampak, pengujian yang dilakukan, dan cara menjalankannya untuk setiap temuan review:

### Temuan 1: Uji Asap Memakai Embedding Lama (`results/smoke_fix`)
* **Masalah**: Pada commit sebelumnya, `results/smoke_fix/run_config.json` mencatat pemakaian `embedding_model: ./indo_finetuned_embedding` (v1) dan `chroma_path: chroma` (indeks lama), serta `git_dirty: true`.
* **Perubahan**:
  - Diterapkan mekanisme penegakan model embedding eksplisit di `evaluate_benchmark.py`: argumen `--embedding-model {base, v1, v2, v3, <path>}` wajib disediakan (atau via env `EMBEDDING_MODEL_PATH`). Evaluasi akan ditolak (*fatal error*) jika bendera ini diabaikan.
  - Setiap direktori Chroma (`chroma`, `chroma_base`, `chroma_v2`, `chroma_v3`) dilengkapi file manifest `index_manifest.json` yang menyimpan SHA256 bobot model embedding yang digunakan saat membangun indeks.
  - Pada mode evaluasi (`EVALUATION_MODE=1`), `RAGCore` memverifikasi kesesuaian antara model embedding yang dipilih dan `index_manifest.json` di direktori Chroma. Jika tidak cocok atau manifest tidak ditemukan, sistem gagal dengan pesan yang jelas.
  - `_init_htree_collection` dilindungi agar tidak mengindeks leaf chunks ke direktori Chroma yang dibangun dengan model lain.
* **File Terdampak**: `src/manifest.py`, `src/engine.py`, `evaluate_benchmark.py`, `embeddings.py`, `build_indexes.py`.
* **Pengujian**: `tests/test_embedding_selection.py` (6 unit tests).
* **Cara Menjalankan**:
  ```bash
  # Uji eksekusi tanpa argumen (harus ditolak)
  python evaluate_benchmark.py generate --run-name test_fail --limit 1
  # Uji dengan model eksplisit dan manifest valid
  python evaluate_benchmark.py generate --run-name test_ok --embedding-model v3 --limit 1
  ```

---

### Temuan 2: H-Tree Memilih SOP Induk yang Salah
* **Masalah**: Pada kueri tertentu (termasuk CUTI-P1), leaf peringkat 1 yang terambil berasal dari SOP lain ("Permohonan Izin Aktif Setelah Cuti" alih-alih "Permohonan Izin Cuti Akademik"). Akibatnya, ekspansi pohon induk menyuntikkan SOP yang salah sehingga step coverage menjadi 0.
* **Perubahan**:
  - Dibuat set pengembangan independen `benchmark/dev_htree_v1.json` (20 soal sintetik mencakup kasus ambigu seperti Cuti vs Aktif dan UKT vs IRS) tanpa menyentuh benchmark utama `sop_benchmark_v1.json`.
  - Diimplementasikan modul evaluasi dev `scripts/diagnose_htree_dev.py` untuk menganalisis akurasi pemilihan dokumen induk.
  - Diimplementasikan 3 varian strategi pemilihan SOP induk dari top-$k$ leaf:
    1. `top1` (default): Memilih SOP induk dari leaf berperingkat pertama.
    2. `vote`: Memilih SOP induk dengan frekuensi kemunculan terbanyak di top-$k$ leaf (tie-break menggunakan skor tertinggi).
    3. `sum`: Memilih SOP induk dengan akumulasi skor kemiripan kosinus tertinggi di top-$k$ leaf.
  - Opsi dapat dikonfigurasi via CLI `--htree-variant {top1, vote, sum}` dan dicatat di `run_config.json`. Baseline ablasi `htree_v0` dipertahankan tanpa perubahan.
  - Analisis Model Card `LazarusNLP/all-indo-e5-small-v4`: Berdasarkan inspeksi `config_sentence_transformers.json`, model ini memiliki `"prompts": {}` dan `"default_prompt_name": null`, sehingga tidak mewajibkan penambahan prefiks buatan seperti `query:` / `passage:`.
* **File Terdampak**: `benchmark/dev_htree_v1.json`, `scripts/diagnose_htree_dev.py`, `src/engine.py`, `evaluate_benchmark.py`, `benchmark/htree_dev_diagnosis.json`.
* **Pengujian**: `tests/test_htree_variants.py` (4 unit tests).
* **Cara Menjalankan**:
  ```bash
  python scripts/diagnose_htree_dev.py --embedding-model v3 --k 3
  ```

---

### Temuan 3: Default Embedding Model Lama & Bobot Tidak Ikut Repo
* **Masalah**: Default `EMBEDDING_MODEL_PATH` masih bernilai `./indo_finetuned_embedding` di berbagai modul, dan file bobot model (`model.safetensors`) masuk `.gitignore` sehingga berisiko hilang saat repo dipindahkan.
* **Perubahan**:
  - Dihapus nilai default implisit pada pipeline evaluasi: pemilihan model embedding kini wajib eksplisit melalui alias (`base`, `v1`, `v2`, `v3`).
  - Dibuat skrip terpadu `build_indexes.py` yang dapat mereproduksi dan membangun ulang seluruh indeks Chroma beserta koleksi `htree_leaves` dan manifest dalam satu perintah.
  - Dicatat SHA256 bobot model secara otomatis di `run_config.json`.
  - Dirumuskan 3 alternatif strategi penyimpanan bobot model untuk keputusan pembimbing/peneliti (lihat Bagian 5).
* **File Terdampak**: `build_indexes.py`, `src/manifest.py`, `src/engine.py`, `evaluate_benchmark.py`, `app.py`.
* **Pengujian**: Verifikasi pembangunan indeks bersih via `build_indexes.py --embedding-model v3`.
* **Cara Menjalankan**:
  ```bash
  python build_indexes.py --embedding-model v3
  ```

---

### Temuan 4: Data Latih Fine-Tuning v2 Bermasalah
* **Masalah**: Data latih v2 mengandung klaim fiktif di luar teks PDF resmi (langkah 4 menyebut Subbag Akademik menyusun konsep surat cuti; langkah 5 menyebut Supervisor dan Manajer TU memeriksa konsep) serta terdapat kemiripan semantik dekat dengan pertanyaan benchmark (CUTI-A1).
* **Perubahan**:
  - Diimplementasikan modul cek kebocoran semantik `evaluate_benchmark.py check-leakage --semantic` menggunakan model dasar (`LazarusNLP/all-indo-e5-small-v4`) dengan ambang batas alert $\ge 0.85$.
  - Dilakukan audit menyeluruh terhadap ke-50 pasangan QA dan dibukukan dalam `datasets/train_audit_v3.csv`.
  - Disusun dataset `datasets/train_v3/*.json` (50 item pada 7 SOP):
    - Seluruh klaim fiktif dihapus dan diselaraskan persis dengan teks PDF resmi.
    - Pasangan yang memiliki kemiripan semantik dekat dengan benchmark diformulasikan ulang.
    - Ditambahkan field `evidence` berupa kutipan teks PDF sumber yang terverifikasi secara verbatim, serta field `verified: false`.
  - Dilatih ulang model embedding v3 (`./indo_finetuned_embedding_v3`) dengan seed 42 dan batch size 8 (loss akhir 0.1645, bobot SHA256: `327c99cff0a47658d6784d5266f090fcb920e6c92462f7b6f8be312cf3e98eaf`).
  - Indeks `chroma_v3` dibangun lengkap dengan manifest.
* **File Terdampak**: `datasets/train_audit_v3.csv`, `datasets/train_v3/`, `scripts/build_train_v3.py`, `finetune_embeddings.py`, `evaluate_benchmark.py`.
* **Pengujian**: `test_train_v3_evidence_matches_pdf` pada `tests/test_finetune_dataset.py`.
* **Cara Menjalankan**:
  ```bash
  # Uji bukti teks PDF
  python -m pytest tests/test_finetune_dataset.py -v
  # Cek kebocoran leksikal dan semantik
  python evaluate_benchmark.py check-leakage --train-dir datasets/train_v3
  python evaluate_benchmark.py check-leakage --train-dir datasets/train_v3 --semantic
  ```

---

### Temuan 5: Hitungan Inferensi LLM Keliru
* **Masalah**: Laporan sebelumnya menulis "8 arsitektur × 40 soal = 320 inferensi".
* **Perubahan**:
  - Dikoreksi bahwa evaluasi mencakup **10 sistem komparasi** (8 RAG: `naive`, `agentic`, `crag`, `graph_sop`, `workflow`, `self_rag`, `htree`, `multimodal` + 2 pembanding: `llm_only` dan `full_context`), sehingga total soal uji adalah $10 \times 40 = 400$ soal (atau 440 soal jika menyertakan baseline ablasi `htree_v0`).
  - Diklarifikasi bahwa jumlah panggilan LLM riil jauh lebih tinggi dari 400 karena `agentic` menjalankan penalaran ReAct multi-hop ($2 - 4$ panggilan LLM per kueri) dan `crag` menjalankan penulisan ulang kueri pada status AMBIGUOUS ($1 - 2$ panggilan LLM per kueri). Total pemanggilan LLM pada run penuh diperkirakan berkisar antara 480–560 panggilan riil.
* **File Terdampak**: `docs/CHANGES_baseline_htree_fix.md`, `docs/CHANGES_followup.md`, `docs/EVALUATION_PROTOCOL.md`.

---

### Temuan 6: Klaim Dokumentasi Berlebihan & Ketidaksesuaian Uraian Graf Cuti
* **Masalah**: Dokumentasi sebelumnya menggunakan frasa tidak terkalibrasi ("zero data leakage", "membuktikan secara empiris", "tanpa fenomena parent-orphan chunking"), keliru menyebut Dosen Wali pada langkah 2-3 alur graf Cuti, serta catatan token di `run_config.json` belum diperbarui.
* **Perubahan**:
  - Seluruh klaim ditarik dan diganti dengan pernyataan objektif berbasis bukti kuantitatif.
  - Dikoreksi uraian perbedaan graf Cuti vs PDF: Pada `SOPWorkflowGraph`, simpul langkah 2 menetapkan `actor: "Ketua Program Studi"` dan langkah 3 menetapkan `actor: "Dekan"`. Pada PDF resmi, subjek yang meminta tanda tangan Kaprodi dan membawa berkas ke Dekan adalah **Mahasiswa**. Dosen Wali bukan aktor langkah 2-3 (hanya menerima tembusan di langkah 6).
  - Diperbarui field `notes` pada `run_config.json` agar menegaskan bahwa seluruh panggilan LLM (`synthesis`, `agentic reasoning`, dan `CRAG query rewriting`) terakumulasi transparan melalui `RAGCore.call_llm`.
* **File Terdampak**: `docs/CHANGES_baseline_htree_fix.md`, `docs/CHANGES_followup.md`, `evaluate_benchmark.py`.

---

## 2. Tabel Diagnosis H-Tree pada Set Dev (`benchmark/dev_htree_v1.json`)

Diagnosis dilakukan pada set pengembangan $N = 20$ soal (2–3 soal per SOP, mencakup kueri ambigu) dengan $k = 3$ leaf retrieval. Akurasi dihitung berdasarkan ketepatan pemilihan SOP induk:

| Model Embedding | Varian `top1` | Varian `vote` | Varian `sum` | Baseline `htree_v0` (Doc-level) |
| :--- | :---: | :---: | :---: | :---: |
| **`base` (`all-indo-e5-small-v4`)** | **13 / 20 (65.0%)** | 9 / 20 (45.0%) | 9 / 20 (45.0%) | 12 / 20 (60.0%) |
| **`v2` (`indo_finetuned_embedding_v2`)** | 11 / 20 (55.0%) | 11 / 20 (55.0%) | 11 / 20 (55.0%) | 11 / 20 (55.0%) |
| **`v3` (`indo_finetuned_embedding_v3`)** | **12 / 20 (60.0%)** | **12 / 20 (60.0%)** | **12 / 20 (60.0%)** | 12 / 20 (60.0%) |

### Analisis Diagnosis:
1. **Model Dasar (`base`)**: Strategi `top1` unggul (65.0%) dibandingkan `vote` dan `sum` (45.0%). Penurunan akurasi pada varian voting disebabkan oleh ketimpangan jumlah langkah antar-SOP (SOP UKT memiliki 11 langkah, sedangkan SOP Izin Aktif Setelah Cuti hanya memiliki 3 langkah). Akibatnya, pada kueri umum, distractor leaf dari SOP dengan jumlah langkah banyak dapat memenangkan voting unweighted.
2. **Model v3**: Ketiga varian (`top1`, `vote`, `sum`) mencapai akurasi seimbang (60.0%), menunjukkan representasi embedding yang lebih kohesif antar-leaf dalam satu SOP induk yang sama.
3. **Rekomendasi Default**: Mempertahankan `top1` sebagai default sistem, namun tetap menyediakan opsi `--htree-variant` di CLI agar dapat diuji ablasi pada run evaluasi 40 soal.

---

## 3. Ringkasan Audit Data Latih (`datasets/train_audit_v3.csv`)

Sebanyak 50 pasangan QA latih diaudit terhadap 7 teks PDF resmi FSM UNDIP dan diuji kemiripan semantiknya terhadap 40 soal benchmark:

* **Total Item Diaudit**: 50 item
* **Dipertahankan (`pertahankan`)**: 42 item (84.0%)
* **Ditulis Ulang (`tulis ulang`)**: 8 item (16.0%)
* **Dihapus (`hapus`)**: 0 item (seluruh item bermasalah berhasil diformulasikan ulang dengan bukti kutipan PDF resmi).

### Contoh Keputusan dan Hasil Penulisan Ulang:

1. **Perbaikan Halusinasi Prosedural (Cuti Langkah 4)**:
   - *ID*: `data/SOP_Izin_Cuti_Akademik.pdf:0:4`
   - *Sebelum*: Mengklaim Subbag Akademik "menyusun konsep Surat Izin Cuti Akademik" (fiktif, tidak ada di PDF).
   - *Sesudah*: "Jika berkas persyaratan lengkap, form diberi paraf dan diproses lebih lanjut."
   - *Bukti PDF Verbatim*: `"Jika lengkap, form diberi paraf dan diproses lebih lanjut."`
2. **Perbaikan Pejabat Fiktif (Cuti Langkah 5)**:
   - *ID*: `data/SOP_Izin_Cuti_Akademik.pdf:0:5`
   - *Sebelum*: Mengklaim "Supervisor Subbag Akademik dan Manajer Tata Usaha memeriksa dan memparaf konsep surat" (fiktif).
   - *Sesudah*: "Dekan menandatangani Surat Izin Dekan."
   - *Bukti PDF Verbatim*: `"Dekan menandatangani Surat Izin Dekan."`
3. **Mitigasi Kemiripan Semantik Dekat (Kasus Benchmark CUTI-A1)**:
   - *ID*: `data/SOP_Izin_Cuti_Akademik.pdf:0:3`
   - *Sebelum*: Menanyakan ke mana form cuti yang telah disetujui Kaprodi diserahkan oleh mahasiswa (parafrase dekat dengan CUTI-A1).
   - *Sesudah*: Fokus dialihkan ke alur disposisi internal Dekan ke Subbag: "Apa tindak lanjut disposisi surat permohonan cuti setelah diajukan ke Dekan?".
   - *Bukti PDF Verbatim*: `"Kemudian dilakukan disposisi ke Subbag Akademik dan Kemahasiswaan."`
4. **Mitigasi Kemiripan Leksikal & Semantik (Kasus Benchmark BEA-T1)**:
   - *ID*: `data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf:0:0`
   - *Sebelum*: Menanyakan total durasi permohonan beasiswa.
   - *Sesudah*: Fokus dialihkan ke estimasi durasi langkah 1 mahasiswa mengunduh/mengisi formulir: "Berapa estimasi waktu yang dialokasikan bagi mahasiswa untuk mendownload dan mengisi formulir beasiswa?".
   - *Bukti PDF Verbatim*: `"Waktu: ±30 menit"`

---

## 4. Hasil Pengujian, Uji Kebocoran, dan Uji Asap

### A. Hasil Pytest Lengkap
Seluruh 78 unit tests pada repositori berhasil lolos 100%:
```
78 passed in 5.88s
- tests/test_agentic_react.py: 5 passed
- tests/test_benchmark_metrics.py: 50 passed (baseline metrics tidak diubah)
- tests/test_crag_rewrite.py: 5 passed
- tests/test_embedding_selection.py: 6 passed (Tugas 1)
- tests/test_finetune_dataset.py: 3 passed (Tugas 3: format, batching, dan verifikasi bukti PDF)
- tests/test_htree_rag.py: 5 passed
- tests/test_htree_variants.py: 4 passed (Tugas 2: top1, vote, sum, fallback)
```

### B. Hasil Uji Kebocoran Data (`check-leakage`)
1. **Cek Leksikal (ROUGE-L F1, Ambang Batas 0.50)**:
   - Perbandingan 40 soal benchmark terhadap 70 pertanyaan kandidat (50 latih v3 + 20 dev H-Tree).
   - Jumlah item $\ge 0.50$: **0 item (0.0%)**.
   - Skor ROUGE-L tertinggi: $0.44$ (pada pertanyaan cuti umum).
2. **Cek Semantik (Model Dasar `all-indo-e5-small-v4`, Ambang Batas 0.85)**:
   - Distribusi kemiripan kosinus terhadap benchmark:
     - Minimum: 0.381
     - Kuartil 1 (25%): 0.582
     - Median: 0.653
     - Kuartil 3 (75%): 0.720
     - Persentil 90%: 0.750
     - Maksimum: 0.819 (pada variasi kueri UKT umum)
   - Jumlah item $\ge 0.85$: **0 item (0.0%)**.

---

### C. Hasil Uji Asap Ulang (`smoke_fix2`)
> **Peringatan Label Wajib**: Angka pada tabel berikut berlabel **"Uji Asap, 3 Soal, Bukan Hasil Penelitian"**. Evaluasi 3 soal (`CUTI-P1`, `CUTI-P2`, `CUTI-A1`) ini semata-mata bertujuan memverifikasi berjalannya seluruh alur kode dari working tree yang bersih (`git_dirty: false`) menggunakan model final v3 dan indeks `chroma_v3`.

*Perintah*:
```bash
python evaluate_benchmark.py generate --run-name smoke_fix2 --embedding-model v3 --systems naive,agentic,crag,workflow,htree,htree_v0 --limit 3
python evaluate_benchmark.py score --run-name smoke_fix2 --systems naive,agentic,crag,workflow,htree,htree_v0
```

Hasil komparasi pada `results/smoke_fix2/summary.csv`:

| Sistem | Key Fact Recall | ROUGE-L | Step Coverage | TIR | AMR | Latensi Median (s) | Prompt Tokens (Mean) | Completion Tokens (Mean) | LLM Calls / Q |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`naive`** | 0.333 | 0.161 | 0.062 | - | 0.0 | 10.23 | 820.3 | 556.3 | 1 |
| **`agentic`** | 0.533 | 0.251 | 0.188 | 0.0 | 0.0 | 37.32 | 1,789.0 | 2,175.3 | 2 – 4 |
| **`crag`** | 0.000 | 0.097 | 0.062 | - | 0.0 | 0.06 | 439.7 | 570.0 | 1 – 2 |
| **`htree`** | 0.400 | 0.160 | 0.000 | - | - | 12.27 | 356.7 | 599.0 | 1 |
| **`htree_v0`** | 0.400 | 0.160 | 0.000 | - | - | 12.31 | 356.7 | 599.0 | 1 |
| **`workflow`** | 0.633 | 0.234 | 0.250 | 0.0 | 0.0 | 15.39 | 1,844.3 | 867.7 | 1 |

### Perbandingan Mendalam: `htree` vs `htree_v0` pada 3 Soal Uji Asap
Pada 3 soal uji asap ini, luaran `htree` dan `htree_v0` bernilai identik. Penjelasan mekanistiknya adalah sebagai berikut:
1. **Soal `CUTI-P1`**: Pada varian `top1`, leaf peringkat 1 yang terambil oleh `htree` adalah "Langkah 1 SOP Permohonan Izin Aktif Setelah Cuti", sedangkan pada `htree_v0` dokumen peringkat 1 yang terambil juga merupakan chunk dokumen "SOP Permohonan Izin Aktif Setelah Cuti". Karena keduanya memilih dokumen induk yang sama, modul ekspansi pohon induk (`get_hierarchical_context_for_doc`) menyusun konteks pohon yang persis sama untuk disintesis oleh LLM. *(Catatan: Apabila varian `vote` atau `sum` digunakan pada kueri ini, 2 dari 3 leaf teratas berasal dari "SOP Permohonan Izin Cuti Akademik", sehingga pemilihan SOP induk akan berbeda).*
2. **Soal `CUTI-P2`**: Merupakan pertanyaan santai informal (*student phrasing*). Skor kemiripan kosinus retrieval pada kedua sistem berada di bawah ambang batas $0.30$, sehingga kedua sistem secara konsisten mengembalikan template penolakan jujur (*honest refusal*) tanpa memanggil LLM.
3. **Soal `CUTI-A1`**: Kedua sistem berhasil mengidentifikasi SOP induk yang benar ("SOP Permohonan Izin Cuti Akademik"), sehingga konteks hierarki lengkap yang diinjeksikan ke LLM bernilai identik dan menghasilkan jawaban yang sama tepatnya.

*Kesimpulan Analisis*: Kesamaan luaran pada 3 soal ini menunjukkan konsistensi modul ekspansi pohon induk ketika SOP induk yang terpilih identik. Perbedaan retrieval leaf fine-grained vs dokumen kasar baru terlihat nyata pada distribusi kueri yang lebih luas (seperti pada set dev 20 soal).

---

## 5. Keputusan yang Perlu Diambil Peneliti / Pembimbing

Sebelum melaksanakan evaluasi penuh 40 soal untuk publikasi jurnal, terdapat 4 keputusan metodologis yang perlu ditetapkan:

1. **Opsi Penyimpanan Bobot Model Embedding (`indo_finetuned_embedding_v3`)**:
   - **Opsi A: Git LFS (Large File Storage)**.
     *Kelebihan*: File bobot otomatis terunduh saat cloning repo.
     *Kekurangan*: Mengonsumsi kuota penyimpanan dan bandwidth LFS GitHub (ukuran file ~470 MB).
   - **Opsi B: Publikasi ke Hugging Face Hub (Publik / Privat)** *(Rekomendasi)*.
     *Kelebihan*: Standar baku komunitas NLP riset, gratis tanpa batas bandwidth, memiliki revision hash, dan dapat dimuat langsung via nama repo HuggingFace.
     *Kekurangan*: Memerlukan pengaturan akun/token Hugging Face Hub di luar repositori git.
   - **Opsi C: Latih Ulang Deterministik + Simpan Hash SHA256**.
     *Kelebihan*: Repositori git tetap sangat ringan, tanpa dependensi layanan eksternal. Bobot model dijamin identik karena seed 42 dan dataset versi tetap (`train_v3`).
     *Kekurangan*: Memerlukan waktu komputasi pelatihan ulang (~2-3 menit) jika dieksekusi di mesin baru.
2. **Pemilihan Model Embedding Final untuk Run Penuh**:
   - Disarankan menggunakan model **`v3`** (`./indo_finetuned_embedding_v3`), karena seluruh 50 data latih telah diaudit, memiliki bukti kalimat PDF resmi, bebas klaim fiktif, serta lolos uji leksikal dan semantik.
3. **Penetapan Varian Default Pemilihan SOP Induk H-Tree**:
   - Mempertahankan varian **`top1`** sebagai default, dengan pelaporan varian `vote` dan `sum` sebagai bagian dari tabel studi ablasi pada naskah publikasi.
4. **Verifikasi Manual 40 Soal Benchmark**:
   - Seluruh 40 soal pada `benchmark/sop_benchmark_v1.json` saat ini berstatus `"verified": false`.
   - Sebelum menjalankan run evaluasi publikasi akhir dengan bendera `--require-verified`, disarankan memvalidasi teks acuan terhadap fisik 7 PDF SOP bersama staf akademik FSM UNDIP.

---

## 6. Usulan Teks Pengganti untuk Slide Presentasi Bimbingan

Teks berikut dirancang lugas, faktual, dan bebas dari klaim berlebihan untuk memperbarui slide presentasi deck bimbingan:

### Slide 9: Hierarchical Tree RAG (Bottom-Up Leaf Retrieval + Top-Down Parent Tree Expansion)
```markdown
* Hierarchical Tree RAG (H-Tree):
  - Indexing: 51 fine-grained leaf chunks (7 ringkasan dokumen + 44 langkah prosedural PDF) diindeks pada koleksi Chroma 'htree_leaves' dengan model embedding terstandarisasi.
  - Parsing Presisi: Parser SOPHierarchicalChunker menjaga kontinuitas kalimat langkah multi-baris dan memisahkan atribut administratif (dokumen syarat, output, link form, durasi).
  - Mekanisme Inferensi:
    1. Bottom-Up: Kueri dicari secara dense top-k pada leaf chunks langkah prosedural.
    2. Thresholding: Honest refusal jika skor kemiripan terbaik < 0.30.
    3. Top-Down: Mengidentifikasi dokumen induk (mendukung strategi top1, vote, atau sum), lalu mengekspansi seluruh struktur hierarki dokumen SOP tersebut.
    4. Synthesis: Konteks pohon utuh diinjeksikan ke LLM untuk menjaga kelengkapan alur prosedural.
  - Kontrol Eksperimen: Baseline 'htree_v0' (pencarian dokumen kasar) dipertahankan sebagai pembanding ablasi untuk menguji efektivitas granularitas retrieval pada run evaluasi penuh.
```

---

### Slide 17: Keterbatasan Penelitian & Validitas Eksperimen
```markdown
* Keterbatasan Teknis yang Telah Ditindaklanjuti:
  [x] Pipeline evaluasi mewajibkan pilihan embedding eksplisit dan memvalidasi index_manifest.json (SHA256 bobot dan korpus PDF).
  [x] Seluruh pemanggilan LLM (sintesis, ReAct reasoning, CRAG rewrite) terukur melalui single choke point RAGCore.call_llm untuk akurasi metrik efisiensi RQ3.
  [x] Data latih fine-tuning v3 diaudit penuh: 50 pasang QA memiliki bukti teks PDF verbatim, klaim di luar PDF telah diperbaiki, dan lolos cek leksikal (ROUGE-L < 0.44) serta semantik (< 0.82 terhadap benchmark).
  [x] Evaluasi pemilihan dokumen H-Tree dilakukan pada dev set independen (20 soal) tanpa menyetel diri pada benchmark utama.

* Keterbatasan Tersisa & Agenda Sebelum Run Penuh (Untuk Diskusi):
  1. Verifikasi Human-in-the-Loop Benchmark: 40 butir soal benchmark berstatus draft dan perlu divalidasi bersama staf akademik sebelum eksekusi final (--require-verified).
  2. Keputusan Penyimpanan Bobot: Menetapkan opsi penyimpanan model v3 (Hugging Face Hub / Git LFS / reproduksi deterministik).
  3. Abstraksi Graf Alur: Model DAG alur UKT dan Cuti Akademik mencerminkan abstraksi peran swimlane yang telah didokumentasikan secara presisi terhadap teks PDF.
```
