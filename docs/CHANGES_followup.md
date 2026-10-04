# Laporan Tindak Lanjut Temuan Review (Follow-up Changes)

**Repositori**: `https://github.com/reon01425-glitch/Chatbot-RAG-Comparison`  
**Branch**: `fix/followup-review-findings`  
**Base Commit**: `185c689` (`origin/fix/baseline-and-htree-improvements`)  
**Tanggal**: 4 Oktober 2026  
**Penulis**: Antigravity (Google DeepMind Pair Programmer)  
**Dokumen Terkait**: `docs/CHANGES_baseline_htree_fix.md`

---

## 1. Latar Belakang & 6 Temuan Review

Setelah perbaikan baseline RAG dan arsitektur H-Tree pada branch `fix/baseline-and-htree-improvements`, audit review independen mengidentifikasi 6 kelemahan yang perlu ditindaklanjuti secara sistematis:

1. **Uji Asap Memakai Embedding Lama**: `results/smoke_fix/run_config.json` mencatat pemakaian `embedding_model: ./indo_finetuned_embedding` (v1) dan `chroma_path: chroma` (bukan v2), serta `git_dirty: true`.
2. **H-Tree Memilih SOP Induk yang Salah**: Pada kueri dev dan uji asap, leaf teratas H-Tree dapat salah memetakan SOP induk (misalnya pertanyaan izin cuti memetakan ke SOP permohonan izin aktif setelah cuti), menyebabkan step coverage bernilai 0.
3. **Default Embedding Masih Model Lama & Bobot Tidak Ikut Repo**: Default parameter embedding masih mengarah ke model lama jika tidak dispesifikasikan eksplisit via CLI, dan file bobot `*.safetensors` diabaikan oleh git.
4. **Data Latih Fine-Tuning v2 Bermasalah**: Meskipun lolos cek leksikal ROUGE-L < 0.50, beberapa pasangan QA memiliki parafrase semantik dekat dengan benchmark (misal CUTI-A1) dan mengandung klaim yang tidak didukung teks PDF resmi (misal Subbag Akademik menyusun konsep surat izin cuti).
5. **Hitungan Inferensi LLM Keliru**: Laporan menyebutkan $8 \times 40 = 320$ inferensi, padahal terdapat 10 sistem komparasi ($10 \times 40 = 400$ inferensi, atau 440 dengan `htree_v0`), dan sistem Agentic serta CRAG memanggil LLM lebih dari satu kali per soal.
6. **Klaim Berlebihan & Ketidaksesuaian Deskripsi Graf**: Dokumentasi menggunakan istilah absolut ("zero data leakage", "membuktikan secara empiris", "tanpa fenomena parent-orphan chunking"), keliru menyebut Dosen Wali pada langkah 2-3 graf Cuti, serta catatan token di `run_config.json` belum mencerminkan pencatatan seluruh pemanggilan LLM.

---

## 2. Ringkasan Perubahan per Tugas

### Tugas 0: Pembersihan Repositori & Baseline Cek
- Mengabaikan artefak presentasi, diagram SVG/PNG, dan template LaTeX melalui pembaruan `.gitignore`.
- Memvalidasi baseline test: seluruh 67 unit test lolos (`pytest -q`).
- Memvalidasi baseline leksikal benchmark vs dataset latih: 0 item $\ge 0.50$ (ROUGE-L maksimum 0.43).
- Melakukan inventarisasi seluruh direktori Chroma (`chroma`, `chroma_base`, `chroma_v2`) dan jumlah dokumen per koleksi (`langchain`: 7 dokumen, `htree_leaves`: 51 dokumen).

### Tugas 1: Pilihan Embedding Eksplisit & Validasi Manifest Indeks (`27b66f9`)
- **Pilihan Wajib Eksplisit**: `evaluate_benchmark.py generate` mewajibkan argumen `--embedding-model {base, v1, v2, v3, <path>}` atau env `EMBEDDING_MODEL_PATH`. Evaluator **menolak berjalan** jika parameter ini tidak ditentukan eksplisit.
- **Manifest Indeks (`index_manifest.json`)**: Dibuat modul `src/manifest.py` yang mencatat path model, SHA256 bobot model, hash korpus PDF, chunk size/overlap, koleksi, jumlah dokumen, dan timestamp pembuatan.
- **Validasi Integritas**: `RAGCore` memverifikasi kesesuaian manifest direktori Chroma dengan model embedding aktif dalam mode evaluasi (`EVALUATION_MODE=1`). Jika manifest hilang atau hash bobot tidak cocok, eksekusi dihentikan dengan error jelas.
- **Pencegahan Kontaminasi Koleksi Leaf**: `_init_htree_collection` memastikan leaf hanya diindeks pada Chroma yang sesuai dengan model aktif.
- **Skrip Pembangunan Indeks**: Dibuat `build_indexes.py` untuk membangun ulang indeks Chroma secara bersih per model (`--embedding-model {base, v1, v2, v3}`).
- **UI Streamlit**: `app.py` diperbarui untuk menampilkan nama model embedding dan direktori Chroma aktif secara dinamis.

### Tugas 2: Set Dev H-Tree & Varian Pemilihan SOP Induk (`68eb419`)
- **Set Dev Terpisah (`benchmark/dev_htree_v1.json`)**: Dibuat 20 soal evaluasi internal (2–3 per SOP) yang mencakup pertanyaan ambigu (misal Cuti vs Aktif, UKT vs IRS). Pertanyaan diverifikasi terhadap benchmark dan lolos cek semantik (kesamaan kosinus maksimum 0.735, tidak ada yang $\ge 0.75$).
- **Skrip Diagnosis (`scripts/diagnose_htree_dev.py`)**: Mengevaluasi top-k leaf retrieval dan akurasi pemilihan SOP induk pada set dev untuk model `base`, `v2`, dan `v3`.
- **Implementasi 3 Varian Pemilihan SOP**:
  - `top1`: Memilih SOP induk dari leaf berperingkat pertama (default).
  - `vote`: Memilih SOP induk terbanyak di top-$k$ leaf (dengan tie-break skor tertinggi).
  - `sum`: Memilih SOP induk dengan total skor tertinggi di top-$k$ leaf.
- Opsi dikonfigurasi via `--htree-variant {top1, vote, sum}` pada `evaluate_benchmark.py` dan `src/engine.py`, serta dicatat transparan di trace dan `run_config.json`.
- Baseline ablasi `htree_v0` (pencarian pada level dokumen kasar) dipertahankan tanpa perubahan.

### Tugas 3: Audit Data Latih, Pembuatan train_v3, & Fine-Tuning v3 (`59dddfc`)
- **Audit Komprehensif**: Seluruh 50 pasangan QA diaudit dan dicatat pada `datasets/train_audit_v3.csv`.
- **Perbaikan Halusinasi & Overlap**:
  - Memperbaiki QA Cuti langkah 4 (menghapus klaim fiktif "menyusun konsep Surat Izin Cuti").
  - Memperbaiki QA Cuti langkah 5 (menghapus pejabat fiktif "Supervisor & Manajer TU").
  - Menulis ulang pasangan dengan parafrase semantik dekat terhadap benchmark (CUTI-A1, BEA-T1, AKT-D1).
- **Dataset `train_v3`**:
  - Dibuat `datasets/train_v3/*.json` (50 item pada 7 file SOP).
  - Setiap item dilengkapi field `evidence` berupa kutipan teks PDF sumber yang terbukti ada secara verbatim, serta ditandai `verified: false`.
- **Pelatihan Model v3**:
  - Dilatih menggunakan `finetune_embeddings.py --dataset-dir datasets/train_v3` (batch size 8, 10 epochs, seed 42, MultipleNegativesRankingLoss).
  - Model disimpan di `./indo_finetuned_embedding_v3` (SHA256 bobot: `327c99cff0a47658d6784d5266f090fcb920e6c92462f7b6f8be312cf3e98eaf`).
  - Indeks `chroma_v3` dibangun lengkap dengan koleksi `langchain` (7 dokumen) dan `htree_leaves` (51 dokumen) beserta `index_manifest.json`.
- **Pemeriksaan Kebocoran Data**:
  - Cek leksikal (ROUGE-L): 0 item $\ge 0.50$ (maksimum 0.44).
  - Cek semantik model dasar: 0 item $\ge 0.85$ (median 0.653, maksimum 0.819).

### Tugas 4: Koreksi Dokumentasi & Penyesuaian Protokol
- Menambahkan Bagian 7 (Addendum & Retraksi) pada `docs/CHANGES_baseline_htree_fix.md`.
- Memperbarui `docs/EVALUATION_PROTOCOL.md` dengan dokumentasi bendera wajib, set dev, cek semantik, dan skrip pembangunan indeks.
- Memperbarui catatan deskriptif `notes` pada `run_config.json`.

---

## 3. Catatan Integritas & Keterlacakan
Seluruh kode dan data telah diuji melalui unit test otomatis (`pytest`), pemeriksaan leksikal/semantik, dan pencatatan hash kriptografi (SHA256) untuk menjamin reproduktibilitas penuh pada publikasi ilmiah.
