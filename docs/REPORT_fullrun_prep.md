# Laporan Goal A — Persiapan run penuh

Branch `exp/full-run-v1` dari `fix/followup-review-findings` (`d421b93`). Satu commit per tugas. `main` tidak diubah, tidak ada PR. Tanggal: 2026-10-07.

| Tugas | Commit | Isi |
|---|---|---|
| 0 | `624c676` | preflight: pytest, leakage, manifest indeks |
| 1 | `6d0ea61` | rekonsiliasi graf alur dengan PDF (44 langkah, `evidence`, `workflow_graph_sha256`) |
| 2 | `5d1f72a` | signifikansi multi-referensi (`--reference naive,graph`), Holm per (referensi, metrik), CI selisih berpasangan |
| 3 | `6d803ce` | lembar verifikasi benchmark + `apply_verification.py` |
| 4 | `23f08f0` | jejak hardware/OS/daya/Ollama/digest model di `run_config.json` |
| 5 | (commit terakhir) | uji asap `smoke_fullprep` + laporan ini |

Di luar repo: `"$PAPER_DIR/HUMAN_INPUTS.md"` (Tugas 4.2–4.3). Paper **tidak** diedit.

> **Catatan lingkungan.** Repo kerja adalah clone lokal `~/Documents/claude/rags`. Hanya clone ini yang memuat bobot `indo_finetuned_embedding_v3/model.safetensors` dan indeks `chroma_v3`/`chroma_base`, karena keduanya di-gitignore. Clone `~/Documents/claude/Chatbot-RAG-Comparison` tidak punya bobot, dan branch kerja yang sempat saya buat di sana sudah dihapus.
>
> Interpreter: `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` (3.12.5). `python` di shell menunjuk ke miniconda 3.13 tanpa dependensi, jadi untuk Goal B pakailah path interpreter ini.

---

## Tugas 0 — Persiapan

| Pemeriksaan | Hasil |
|---|---|
| `python -m pytest -q` (awal, d421b93) | 78 passed |
| `check-leakage --train-dir datasets/train_v3` (leksikal, ROUGE-L) | 40 soal vs 70 kandidat; maks 0,44; **0 item ≥ 0,5** |
| `check-leakage ... --semantic` (cosine, model base) | median 0,653, maks 0,819 (UKT-A1); **0 item ≥ 0,85** |
| `chroma_v3` | manifest valid; sha256 bobot v3 `327c99cf…` dan corpus hash `bf193de1…` cocok; koleksi 7 + 51 |
| `chroma_base` | manifest valid; sha256 base `228f63c6…` dan corpus hash cocok; koleksi 7 + 51 |

Indeks tidak dibangun ulang. Rincian: `docs/fullrun_prep/TUGAS0_preflight.md` dan output leakage di folder yang sama.

## Tugas 1 — Graf alur = PDF

**File:** `src/graph_sop/workflow_graph.py`, `docs/GRAPH_PDF_RECONCILIATION.md` (tabel per SOP), `docs/workflow_graph_v1.json` (graf lama, sha `e02511ff…`), `tests/test_workflow_graph_pdf.py` (12 tes), `evaluate_benchmark.py` (`workflow_graph_sha256` di `run_config.json`), `app.py` (label bila batas waktu tidak ada), `docs/ARCHITECTURE.md`.

**Ringkasan:**

| | Sebelum | Sesudah |
|---|---|---|
| simpul langkah | 42 | **44** (= PDF) |
| simpul aktor | 21 | 15 |
| sisi | 119 | 125 |
| langkah dengan pihak pelaksana diganti | – | **6**: Cuti 2, Cuti 3, UKT 2, UKT 3, IRS 4, Aktif 3 |
| nama aktor diseragamkan ke PDF | – | 12 |
| langkah dengan `inputs` / `output` / `duration` | 42 / 42 / 20 | 5 / 18 / 20 |
| sidik graf | `e02511ff…` | `8691220f…` |

- **UKT 9 → 11:** graf lama menggabungkan PDF 3+4 ("Mahasiswa menyerahkan …" + "Supervisor Sumber Daya memeriksa …") dan PDF 5+6 ("WD Sumber Daya memproses" + "menandatangani").
- **Contoh sebelum → sesudah (Cuti langkah 2):**
  - actor `Ketua Program Studi` → `Mahasiswa`;
  - action "Meminta persetujuan … dari Kaprodi" → "Meminta persetujuan dan tanda tangan Ketua Program Studi";
  - inputs `["Form Cuti Terisi & Berkas"]` → `[]`;
  - output "Form Cuti Ditandatangani Kaprodi" → "" (tidak ada di PDF);
  - evidence "Mahasiswa meminta persetujuan dan tanda tangan Ketua Program Studi."
- **Atribut karangan dihapus:** `inputs`/`output` lama hampir semuanya artefak antara yang tidak tertulis di PDF. Kini kedua field hanya diisi dari baris *Dokumen yang dibutuhkan* / *Output:* PDF. `max_duration` SOP Aktif "1 hari kerja" → `null` (PDF tidak menyebutnya). Durasi per langkah sudah benar sejak awal.
- **`evidence`** ada di setiap langkah, dan `max_duration_evidence` di setiap SOP. Tes memeriksa bahwa setiap kutipan muncul di teks PDF setelah normalisasi spasi.
- **Tes baru:**
  - jumlah langkah per SOP = langkah bernomor PDF, total 44;
  - `evidence` ada di PDF;
  - durasi dan output hanya dari baris PDF;
  - `get_workflow_context` Cuti menulis Mahasiswa untuk langkah 2 dan 3;
  - satu jalur `NEXT_STEP` linear (DAG) per SOP;
  - hash deterministik;
  - snapshot v1 tersimpan.
- **Keputusan yang ragu, dengan opsi** (rinci di §3 dokumen rekonsiliasi):
  1. IRS 4: subjek kalimat pertama Mahasiswa, kalimat kedua Pembimbing. Dipilih Mahasiswa; opsi lain Pembimbing Akademik.
  2. Aktif 3: kalimat keadaan; subjek Mahasiswa.
  3. Proposal 4: kalimat tanpa verba.
  4. Cuti 3: disposisi pasif tanpa subjek tetap di `action`.
  5. `inputs`/`output` ketat (hanya field eksplisit PDF) vs diturunkan dari kalimat aksi.
  6. Usulan field `counterpart` **tidak ditambahkan**; untung-ruginya ditulis di sana.
  7. `evidence` tidak dimasukkan ke konteks LLM.
- **Dampak samping (bukan penyetelan):** `match_sop` (tidak diubah) memberi skor dari kata di `action`, dan kueri vektor `workflow` ditambah `action` tiga langkah pertama. Jadi pemilihan SOP dan chunk sistem `workflow` dapat sedikit berubah karena teks `action` kini mengikuti PDF.
- **Graf vs `procedures` benchmark** (hanya dicatat): jumlah langkah ketujuh prosedur sama. Aktor berbeda di 3 langkah, dan pada ketiganya emas = pejabat yang dituju, graf = Mahasiswa (yang juga ada di `also_ok`):
  - CUTI 2 (emas Ketua Program Studi);
  - IRS 4 (emas Dosen Wali);
  - UKT 2 (emas Dosen Wali + Ketua Program Studi).

**Cara menjalankan:** `python -m pytest tests/test_workflow_graph_pdf.py -q`.

## Tugas 2 — Uji signifikansi terhadap > 1 referensi

**File:** `evaluate_benchmark.py` (`parse_references`, `paired_significance`, `significance_table_tex`), `tests/test_significance_multi_ref.py` (8 tes), `docs/EVALUATION_PROTOCOL.md`.

- **Keluarga Holm sebelumnya** adalah satu (referensi, metrik): semua sistem non-referensi untuk satu metrik dikoreksi bersama. Definisi ini dipertahankan dan kini berlaku per referensi. Referensi kedua menambah blok baru tanpa mengubah blok pertama. Tidak ada koreksi lintas metrik atau lintas referensi; ini ditulis di `EVALUATION_PROTOCOL.md`.
- `--reference naive,graph` menghasilkan dua blok di `significance.csv` (kolom `reference` sudah ada sebelumnya). Kolom baru `mean_diff_ci_lo` dan `mean_diff_ci_hi` berisi 95% bootstrap CI selisih rata-rata berpasangan (10.000 resample selisih per soal, seed 42), di samping `mean_diff`, `median_diff`, `p_wilcoxon`, dan `p_holm`.
- Tabel baru `significance_table.tex`: selisih berpasangan + CI, satu blok per referensi. `$^{*}$` menandai `p_holm < 0,05` di dalam blok itu, jadi signifikansi terhadap Naive dan terhadap GraphRAG terlihat terpisah. `summary_table.tex` tidak diubah.
- **Default tidak berubah.** Pada data sintetis, semua kolom lama identik byte demi byte dengan salinan verbatim kode d421b93. Pada data nyata, saya menskor ulang salinan `smoke_fix2`: kolom lama `significance.csv` identik dengan file asli. Salinan sementara itu lalu dihapus.
- **Keputusan yang perlu Anda ketahui:** "file identik" saya penuhi untuk semua kolom lama; dua kolom CI ditambahkan di **akhir**. Kalau Anda butuh file yang benar-benar identik byte demi byte untuk default, opsinya memindahkan CI ke file terpisah (mis. `significance_ci.csv`). Untungnya file lama tidak tersentuh; ruginya CI tidak lagi berdampingan dengan p-value.
- **Tes:**
  - dua referensi menghasilkan dua blok independen;
  - blok pertama identik dengan run satu-referensi;
  - Holm dihitung manual per blok (p mentah dipatok lewat monkeypatch: blok Naive → 0,03/0,06/0,06; blok Graph → 0,03/0,04/0,50), berbeda dari Holm gabungan 6;
  - CI deterministik dan mengapit rata-rata;
  - tanda bintang tabel terpisah per blok;
  - referensi yang tidak ada di run dilewati.

**Cara menjalankan:** `python evaluate_benchmark.py score --run-name <run> --reference naive,graph`.

## Tugas 3 — Lembar verifikasi benchmark

**File:** `scripts/make_verification_sheet.py`, `scripts/apply_verification.py`, `benchmark/verification_sheet_v1.csv` (40 baris, kolom manusia kosong), `tests/test_verification_sheet.py` (15 tes), `docs/EVALUATION_PROTOCOL.md`.

- **Kolom lembar:**
  - isi soal: `id, category, sop, answerable, question, reference, must_include`;
  - pembanding: `langkah_emas` (aktor + `also_ok`), `langkah_pdf` (subjek kalimat PDF per langkah), `kutipan_pdf`, `auto_flags`;
  - integritas: `item_sha256` (sidik isi soal + prosedur emasnya);
  - isian manusia: `verified (y/n), perlu_koreksi, catatan, verifikator`.
- **`kutipan_pdf`:** untuk soal prosedural berisi seluruh langkah PDF. Untuk soal lain berisi segmen PDF dengan peringkat dari fakta `must_include` dan kata yang sama dengan soal/referensi; soal lintas-SOP mendapat minimal satu segmen per SOP. Kutipan hanya **petunjuk**: untuk soal lintas-SOP (mis. X-3), bacalah PDF lengkapnya.
- **Format:** UTF-8 dengan BOM dan pemisah koma. Numbers membukanya langsung. Excel dengan pengaturan regional Indonesia memakai `;` sebagai pemisah daftar, jadi bila semua isi masuk satu kolom, pakai *Data → From Text/CSV*. Pembaca di `apply_verification.py` menerima `,` maupun `;`.
- **`apply_verification.py`:**
  - menyetel `verified: true`, `verified_by`, dan `verified_at` hanya untuk baris `y` dengan `perlu_koreksi` kosong (`-`, `n`, `tidak`, `t` dianggap kosong; ini keputusan saya, mohon dicek);
  - **menolak menulis apa pun** bila ada baris `y` yang juga perlu koreksi, baris `y` tanpa `verifikator`, nilai y/n yang tidak dikenali, id ganda/tidak dikenal, atau `item_sha256` yang berubah (soal diedit setelah lembar dibuat; buat ulang dengan `--merge`);
  - hanya mengubah baris item yang bersangkutan, sehingga tata letak JSON yang ditulis tangan tetap utuh, lalu memeriksa ulang bahwa tidak ada isi lain yang berubah;
  - tidak pernah menyetel `verified` kembali ke false.
- `make_verification_sheet.py --merge` membuat ulang lembar sambil mempertahankan isian manusia. Untuk item yang isinya berubah, `verified` dikosongkan dan diberi catatan.
- **Benchmark tidak berubah:** `git diff` kosong dan `"verified": true` muncul 0 kali.

**Ringkasan `auto_flags`:** 6 dari 40 soal ber-flag, semuanya `AKTOR_EMAS_BEDA_SUBJEK`, dan semuanya berasal dari 3 langkah emas yang sama:

| Soal | Langkah | Aktor emas | `also_ok` | Subjek kalimat PDF |
|---|---|---|---|---|
| CUTI-P1, CUTI-P2 | 2 | Ketua Program Studi | Mahasiswa | Mahasiswa |
| IRS-P1, IRS-P2 | 4 | Dosen Wali | Mahasiswa | Mahasiswa |
| UKT-P1, UKT-P2 | 2 | Dosen Wali, Ketua Program Studi | Mahasiswa | Mahasiswa |

Tidak ada flag `FAKTA_TIDAK_DI_PDF`, `ANGKA_TIDAK_DI_PDF`, `JUMLAH_LANGKAH`, atau `KW_TIDAK_DI_LANGKAH_PDF`. Artinya semua fakta `must_include` dan angka waktu di referensi cocok secara leksikal dengan teks PDF, dan semua grup kata kunci langkah emas ada di teks langkah PDF-nya. Ini pemeriksaan mekanis, **bukan** pengganti verifikasi manusia: makna, kelengkapan referensi, dan soal tak-terjawab tetap perlu dibaca.

Keputusan yang dibutuhkan dari verifikator untuk 6 soal ini: apakah "aktor" di benchmark berarti *pihak yang melakukan* (subjek PDF, seperti graf) atau *pejabat yang menyetujui/menandatangani*. Benchmark sengaja tidak saya ubah.

**Cara menjalankan:** `python scripts/make_verification_sheet.py` (sudah dijalankan), lalu `python scripts/apply_verification.py --dry-run` / tanpa `--dry-run`.

## Tugas 4 — Jejak hardware dan model

**File:** `evaluate_benchmark.py` (`_host_info`, `_ollama_model_info`, `_parse_ollama_list`, `_cmd_out`), `tests/test_run_provenance.py` (4 tes), `"$PAPER_DIR/HUMAN_INPUTS.md"`.

- `run_config.json` kini memuat:
  - `host`: cpu, core logis, RAM, OS + build, arsitektur, Python + path interpreter, `power_source`;
  - `ollama_cli_version` (di samping `ollama_version` dari API);
  - `ollama_models`: digest sha256 penuh, ukuran, kuantisasi, dan jumlah parameter dari `/api/tags` (cadangan: ID pendek dari `ollama list`) untuk generator dan judge.
- `score --ragas` mencatat `judge_model_info` di `score_config.json`.
- Semua field bernilai `null` bila tidak tersedia; tidak ada yang membuat run gagal. `ollama_version` kini `null`, bukan `"unknown"`, saat API mati.
- `HUMAN_INPUTS.md` ada di folder paper. Isinya: jurnal + gaya referensi, pendanaan/nomor hibah, verifikator benchmark/graf/audit, ucapan terima kasih, serta CRediT/konflik kepentingan/deklarasi AI. Juga ada 4 kandidat jurnal Elsevier (NLP Journal, IP&M, ESWA, ISWA) dengan cakupan, alasan, dan tautan resmi. Dua tautan ScienceDirect tidak bisa dicek otomatis (403 untuk bot) dan ditandai begitu.

## Tugas 5 — Uji asap

**Perintah** (dari working tree bersih, commit `23f08f0`):

```
python evaluate_benchmark.py generate --run-name smoke_fullprep --embedding-model v3 --systems naive,graph,workflow --limit 3
python evaluate_benchmark.py score --run-name smoke_fullprep --reference naive,graph      # tanpa Ragas
```

**`results/smoke_fullprep/run_config.json`** memuat semua field yang diminta:

| Field | Nilai |
|---|---|
| `git_dirty` | `false` (commit `23f08f0`) |
| `embedding_weights_sha256` | `327c99cf…` (v3) |
| `index_manifest` | ada; corpus hash `bf193de1…` |
| `workflow_graph_sha256` | `8691220f…` |
| `ollama_version` / `ollama_cli_version` | 0.35.1 |
| `ollama_models` | `gemma4:e2b` digest `7fbdbf8f5e45…` (Q4_K_M, 5.1B); `llama3.1:8b`: `null` (belum di-pull) |
| `host` | Apple M5, 10 core, 24 GB, macOS 26.6.2 (25G83), arm64, Python 3.12.5, `power_source: "Battery Power"` |

Generate: 9/9 baris tanpa error LLM, 2 menit 8 detik termasuk memuat model.

### Hasil — **uji asap, 3 soal, bukan hasil** (tidak boleh masuk paper)

Soal: CUTI-P1, CUTI-P2, CUTI-A1, semuanya dari SOP Cuti dan belum diverifikasi. Dengan n = 3, uji Wilcoxon tidak dijalankan (butuh ≥ 6 pasangan), sehingga `p_wilcoxon`/`p_holm` kosong. CI bootstrap dari 3 titik tidak bermakna. Angka di bawah hanya membuktikan bahwa pipeline berjalan ujung ke ujung.

| sistem | KFR | R-L | StepCov | latensi median (s) | token prompt rata-rata |
|---|---|---|---|---|---|
| naive | 0,333 | 0,161 | 0,062 | 11,5 | 820 |
| graph | 0,467 | 0,191 | 0,062 | 10,5 | 850 |
| workflow | 0,783 | 0,284 | 0,375 | 16,2 | 1.699 |

`significance.csv` berisi dua blok (`reference = naive` dan `reference = graph`) dengan kolom CI. `significance_table.tex` juga terbentuk dengan dua blok.

### Temuan uji asap yang perlu Anda putuskan sebelum Goal B (tidak saya ubah)

1. **Jawaban prosedural terpotong oleh `max_tokens = 1024`.**
   - Keempat jawaban prosedural yang memanggil LLM (naive/graph/workflow CUTI-P1, workflow CUTI-P2) berhenti tepat di 1.024 token keluaran, di tengah kalimat. Contoh: jawaban `workflow` CUTI-P2 terpotong di langkah 5 dari 8.
   - Akibatnya step coverage dan KFR soal prosedural akan terukur lebih rendah karena pemotongan, bukan karena isi.
   - `max_tokens` tidak ada di daftar kontrol yang dikunci, tetapi mengubahnya adalah keputusan desain eksperimen.
   - Opsi A: biarkan 1.024 dan laporkan jumlah jawaban yang mentok (`completion_tokens == max_tokens`) sebagai keterbatasan.
   - Opsi B: naikkan, mis. `--max-tokens 2048`, untuk **semua** sistem dalam satu run. Ini menambah latensi; nilainya tercatat di `run_config.json`.
   - Rekomendasi saya B, diputuskan sebelum run, bukan setelah melihat skor.
2. **Naive dan GraphRAG menolak CUTI-P2 tanpa memanggil LLM** (`no_llm_call`, 0,06–0,09 s): skor kosinus terbaik di bawah ambang 0,3 untuk frasa informal mahasiswa. Workflow menjawab karena `match_sop` memilih SOP Cuti. Ini perilaku sistem yang sudah dijelaskan di paper, bukan bug. Laporkan sebagai bagian dari *false refusal*.
3. **GraphRAG = Naive pada CUTI-A1:** token prompt identik (1.270) dan jawaban identik, jadi baseline graf entitas tidak menambahkan relasi apa pun untuk soal ini. Perlu diperhatikan saat menafsirkan perbandingan Workflow vs GraphRAG. Saya tidak menyelidiki lebih jauh karena baseline di luar cakupan.

### Perkiraan durasi run penuh (dari latensi uji asap; kasar)

- Latensi per kueri yang memanggil LLM: 10,5–17,3 s (rata-rata ≈ 14 s, di baterai). Soal yang mentok di 1.024 token ≈ 17 s.
- **Generate**, 40 soal × 9 sistem (naive, hybrid, graph, agentic, crag, multimodal, workflow, llm_only, full_context):

  | Bagian | Perkiraan |
  |---|---|
  | 6 sistem satu-panggilan | ≈ 40 × 6 × 14 s ≈ 56 menit |
  | full_context (prompt ±3–4× lebih panjang) | ≈ 40 × 20 s ≈ 13 menit |
  | agentic (hingga 3 langkah ReAct + sintesis) | ≈ 40 × 45 s ≈ 30 menit |
  | CRAG (kadang + rewrite) | ≈ 40 × 20 s ≈ 13 menit |
  | **Total** | **≈ 2 jam** (rentang 1,5–3 jam) |

  Jika `--max-tokens 2048`, tambahkan ± 30–50 %.
- **Ablation embedding `base`:** run kedua dengan ukuran sama, ≈ 2 jam lagi bila semua sistem diulang.
- **Ragas** dengan `llama3.1:8b` (tiga metrik, beberapa panggilan judge per jawaban): perkiraan kasar 30–90 s per jawaban × 360 ≈ **3–9 jam**. Belum diukur karena judge belum terpasang.
- **Total realistis:** ±1 hari kerja mesin. Jalankan semalam dengan charger dan `caffeinate -i`. `generate` dan Ragas bisa dilanjutkan (resumable) bila terhenti.


---

## Status definisi selesai

- `pytest` akhir: **117 passed**. Rinciannya: 78 tes lama tidak diubah, 12 tes graf (Tugas 1), 8 tes signifikansi (Tugas 2), 15 tes lembar verifikasi (Tugas 3), dan 4 tes provenance (Tugas 4).
- Leakage: 0 item di atas ambang, baik leksikal maupun semantik.
- Tidak ada run penuh; satu-satunya generate adalah uji asap 3 soal.
- Tidak ada klaim hasil, dan paper tidak diedit.
- `benchmark/sop_benchmark_v1.json` tidak berubah dan tidak ada `verified: true`.
- `results/smoke_fix*`, model, dan indeks lama tetap ada.

## Daftar yang harus Anda kerjakan sebelum Goal B

- [ ] **Isi `benchmark/verification_sheet_v1.csv`**: 40 baris, kolom `verified (y/n)`, `perlu_koreksi`, `catatan`, `verifikator`.
  - Putuskan 6 soal ber-flag aktor (lihat Tugas 3).
  - Koreksi isi soal dilakukan langsung di `benchmark/sop_benchmark_v1.json`, lalu jalankan `python scripts/make_verification_sheet.py --merge`.
- [ ] **Jalankan** `python scripts/apply_verification.py --dry-run`, lalu tanpa `--dry-run`. Commit benchmark yang sudah diverifikasi.
- [ ] **Periksa rekonsiliasi graf** di `docs/GRAPH_PDF_RECONCILIATION.md`, terutama §3:
  - IRS 4 (Mahasiswa vs Pembimbing Akademik);
  - Aktif 3;
  - Proposal 4;
  - kebijakan `inputs`/`output` ketat;
  - usulan `counterpart`.

  Jika ada yang Anda ubah, sidik graf berubah dan run penuh harus memakai graf final (`workflow_graph_sha256` tercatat).
- [ ] **Isi `HUMAN_INPUTS.md`** di folder paper.
- [ ] **Siapkan Ollama:**
  - `gemma4:e2b` sudah ada (digest `7fbdbf8f…`).
  - **`llama3.1:8b` BELUM ADA** di `ollama list`. Jalankan `ollama pull llama3.1:8b` (± 4,9 GB) sebelum skor Ragas. Saya tidak mengunduhnya karena Goal A tidak memakai Ragas dan pengunduhan perlu izin Anda.
- [ ] **Colokkan charger dan cegah Mac tidur.** Saat uji asap, Mac **berjalan dengan baterai** (`power_source: Battery Power`). Latensi pada baterai bisa berbeda dari listrik, dan angka latensi masuk paper. Jalankan run penuh dengan charger terpasang dan `caffeinate -i` (mis. `caffeinate -i python …`).
- [ ] **Pakai interpreter yang benar** untuk semua perintah: `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` dari `~/Documents/claude/rags`, dengan working tree bersih (`git_dirty: false`).

## Usulan perubahan teks paper (BELUM diterapkan)

Semua usulan di bawah berlaku untuk `paper_concise.tex`. Angka bertanda ⟨…⟩ diisi setelah Goal B.

1. **Tabel 2 (`tab:corpus`), kolom "Graph steps":** baris UKT `9` → `11`, Total `42` → `44`. Semua baris kini sama dengan "PDF steps". Caption bisa ditambah: *"After reconciliation, the graph has one step node per numbered step."*
2. **§4.1, paragraf baris 225** (mengganti kalimat jumlah simpul dan `\todo`):
   > The graph was built by hand from the PDF text and is stored with NetworkX. It contains 7 root nodes, 44 step nodes (one per numbered step in the PDFs) and 15 actor nodes, connected by 125 edges. The actor of a step is the party that performs it, i.e.\ the grammatical subject of the PDF sentence; an official who is only addressed, such as the head of study programme whose signature the student requests, is named in the action text. Input documents, output and duration are filled in only where the PDF states them (5, 18 and 20 of the 44 steps). Each step also stores a verbatim quote of its PDF sentence, and an automated test checks that every quote occurs in the source PDF and that the step counts match. ⟨Verifier⟩ checked the graph against the source documents.
3. **§4.1, daftar simpul:**
   - root: "the maximum processing time" → "the maximum processing time, when the SOP states one (6 of 7)";
   - step: "with five attributes: actor, action, inputs (required documents), output and duration" → "with an actor and an action, and, where the SOP states them, inputs (required documents), output and duration".
4. **Gambar 3 (`fig:context`, SOP Aktif)** tidak lagi cocok dengan graf. Usulan isi baru:
   - "Maximum processing time: not stated in the SOP";
   - langkah 1: Actor Student; documents "Leave permit letter, copy of student card (KTM)"; Duration ±5 minutes; tanpa baris Output;
   - langkah 3: "Actor: Student / Action: Is registered as a course participant / active student of the faculty".

   Alternatif: ganti contoh dengan SOP Cuti (langkah 2–3), yang lebih jelas menunjukkan aktor = pelaksana dan pejabat dalam aksi.
5. **§3.2 "What a correct answer needs"** (baris ~157): "says who acts at each step (the student, the head of study programme, the Dean, …)" → "says who acts at each step (the student, the academic office, the Dean, the resources office) and whose approval is needed (the head of study programme)". Pada SOP Cuti, Kaprodi kini bukan pelaksana langkah.
6. **Abstrak:** "whose nodes are steps annotated with an actor, an action, input documents and an output" → tambahkan "where the SOP states them". Opsional.
7. **§5 Statistik (baris 378):** ganti `\todo` dengan
   > Holm correction is applied separately to the comparisons against naive RAG and against GraphRAG; each (reference, metric) pair is one family. For each pair we also report the paired mean difference with a 95\% bootstrap CI (10,000 resamples).
8. **Hardware (baris 331):** isi dari `run_config.json` run final. Saat ini host-nya Apple M5, 24 GB RAM, macOS 26.6.2, Ollama 0.35.1, `gemma4:e2b` Q4_K_M digest `7fbdbf8f…`. Sebutkan juga bahwa run berjalan dengan daya listrik.
9. **Definisi aktor di benchmark (baris 334 / §5):** setelah verifikator memutuskan 6 soal ber-flag, sebutkan definisi "aktor" yang dipakai benchmark. Bila benchmark memakai pejabat penyetuju sedangkan graf memakai pelaksana, jelaskan bahwa `also_ok` menerima keduanya.
