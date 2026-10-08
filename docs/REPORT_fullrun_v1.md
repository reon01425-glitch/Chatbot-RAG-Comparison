# Laporan Goal B — Run penuh dan pengisian paper

Branch `exp/full-run-v1` (lanjutan Goal A). Working tree bersih (`git_dirty: false`) di setiap tahap generate/score. Interpreter: `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`. Tanggal: 2026-10-07/08 (WIB).

## Perintah persis yang dijalankan

```
# Tugas 0: pre-flight (lihat juga docs/REPORT_fullrun_prep.md Goal A)
python -m pytest -q
python evaluate_benchmark.py check-leakage --train-dir datasets/train_v3
python evaluate_benchmark.py check-leakage --semantic --train-dir datasets/train_v3
pip install "ragas==0.4.3"   # sudah tercantum di requirements.txt/requirements-mac.txt, hanya belum terpasang

# Tugas 1: run utama
python evaluate_benchmark.py generate --run-name full_v3_20261008 --embedding-model v3 \
  --systems naive,graph,workflow,hybrid,agentic,crag,multimodal,llm_only,full_context --require-verified
python evaluate_benchmark.py score --run-name full_v3_20261008 --reference naive,graph --require-verified
python evaluate_benchmark.py score --run-name full_v3_20261008 --reference naive,graph \
  --ragas --judge-model llama3.1:8b --require-verified   # dijalankan ulang berkali-kali (lihat di bawah), resume otomatis lewat ragas_cache.jsonl

# Tugas 2: ablation embedding base
python evaluate_benchmark.py generate --run-name full_base_20261008 --embedding-model base \
  --systems naive,graph,workflow --require-verified
python evaluate_benchmark.py score --run-name full_base_20261008 --reference naive,graph --require-verified

# Tugas 3/5: tabel paper dan lembar audit
python scripts/make_paper_tables.py --run full_v3_20261008 --ablation-run full_base_20261008
python scripts/make_audit_sheet.py --run full_v3_20261008
python scripts/audit_agreement.py --run full_v3_20261008   # dijalankan sekarang hanya untuk uji logika; lembar manusia masih kosong
```

`--retry-errors` **tidak pernah diperlukan**: `n_errors` (status `llm_error`/`exception`) adalah 0 di kedua run, untuk semua sistem.

## Durasi

| Tahap | Mulai | Selesai | Durasi |
|---|---|---|---|
| Generate `full_v3_20261008` (9 sistem × 40 soal) | 2026-10-08 00:32:48 | 2026-10-08 01:48:25 | ≈ 1 jam 16 menit |
| Score `full_v3_20261008` tanpa Ragas | 2026-10-08 01:49 | — | < 1 menit |
| Generate `full_base_20261008` (3 sistem × 40 soal, ablation) | 2026-10-08 02:16:37 | 2026-10-08 02:46:55 | ≈ 30 menit (termasuk satu kali kill/resume saat berjalan bersamaan dengan Ragas, lihat di bawah) |
| Score `full_base_20261008` | 2026-10-08 02:47:15 | — | < 1 menit |
| Score `full_v3_20261008` **dengan** `--ragas` | mulai 2026-10-08 ≈01:49 | selesai 2026-10-08 09:01:04 | ≈ 7 jam elapsed wall-clock, dengan **9 kali proses dibunuh dan di-resume** dari `ragas_cache.jsonl` (lihat "Hal yang mengejutkan") |

Total durasi mesin ≈ 9 jam, sesuai kisaran perkiraan Goal A (2 jam generate + 3–9 jam Ragas).

## Jumlah error per sistem

0 untuk semua sistem, di kedua run (`full_v3_20261008` dan `full_base_20261008`). Tidak ada baris berstatus `llm_error` atau `exception`. Ada baris berstatus `no_llm_call` (penolakan berbasis skor retrieval, sebelum memanggil LLM) — ini bukan error menurut definisi skrip, dan jumlahnya menarik secara substantif (lihat di bawah), bukan tanda kegagalan.

| Sistem | `no_llm_call` (dari 40) — `full_v3_20261008` |
|---|---|
| naive, graph, hybrid, multimodal | 10 |
| crag | 18 |
| **workflow** | **2** (keduanya soal `unanswerable`, benar) |
| agentic, llm_only, full_context | 0 |

## n terverifikasi per kategori

Semua 40 soal benchmark **sudah** `verified: true` sebelum run ini dimulai (diselesaikan sebagai pekerjaan susulan Goal A, 2026-10-07, `verified_by: "manusia"`):

| Kategori | n total | n terverifikasi |
|---|---|---|
| Procedural | 11 | 11 |
| Actor | 7 | 7 |
| Documents | 7 | 7 |
| Time/location | 7 | 7 |
| Cross-SOP | 3 | 3 |
| Unanswerable | 5 | 5 |
| **Total** | **40** | **40** |

`--require-verified` dipakai di semua `generate`/`score`; karena 40/40 terverifikasi, tidak ada n yang dikurangi.

## Ringkasan hasil utama (apa adanya)

Angka lengkap: `results/full_v3_20261008/summary.csv`, `significance.csv`, `paper_tables/*.tex`. Tabel di bawah hanya ringkasan.

| Sistem | KFR | Step cov. | TIR | AMR | Refusal acc. | Prompt tok. |
|---|---|---|---|---|---|---|
| Naive RAG | 0.712 | 0.727 | 0.000 | 0.000 | 0.600 | 916 |
| GraphRAG (entity) | 0.698 | 0.687 | 0.083 | 0.000 | 0.600 | 949 |
| **Workflow GraphRAG** | **0.955** | **1.000** | 0.000 | 0.000 | **1.000** | 1,525 |
| Hybrid RAG | 0.683 | 0.712 | 0.000 | 0.000 | 0.600 | 882 |
| Agentic RAG | 0.712 | 0.702 | 0.000 | 0.000 | 0.600 | 1,815 |
| Corrective RAG | 0.569 | 0.636 | 0.000 | 0.000 | 1.000 | 725 |
| Multimodal RAG | 0.702 | 0.702 | 0.000 | 0.000 | 0.600 | 994 |
| LLM only | 0.113 | 0.105 | 0.333 | 0.444 | 0.000 | 107 |
| Full context | 0.962 | 0.944 | 0.027 | 0.000 | 1.000 | 2,722 |

- Workflow GraphRAG menang signifikan atas GraphRAG untuk KFR (Holm $p=0.034$) dan ROUGE-L (Holm $p=0.038$), **tidak** signifikan untuk step coverage (Holm $p=0.438$).
- Workflow GraphRAG **tidak** menang signifikan atas Naive RAG untuk KFR (Holm $p=0.053$, di atas ambang 0,05) meski selisih mentahnya besar (+0,243). Ini ditulis apa adanya di paper, bukan dibulatkan jadi "signifikan".
- **Workflow GraphRAG tidak menang di semua metrik**: TIR dan AMR hampir 0 untuk semua sistem RAG (bukan keunggulan khusus Workflow), dan biaya token Workflow (1.525) lebih tinggi dari lima dari enam pembanding lain (semua kecuali Agentic dan Full context).
- Ablation embedding `base` vs `v3`: **temuan mengejutkan** — Naive RAG dan GraphRAG justru lebih tinggi KFR dengan embedding `base` (0,910 dan 0,900) dibanding `v3` yang di-fine-tune (0,712 dan 0,698), sedangkan Workflow GraphRAG tidak berubah (0,955 keduanya). Tidak ada penjelasan pasti; diduga fine-tuning pada 50 pasang tanya-jawab overfit ke frasa pasangan itu sendiri. Ditulis di paper §7 sebagai pertanyaan terbuka, bukan diklaim sebagai temuan final.
- Ragas (sekunder, judge `llama3.1:8b`): Workflow GraphRAG tertinggi di faithfulness (0,805) dan answer relevancy (0,682) di antara tiga sistem utama, konsisten dengan metrik deterministik. Judge gagal mem-parse keluaran JSON untuk sebagian item (n efektif 35/40 untuk Workflow, 26–27/40 untuk Naive/GraphRAG) — dilaporkan sebagai pendukung, bukan konfirmasi independen dengan n yang sama.

## `\todo` yang tersisa di paper

Hanya 2, keduanya menunggu Goal C (audit manual):
1. Baris §5.4 (`\textbf{tir targets...}` paragraf Metrics): agreement rate audit manual.
2. §6.4 Error analysis: contoh ketiga dari audit penyelarasan langkah manual.

Semua `\todo`/`\tbd` lain (jurnal, pendanaan, ucapan terima kasih, hardware, verifikasi benchmark/graf, statistik, Tabel 5, per-kategori, refusal, cost, error analysis #1–2, ablation, kesimpulan, data availability, tanggal akses model card) sudah diisi dari `HUMAN_INPUTS.md` dan `results/full_v3_20261008/`.

## Status kompilasi LaTeX

**`pdflatex` tidak terpasang** di mesin ini. Sesuai aturan, saya tidak memasangnya tanpa izin. Sebagai gantinya saya jalankan `validate_latex.py` (skrip yang sudah ada di folder paper) yang memeriksa keseimbangan `\begin`/`\end` dan bahwa semua `\ref`/`\label`/`\input` cocok — semuanya lolos. **Jumlah halaman dan rendering visual belum diverifikasi** karena tidak ada compiler. Mohon dikompilasi manual (`pdflatex` 3× dari folder paper) sebelum submit, dan periksa jumlah halaman (target 15–20) serta tidak ada `??`.

## Hal yang mengejutkan atau mencurigakan

1. **Proses `score --ragas` dibunuh berulang kali oleh environment (9×), bukan oleh beban kerja aktual.** Penyebab pertama (kill #1) nyata: pemanggilan `ChatOllama` untuk judge tidak menyetel `num_ctx`, sehingga Ollama memuat context window penuh `llama3.1:8b` (131.072 token) — ±23 GB RAM, memicu swap berat dan proses di-OOM-kill. **Ini dilaporkan sebagai bug infrastruktur dan diperbaiki** (`--judge-num-ctx`, default 8192; lihat commit `fix(score): bound Ragas judge context window`), bukan di-akali, karena ini bukan parameter retrieval/generation yang dikunci oleh aturan. Setelah perbaikan, proses tetap dibunuh berulang (siklus ±30–70 menit) walau memori sehat (±8,5 GB wired, tidak ada lonjakan) — pola ini konsisten dengan batas waktu proses background di environment, bukan OOM asli. Skor tetap benar karena `_run_ragas` resumable lewat `ragas_cache.jsonl`; setiap `relaunch` melanjutkan dari cache, tidak mengulang dari nol.
2. **Proses `llama-server` yang ditinggalkan (orphan) menumpuk memori antar siklus.** Saat satu siklus di-kill, proses `llama-server` Ollama di baliknya terkadang tidak langsung dibebaskan (ditemukan satu instance bertahan 50+ menit, 8,1 GB RSS, setelah klien Python-nya sudah mati). Ini yang membuat laptop "terasa berat" meski CPU idle tinggi. Solusi: `pkill -f Ollama.app` + buka ulang aplikasi Ollama di antara siklus membersihkan ini (dari 1,4 GB RAM bebas menjadi 11 GB bebas).
3. **Ablation embedding base > v3 untuk Naive/GraphRAG** (lihat di atas) — berlawanan dengan asumsi bahwa fine-tuning selalu membantu retrieval. Tidak diselidiki lebih lanjut (di luar cakupan Goal B); ditulis sebagai pertanyaan terbuka di paper.
4. **Workflow GraphRAG jauh lebih tahan terhadap frasa informal/tak-baku** dibanding baseline lain: hanya 2/40 `no_llm_call` (keduanya benar, soal `unanswerable`), dibanding 10–18/40 untuk sistem lain. Ini karena `match_sop` Workflow tidak bergantung pada ambang kosinus retrieval untuk memutuskan menjawab, sedangkan sistem lain menolak begitu similarity top-3 di bawah 0,3 — termasuk untuk soal yang sebenarnya terjawabkan tapi frasanya informal (`CUTI-P2`, dst., dicatat juga di uji asap Goal A).
5. **Corrective RAG menolak 37% soal yang sebenarnya terjawabkan** (false-refusal rate 0,371) meski refusal accuracy-nya 1,000 untuk soal tak-terjawab — ambang pemeringkatannya (0,35) lebih ketat dari ambang 0,3 yang dipakai sistem lain, sehingga ia "menolak dengan percaya diri" pada soal yang salah.
6. **`ragas==0.4.3` memakai API yang sudah deprecated** relatif terhadap versi `ragas` yang dipanggil kode (`DeprecationWarning` untuk `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper`/import langsung dari `ragas.metrics`), dan sering gagal mem-parse keluaran JSON `llama3.1:8b` (`OutputParserException`/`RagasOutputParserException`) — ditangani oleh kode yang sudah ada (`raise_exceptions=False`, dicatat sebagai `NaN` per item), tapi menjelaskan mengapa `faithfulness_n`/`context_recall_n` lebih kecil dari 40 untuk sebagian sistem.

## Catatan commit dan push

- `scripts/make_paper_tables.py`, `scripts/make_audit_sheet.py`, `scripts/audit_agreement.py` sudah di-commit terpisah dari hasil run.
- Perbaikan `--judge-num-ctx` sudah di-commit terpisah, sebelum hasil run dengan Ragas yang berhasil.
- `results/full_v3_20261008/` dan `results/full_base_20261008/` (termasuk `paper_tables/`, `audit_sheet.csv`, `pre_ragas_backup/`) di-commit utuh pada commit laporan ini.
- Paper (`"$PAPER_DIR/paper_concise.tex"` dan `paper_tables/`) berada **di luar** repo ini sesuai instruksi; salinan template sebelum edit tersimpan di `paper_concise_v1_template.tex`.
- Branch `exp/full-run-v1` di-push ke origin setelah commit ini (tidak ada PR, tidak ada force-push).

## Yang masih perlu Anda kerjakan (Goal C)

1. Isi kolom manusia di `results/full_v3_20261008/audit_sheet.csv` (60 baris, 3 sistem utama) — baca jawaban lengkap di `audit_sheet_answers.txt` untuk konteks.
2. Jalankan `python scripts/audit_agreement.py --run full_v3_20261008` setelah lembar terisi, untuk menghitung Cohen's kappa dan AMR dari label manusia.
3. Isi 2 `\todo` terakhir di paper dari hasil langkah di atas.
4. Kompilasi `pdflatex` 3× dan periksa jumlah halaman (LaTeX belum terpasang di mesin ini).
