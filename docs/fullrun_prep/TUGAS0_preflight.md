# Goal A — Tugas 0: Persiapan (preflight)

Tanggal: 2026-10-07. Branch `exp/full-run-v1` dibuat dari `fix/followup-review-findings` (`d421b93`). Working tree bersih.

Repo kerja: clone lokal `~/Documents/claude/rags` (clone ini yang memuat bobot `indo_finetuned_embedding_v3/model.safetensors` dan indeks `chroma_v3`/`chroma_base`; keduanya di-gitignore).
Interpreter: `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` (Python 3.12.5, sama dengan `smoke_fix2`).

| Pemeriksaan | Perintah | Hasil |
|---|---|---|
| Tes unit | `python -m pytest -q` | **78 passed** (7,9 s) |
| Leakage leksikal | `python evaluate_benchmark.py check-leakage --train-dir datasets/train_v3` | 40 soal vs 70 kandidat (ROUGE-L F1); maks 0,44 (CUTI-T1); **0 item ≥ 0,5** |
| Leakage semantik | `... check-leakage --train-dir datasets/train_v3 --semantic` | cosine `all-indo-e5-small-v4`; median 0,653, maks 0,819 (UKT-A1); **0 item ≥ 0,85** |
| Indeks `chroma_v3` | `verify_index_manifest` + hash ulang | valid; sha256 bobot v3 `327c99cf…` cocok; corpus hash `bf193de1…` cocok; koleksi langchain 7, htree_leaves 51 (live = manifest) |
| Indeks `chroma_base` | idem | valid; sha256 bobot base `228f63c6…` cocok; corpus hash cocok; koleksi 7/51 |

Indeks tidak dibangun ulang (tidak perlu). Output lengkap leakage: `check_leakage_lexical.txt`, `check_leakage_semantic.txt`.
