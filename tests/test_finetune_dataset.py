import os
import pytest
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from finetune_embeddings import load_all_dataset_examples, load_qa_pairs, DATASET_DIR


def test_qa_pairs_format_valid():
    """Verify that all JSON training files contain valid QA pairs with 'Q:' and 'A:'."""
    dataset_files = [
        os.path.join(DATASET_DIR, f)
        for f in os.listdir(DATASET_DIR)
        if f.startswith("train_") and f.endswith(".json")
    ]
    assert len(dataset_files) >= 7, f"Expected 7 SOP train files, found {len(dataset_files)}"
    
    for fpath in dataset_files:
        pairs = load_qa_pairs(fpath)
        assert len(pairs) >= 1, f"File {fpath} must have at least 1 valid QA pair"
        for p in pairs:
            assert isinstance(p, InputExample)
            assert len(p.texts) == 2
            question, answer = p.texts
            assert len(question.strip()) > 10, f"Question too short in {fpath}: {question}"
            assert len(answer.strip()) > 10, f"Answer too short in {fpath}: {answer}"


def test_dataset_produces_multi_sample_batches():
    """Verify that the merged dataset provides >= 2 samples per batch for MNR loss."""
    all_examples = load_all_dataset_examples()
    assert len(all_examples) >= 40, f"Expected at least 40 QA pairs, found {len(all_examples)}"
    
    batch_size = 8
    loader = DataLoader(all_examples, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    
    batches = list(loader)
    assert len(batches) >= 2, f"Expected multiple batches, got {len(batches)}"
    
    # Check that batches have >= 2 pairs (so MNR has in-batch negatives)
    for i, batch in enumerate(batches[:-1]):  # all full batches
        batch_len = len(batch)
        assert batch_len >= 2, f"Batch {i} has only {batch_len} samples (must be >= 2)"


def test_train_v3_evidence_matches_pdf():
    """Verify that every item in datasets/train_v3/*.json has:
    1. 'evidence' field that appears verbatim (normalized whitespace) in its source PDF.
    2. 'verified': False
    3. 'qa' with valid Q: and A:
    """
    import re
    import json
    import pypdf
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    v3_dir = root / "datasets" / "train_v3"
    data_dir = root / "data"

    assert v3_dir.exists(), "datasets/train_v3 must exist"
    v3_files = list(v3_dir.glob("train_*.json"))
    assert len(v3_files) == 7, f"Expected 7 v3 training files, found {len(v3_files)}"

    def norm_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    pdf_cache = {}
    total_items = 0

    for fpath in v3_files:
        with open(fpath, "r", encoding="utf-8") as f:
            items = json.load(f)
        assert len(items) >= 1
        for item in items:
            total_items += 1
            assert "evidence" in item and item["evidence"], f"Missing evidence in {item.get('id')}"
            assert item.get("verified") is False, f"Expected verified=False in {item.get('id')}"
            assert "qa" in item and "Q:" in item["qa"] and "A:" in item["qa"]

            src = item.get("source", "")
            pdf_name = os.path.basename(src)
            if pdf_name not in pdf_cache:
                pdf_path = data_dir / pdf_name
                assert pdf_path.exists(), f"PDF {pdf_path} not found for {item.get('id')}"
                reader = pypdf.PdfReader(str(pdf_path))
                raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
                pdf_cache[pdf_name] = norm_ws(raw_text)

            evidence_norm = norm_ws(item["evidence"])
            assert evidence_norm in pdf_cache[pdf_name], (
                f"Evidence '{evidence_norm}' for {item.get('id')} not found in PDF {pdf_name}"
            )

    assert total_items == 50, f"Expected 50 items in train_v3, got {total_items}"

