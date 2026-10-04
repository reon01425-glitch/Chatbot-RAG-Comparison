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
