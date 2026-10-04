import os
import json
import shutil
import random
import datetime
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

DATASET_DIR = "datasets"
BASE_MODEL = os.getenv("BASE_EMBEDDING_MODEL", "LazarusNLP/all-indo-e5-small-v4")
FINAL_MODEL_DIR = os.getenv("FINETUNED_MODEL_DIR", "./indo_finetuned_embedding_v2")
TEMP_MODEL_DIR = "./indo_finetuned_embedding_temp"
BATCH_SIZE = int(os.getenv("FINETUNE_BATCH_SIZE", "8"))
EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "10"))
SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_qa_pairs(file_path: str):
    """Load QA pairs dari file JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    train_examples = []
    for item in qa_pairs:
        qa_text = item.get("qa", "")
        if "Q:" in qa_text and "A:" in qa_text:
            try:
                question = qa_text.split("Q:")[1].split("A:")[0].strip()
                answer = qa_text.split("A:")[1].strip()
                if question and answer:
                    train_examples.append(InputExample(texts=[question, answer], label=1.0))
            except Exception:
                continue
    return train_examples


def load_all_dataset_examples(dataset_dir: str = DATASET_DIR):
    """Gabungkan seluruh pasangan QA dari semua train_*.json menjadi satu dataset tunggal."""
    all_examples = []
    dataset_files = sorted([
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.startswith("train_") and f.endswith(".json")
    ])
    
    print(f"Menggabungkan {len(dataset_files)} file dataset untuk fine-tuning:")
    for f in dataset_files:
        examples = load_qa_pairs(f)
        print(f"   - {os.path.basename(f)}: {len(examples)} QA pairs")
        all_examples.extend(examples)
        
    return all_examples


def main():
    set_seed(SEED)
    all_examples = load_all_dataset_examples()

    if not all_examples:
        print("Tidak ada data valid ditemukan untuk training. Aborting.")
        return

    print(f"\nTotal pasangan QA terkumpul: {len(all_examples)}")
    if len(all_examples) < 2:
        raise ValueError(f"Minimal diperlukan 2 pasangan QA untuk MNR, ditemukan {len(all_examples)}")

    effective_batch_size = min(BATCH_SIZE, len(all_examples))
    print(f"Konfigurasi Training:")
    print(f"   - Base Model: {BASE_MODEL}")
    print(f"   - Output Dir: {FINAL_MODEL_DIR}")
    print(f"   - Batch Size: {effective_batch_size}")
    print(f"   - Epochs    : {EPOCHS}")
    print(f"   - Seed      : {SEED}")

    # Load base model directly for fine-tuning
    print(f"\nMemuat base model: {BASE_MODEL} ...")
    model = SentenceTransformer(BASE_MODEL)

    train_dataloader = DataLoader(all_examples, shuffle=True, batch_size=effective_batch_size, collate_fn=model.smart_batching_collate)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)

    print(f"Memulai pelatihan ({len(train_dataloader)} batch per epoch, warmup_steps={warmup_steps})...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=TEMP_MODEL_DIR
    )

    if os.path.exists(FINAL_MODEL_DIR):
        shutil.rmtree(FINAL_MODEL_DIR)
    shutil.move(TEMP_MODEL_DIR, FINAL_MODEL_DIR)

    print(f"\nPelatihan selesai! Model baru berhasil disimpan di '{FINAL_MODEL_DIR}'")


if __name__ == "__main__":
    main()
