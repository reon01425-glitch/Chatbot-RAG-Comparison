import os
import json
import shutil
import datetime
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

DATASET_DIR = "datasets"
TRAINED_RECORD = os.path.join(DATASET_DIR, "trained.json")
TRAINED_LOG = os.path.join(DATASET_DIR, "trained.log")
BASE_MODEL = "LazarusNLP/all-indo-e5-small-v4"
FINAL_MODEL_DIR = "./indo_finetuned_embedding"
TEMP_MODEL_DIR = "./indo_finetuned_embedding_new"

def load_trained_files():
    """Load daftar file yang sudah pernah dipakai untuk training."""
    if os.path.exists(TRAINED_RECORD):
        with open(TRAINED_RECORD, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_trained_files(trained_files):
    """Simpan daftar file yang sudah pernah dipakai untuk training."""
    with open(TRAINED_RECORD, "w", encoding="utf-8") as f:
        json.dump(list(trained_files), f, ensure_ascii=False, indent=2)

def append_log(dataset_name):
    """Tulis log history training dengan timestamp."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TRAINED_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] Trained on {dataset_name}\n")

def load_qa_pairs(file_path):
    """Load QA pairs dari file JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    train_examples = []
    for item in qa_pairs:
        qa_text = item["qa"]
        if "Q:" in qa_text and "A:" in qa_text:
            try:
                question = qa_text.split("Q:")[1].split("A:")[0].strip()
                answer = qa_text.split("A:")[1].strip()
                if question and answer:
                    train_examples.append(InputExample(texts=[question, answer], label=1.0))
            except Exception:
                continue
    return train_examples

def main():
    all_datasets = [
        os.path.join(DATASET_DIR, f)
        for f in os.listdir(DATASET_DIR)
        if f.startswith("train_") and f.endswith(".json")
    ]

    trained_files = load_trained_files()

    new_datasets = [f for f in all_datasets if os.path.basename(f) not in trained_files]

    if not new_datasets:
        print("Tidak ada dataset baru untuk dilatih. Model sudah up-to-date.")
        return

    print(f"Ditemukan {len(new_datasets)} dataset baru untuk training:" )
    for f in new_datasets:
        print("   -", os.path.basename(f))

    if os.path.exists(FINAL_MODEL_DIR):
        print(f"Load existing fine-tuned model dari {FINAL_MODEL_DIR}")
        model = SentenceTransformer(FINAL_MODEL_DIR)
    else:
        print(f"Load base model {BASE_MODEL}")
        model = SentenceTransformer(BASE_MODEL)

    for dataset_file in new_datasets:
        dataset_name = os.path.basename(dataset_file)
        print(f"\nTraining dengan {dataset_name}")

        train_examples = load_qa_pairs(dataset_file)
        print(f"   -> {len(train_examples)} QA pairs")

        if not train_examples:
            print("Tidak ada data valid, skip.")
            trained_files.add(dataset_name)
            continue

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        num_epochs = 10
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path=TEMP_MODEL_DIR
        )

        if os.path.exists(FINAL_MODEL_DIR):
            shutil.rmtree(FINAL_MODEL_DIR)
        shutil.move(TEMP_MODEL_DIR, FINAL_MODEL_DIR)

        print(f"Training selesai untuk {dataset_name}, model updated!")

        trained_files.add(dataset_name)
        save_trained_files(trained_files)

        append_log(dataset_name)

    print("\nSemua dataset baru sudah selesai dilatih dan dicatat di trained.json")


if __name__ == "__main__":
    main()
