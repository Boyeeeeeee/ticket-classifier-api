

import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_CHECKPOINT = "distilbert-base-uncased"

# --- Drive-backed paths (survive a Colab runtime reset) ---
DRIVE_ROOT = "/content/drive/MyDrive/ticket-classifier-api"
RAW_CSV_PATH = os.path.join(DRIVE_ROOT, "rows.csv")
OUTPUT_DIR = os.path.join(DRIVE_ROOT, "training-output")
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

NUM_TOP_CATEGORIES = 8       # keep the label space manageable
SAMPLES_PER_CLASS = 3000     # cap per class so training is balanced and fast
MAX_LENGTH = 256
RANDOM_STATE = 42

# Known near-duplicate "Product" labels across different CFPB taxonomy
# years/revisions. Right side is the canonical label kept after merging.
# TODO: verify/extend this against the actual unique values in your CSV
# (run df['Product'].unique() and compare) before the real training run.
LABEL_MERGE_MAP = {
    "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    "Payday loan": "Payday loan, title loan, or personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday loan, title loan, or personal loan",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
    "Bank account or service": "Checking or savings account",
}


def clean_text(text: str) -> str:
    """Strip CFPB's PII redaction placeholders and normalize whitespace."""
    text = re.sub(r"X{2,}", "", text)          # remove redaction placeholders
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_data():
    """Load the CFPB CSV, merge near-duplicate labels, filter to top
    categories, balance, clean, and split into stratified train/val/test
    sets."""
    print(f"Loading {RAW_CSV_PATH} ...")
    df = pd.read_csv(
        RAW_CSV_PATH,
        usecols=["Product", "Consumer complaint narrative"],
        low_memory=False,
    )
    df = df.rename(columns={"Consumer complaint narrative": "text", "Product": "label"})
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip().str.len() > 0]

    # Merge near-duplicate labels BEFORE picking top categories, so we
    # don't waste two of the NUM_TOP_CATEGORIES slots on labels that are
    # really the same category under different taxonomy years.
    before_counts = df["label"].value_counts()
    df["label"] = df["label"].replace(LABEL_MERGE_MAP)
    after_counts = df["label"].value_counts()
    merged = {k: v for k, v in LABEL_MERGE_MAP.items() if k in before_counts.index}
    if merged:
        print(f"Merged {len(merged)} near-duplicate label(s): {merged}")
    print("Label counts after merge:")
    print(after_counts)

    top_categories = df["label"].value_counts().nlargest(NUM_TOP_CATEGORIES).index
    df = df[df["label"].isin(top_categories)]
    print("Top categories kept:")
    print(df["label"].value_counts())

    # Balance: cap each class at SAMPLES_PER_CLASS so no single category
    # dominates training.
    df = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), SAMPLES_PER_CLASS), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 10]  # drop near-empty narratives after cleaning

    labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    df["label_id"] = df["label"].map(label2id)

    # Stratified 70/15/15 split.
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label_id"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=RANDOM_STATE
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df, labels, label2id


def tokenize_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

    def _tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH
        )

    ds = ds.map(_tokenize, batched=True)
    ds = ds.remove_columns(["text"])
    ds.set_format("torch")
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def _latest_checkpoint():
    """Return the most recent checkpoint dir under CHECKPOINT_DIR, if any,
    so training can resume after a disconnect instead of restarting."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*"))
    if not ckpts:
        return None
    latest = max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))
    print(f"Found existing checkpoint, will resume from: {latest}")
    return latest


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df, val_df, test_df, labels, label2id = load_and_prepare_data()
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds = tokenize_dataset(val_df, tokenizer)
    test_ds = tokenize_dataset(test_df, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,  # keep Drive usage bounded
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    resume_from = _latest_checkpoint()
    trainer.train(resume_from_checkpoint=resume_from)

    print("\nEvaluating on held-out test set...")
    test_output = trainer.predict(test_ds)
    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_labels = test_output.label_ids

    print(f"\nTest accuracy: {accuracy_score(test_labels, test_preds):.4f}")
    print(f"Test macro-F1: {f1_score(test_labels, test_preds, average='macro'):.4f}\n")
    print(classification_report(test_labels, test_preds, target_names=labels))

    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # Save everything the API needs.
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    with open(os.path.join(FINAL_MODEL_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nSaved final model, tokenizer, and label list to {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    train()