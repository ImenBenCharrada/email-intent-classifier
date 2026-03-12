from __future__ import annotations

import json
from pathlib import Path
import torch

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "emails.csv"
MODEL_DIR = ROOT / "models" / "intent"
REPORTS_DIR = ROOT / "reports"

MODEL_NAME = "distilbert-base-uncased"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[Dataset, Dataset, dict[int, str], dict[str, int]]:
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["text", "intent"]).copy()
    df["text"] = df["text"].astype(str)
    df["intent"] = df["intent"].astype(str)

    label_list = sorted(df["intent"].unique())
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    df["label"] = df["intent"].map(label2id)

    train_df, test_df = train_test_split(
        df[["text", "label"]],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    return train_ds, test_ds, id2label, label2id


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    macro_f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
    weighted_f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "macro_f1": macro_f1["f1"],
        "weighted_f1": weighted_f1["f1"],
    }


def main() -> None:
    print(f"Loading data from: {DATA_PATH}")
    train_ds, test_ds, id2label, label2id = load_data()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True)

    print("Tokenizing datasets...")
    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training DistilBERT...")
    print("Torch device:", "cuda" if torch.cuda.is_available() else "cpu")
    trainer.train()

    print("Evaluating final model...")
    eval_results = trainer.evaluate()

    print("Running predictions on test set...")
    predictions = trainer.predict(test_ds)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)

    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    print("Saving model and tokenizer...")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    metrics = {
        "eval_results": eval_results,
        "classification_report": report,
        "confusion_matrix": cm,
        "id2label": id2label,
        "label2id": label2id,
    }

    metrics_path = REPORTS_DIR / "intent_bert_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone.")
    print(f"Model saved to: {MODEL_DIR}")
    print(f"Metrics saved to: {metrics_path}")

    print("\nFinal metrics:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()