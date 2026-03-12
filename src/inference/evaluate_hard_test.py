from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

ROOT = Path(__file__).resolve().parents[2]
HARD_TEST_PATH = ROOT / "data" / "raw" / "hard_test.csv"

BASELINE_MODEL_PATH = ROOT / "models" / "baseline" / "intent_tfidf_logreg.joblib"
BASELINE_ENCODER_PATH = ROOT / "models" / "baseline" / "label_encoder.joblib"

BERT_MODEL_DIR = ROOT / "models" / "intent"


def evaluate_baseline(df: pd.DataFrame) -> None:
    print("\n=== BASELINE TF-IDF ===")
    model = joblib.load(BASELINE_MODEL_PATH)
    encoder = joblib.load(BASELINE_ENCODER_PATH)

    preds = model.predict(df["text"])
    labels = encoder.inverse_transform(preds)

    print(classification_report(df["intent"], labels, zero_division=0))

    out = df.copy()
    out["predicted"] = labels
    print(out.to_string(index=False))


def evaluate_bert(df: pd.DataFrame) -> None:
    print("\n=== DISTILBERT ===")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    model.eval()

    id2label = model.config.id2label
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    for text in df["text"]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs).item())

        predictions.append(id2label[pred_idx])

    print(classification_report(df["intent"], predictions, zero_division=0))

    out = df.copy()
    out["predicted"] = predictions
    print(out.to_string(index=False))


def main() -> None:
    df = pd.read_csv(HARD_TEST_PATH)
    print(f"Loaded hard test set: {HARD_TEST_PATH}")
    print(df.head().to_string(index=False))

    evaluate_baseline(df)
    evaluate_bert(df)


if __name__ == "__main__":
    main()