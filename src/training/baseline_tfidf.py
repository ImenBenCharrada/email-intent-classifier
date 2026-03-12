from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "emails.csv"
MODELS_DIR = ROOT / "models" / "baseline"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm, labels, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Baseline TF-IDF Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    required_cols = {"text", "intent"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["text", "intent"]).copy()
    df["text"] = df["text"].astype(str)
    df["intent"] = df["intent"].astype(str)

    X = df["text"]
    y = df["intent"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=15000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training baseline model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    target_names = list(label_encoder.classes_)
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classes": target_names,
        "classification_report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    model_path = MODELS_DIR / "intent_tfidf_logreg.joblib"
    label_encoder_path = MODELS_DIR / "label_encoder.joblib"
    metrics_path = REPORTS_DIR / "baseline_metrics.json"
    cm_path = REPORTS_DIR / "baseline_confusion_matrix.png"

    joblib.dump(pipeline, model_path)
    joblib.dump(label_encoder, label_encoder_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(cm, target_names, cm_path)

    print("\nDone.")
    print(f"Model saved to: {model_path}")
    print(f"Label encoder saved to: {label_encoder_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {cm_path}")

    print("\nResults:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    main()
