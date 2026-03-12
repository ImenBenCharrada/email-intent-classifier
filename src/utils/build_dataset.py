from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "emails.csv"


INTENT_LABELS = ["complaint", "request", "praise", "urgent", "spam"]
TONE_LABELS = ["angry", "polite", "neutral"]


def clean_text(text: str) -> str:
    """Basic text cleanup for training."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("\\r", " ").replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " [LINK] ", text)
    text = re.sub(r"\S+@\S+", " [EMAIL] ", text)
    text = re.sub(r"\b\d{5,}\b", " [NUMBER] ", text)
    text = text.replace("\\n", " ").replace("\\r", " ")
    return text.strip()


def infer_tone(text: str) -> str:
    """Temporary rule-based tone labeling."""
    if not isinstance(text, str):
        return "neutral"

    t = text.lower()

    polite_words = ["please", "thank you", "thanks", "kindly", "appreciate", "grateful"]
    angry_words = ["unacceptable", "furious", "terrible", "angry", "refund", "worst", "asap"]

    if any(word in t for word in polite_words):
        return "polite"

    if any(word in t for word in angry_words):
        return "angry"

    if text.count("!") >= 2 or text.isupper():
        return "angry"

    return "neutral"


def load_requests(max_rows: int = 1000) -> pd.DataFrame:
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    df = ds.to_pandas()

    candidate_cols = ["ticket_text", "text", "body", "message", "ticket", "instruction"]
    text_col = next((c for c in candidate_cols if c in df.columns), None)
    if text_col is None:
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not object_cols:
            raise ValueError("Could not find a text column in request dataset.")
        text_col = object_cols[0]

    out = df[[text_col]].rename(columns={text_col: "text"}).copy()
    out = out.dropna(subset=["text"])
    out["text"] = out["text"].astype(str)
    out["intent"] = "request"
    out["tone"] = out["text"].apply(infer_tone)
    out["text"] = out["text"].apply(clean_text)
    out = out[out["text"].str.len() > 20]
    return out.head(max_rows)


def load_complaints(max_rows: int = 1000) -> pd.DataFrame:
    ds = load_dataset("determined-ai/consumer_complaints_short", split="train")
    df = ds.to_pandas()

    # Use the correct column
    text_col = "Consumer Complaint"

    out = df[[text_col]].rename(columns={text_col: "text"}).copy()

    out = out.dropna(subset=["text"])
    out["text"] = out["text"].astype(str)

    out["intent"] = "complaint"
    out["tone"] = out["text"].apply(infer_tone)

    out["text"] = out["text"].apply(clean_text)
    out = out[out["text"].str.len() > 20]

    return out.head(max_rows)


def load_spam(max_rows: int = 1000) -> pd.DataFrame:
    ds = load_dataset("UniqueData/email-spam-classification", split="train")
    df = ds.to_pandas()

    candidate_text_cols = ["text", "email", "message", "body"]
    text_col = next((c for c in candidate_text_cols if c in df.columns), None)
    if text_col is None:
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not object_cols:
            raise ValueError("Could not find a text column in spam dataset.")
        text_col = object_cols[0]

    candidate_label_cols = ["label", "spam", "target", "class"]
    label_col = next((c for c in candidate_label_cols if c in df.columns), None)

    if label_col is None:
        spam_df = df[[text_col]].rename(columns={text_col: "text"}).copy()
    else:
        tmp = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).copy()
        tmp["label_str"] = tmp["label"].astype(str).str.lower()
        spam_df = tmp[
            tmp["label_str"].isin(["spam", "1", "true", "yes"])
            | tmp["label"].astype(str).str.contains("spam", case=False, na=False)
        ][["text"]].copy()

    spam_df = spam_df.dropna(subset=["text"])
    spam_df["text"] = spam_df["text"].astype(str)
    spam_df["intent"] = "spam"
    spam_df["tone"] = spam_df["text"].apply(infer_tone)
    spam_df["text"] = spam_df["text"].apply(clean_text)
    spam_df = spam_df[spam_df["text"].str.len() > 20]
    return spam_df.head(max_rows)


def load_manual_seed() -> pd.DataFrame:
    path = RAW_DIR / "manual_seed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    required = {"text", "intent", "tone"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manual_seed.csv missing columns: {missing}")

    df["text"] = df["text"].astype(str).apply(clean_text)
    df["intent"] = df["intent"].astype(str).str.lower().str.strip()
    df["tone"] = df["tone"].astype(str).str.lower().str.strip()

    df = df[df["intent"].isin(INTENT_LABELS)]
    df = df[df["tone"].isin(TONE_LABELS)]
    df = df[df["text"].str.len() > 10]
    return df


def balance_by_intent(df: pd.DataFrame, n_per_class: int = 300) -> pd.DataFrame:
    parts = []
    for label in INTENT_LABELS:
        group = df[df["intent"] == label].copy()
        if len(group) == 0:
            continue

        if len(group) >= n_per_class:
            sampled = group.sample(n=n_per_class, random_state=42)
        else:
            # oversample smaller classes
            sampled = group.sample(n=n_per_class, replace=True, random_state=42)

        parts.append(sampled)

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return out


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading public datasets...")
    requests_df = load_requests(max_rows=1200)
    complaints_df = load_complaints(max_rows=1200)
    spam_df = load_spam(max_rows=1200)

    print("Loading manual seed...")
    manual_df = load_manual_seed()

    full_df = pd.concat(
        [requests_df, complaints_df, spam_df, manual_df],
        ignore_index=True,
    )

    full_df = full_df.drop_duplicates(subset=["text"])
    full_df = full_df[full_df["text"].str.len() > 20]

    balanced_df = balance_by_intent(full_df, n_per_class=250)
    balanced_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved dataset to: {OUTPUT_PATH}")
    print("\nIntent counts:")
    print(balanced_df["intent"].value_counts())
    print("\nTone counts:")
    print(balanced_df["tone"].value_counts())
    print("\nSample rows:")
    print(balanced_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()