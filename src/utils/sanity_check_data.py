from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "emails.csv"

df = pd.read_csv(DATA_PATH)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nIntent counts:")
print(df["intent"].value_counts())

print("\nTone counts:")
print(df["tone"].value_counts())

print("\nNull values:")
print(df.isnull().sum())

for label in sorted(df["intent"].unique()):
    print(f"\n--- Samples for intent={label} ---")
    print(df[df["intent"] == label]["text"].head(3).to_string(index=False))