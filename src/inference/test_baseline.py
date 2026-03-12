from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "baseline" / "intent_tfidf_logreg.joblib"
ENCODER_PATH = ROOT / "models" / "baseline" / "label_encoder.joblib"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

examples = [
    "I am extremely unhappy with the charges on my account and need this fixed.",
    "Could you please help me reset my password?",
    "Thank you so much for the amazing support.",
    "URGENT: our production server is down and we need help immediately.",
    "Congratulations! You have won a free vacation. Click here now.",
]

preds = model.predict(examples)
labels = label_encoder.inverse_transform(preds)

for text, label in zip(examples, labels):
    print(f"\nTEXT: {text}")
    print(f"PREDICTED INTENT: {label}")