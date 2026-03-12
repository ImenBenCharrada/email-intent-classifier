from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "intent"


class IntentPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        predicted_idx = int(probs.argmax())
        predicted_label = self.id2label[predicted_idx]
        confidence = float(probs[predicted_idx])

        probabilities = {
            self.id2label[i]: float(prob) for i, prob in enumerate(probs)
        }

        return {
            "text": text,
            "intent": predicted_label,
            "confidence": confidence,
            "probabilities": probabilities,
        }


def main() -> None:
    predictor = IntentPredictor()

    examples = [
        "I am very unhappy with the extra charges on my account and need this fixed.",
        "Could you please help me reset my password?",
        "Thank you so much for the quick support, everything works perfectly now.",
        "URGENT: our payment system is down and customers cannot check out.",
        "Congratulations! You've won a free gift card. Click here to claim now.",
    ]

    for text in examples:
        result = predictor.predict(text)
        print("\n" + "=" * 80)
        print("TEXT:", result["text"])
        print("PREDICTED INTENT:", result["intent"])
        print("CONFIDENCE:", f"{result['confidence']:.4f}")
        print("PROBABILITIES:")
        for label, prob in sorted(
            result["probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()