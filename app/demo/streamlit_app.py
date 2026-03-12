import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
from src.inference.predict import IntentPredictor


st.set_page_config(page_title="Email Intent Classifier")

st.title("📧 Email Intent Classifier")

st.write(
    "Paste an email below and the model will classify its intent."
)

predictor = IntentPredictor()

text = st.text_area("Email text")

if st.button("Predict"):

    result = predictor.predict(text)

    sorted_probs = sorted(
        result["probabilities"].items(),
        key=lambda x: x[1],
        reverse=True,
    )

    top1_label, top1_prob = sorted_probs[0]
    top2_label, top2_prob = sorted_probs[1]

    st.subheader("Prediction")

    if top1_prob > 0.85:
        st.success(f"Intent: {top1_label} ({top1_prob:.2f})")
    elif top1_prob > 0.60:
        st.warning(f"Intent: {top1_label} ({top1_prob:.2f})")
    else:
        st.error(f"Uncertain prediction: {top1_label} ({top1_prob:.2f})")

    st.write(f"Second guess: {top2_label} ({top2_prob:.2f})")

    st.subheader("Probabilities")
    st.bar_chart(result["probabilities"])




