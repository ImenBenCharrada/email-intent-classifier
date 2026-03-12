# 📧 Email Intent Classifier

An NLP system that classifies incoming emails by intent using both traditional machine learning and transformer-based models.

The classifier categorizes emails into five intent classes:
- Complaint
- Request
- Praise
- Urgent
- Spam

The project demonstrates an end-to-end machine learning pipeline, including data preparation, model training, evaluation, API deployment, and an interactive demo interface.

## Features

- Email intent classification using NLP
- Two models implemented:
    - TF-IDF + Logistic Regression baseline
    - DistilBERT transformer fine-tuning
- REST API built with FastAPI
- Interactive Streamlit demo
- Unit tests for inference module
- Modular ML project structure
- Error analysis and evaluation

## Problem Motivation

Organizations receive large volumes of emails that must be routed or prioritized correctly.

Automatically detecting intent helps with:
- prioritizing urgent issues
- identifying complaints
- filtering spam
- routing requests to support teams
- measuring customer satisfaction signals

This project explores how machine learning can assist with automated email triaging.

## Project Architecture
```
Email Text
     │
     ▼
Data Processing Pipeline
     │
     ▼
Model Training
  ├─ TF-IDF + Logistic Regression
  └─ DistilBERT Transformer
     │
     ▼
Inference Module
     │
     ▼
FastAPI REST API
     │
     ▼
Streamlit Demo UI
```

## Project Structure
```
email-intent-classifier
│
├── app
│   ├── api
│   │   └── main.py           # FastAPI inference service
│   └── demo
│       └── streamlit_app.py  # Streamlit UI demo
│
├── data
│   ├── raw                   # source datasets
│   └── processed             # cleaned dataset
│
├── models
│   ├── baseline              # TF-IDF model
│   └── intent                # DistilBERT model
│
├── reports
│   ├── baseline_metrics.json
│   └── intent_bert_metrics.json
│
├── src
│   ├── inference
│   │   ├── predict.py
│   │   └── evaluate_hard_test.py
│   │
│   ├── training
│   │   ├── baseline_tfidf.py
│   │   └── train_intent_bert.py
│   │
│   └── utils
│       ├── build_dataset.py
│       └── sanity_check_data.py
│
├── tests
│   └── test_predict.py
│
├── requirements.txt
└── README.md
```

## Dataset

The dataset is constructed from multiple sources and manual examples.

Sources include:
- customer support tickets
- consumer complaint datasets
- spam email datasets
- manually created examples for praise and urgent classes

After processing:
- ~1250 labeled examples
- balanced across five classes
- cleaned and normalized text

## Models
### Baseline Model

TF-IDF + Logistic Regression

Advantages:
- fast
- interpretable
- strong baseline performance

### Transformer Model

DistilBERT fine-tuned for sequence classification

Advantages:
- better contextual understanding 
- handles varied language patterns 
- stronger generalization

Training configuration:
- model: `distilbert-base-uncased`
- optimizer: AdamW
- epochs: 2
- batch size: 8

## Results
```
Model	                        Accuracy	 Macro F1
TF-IDF + Logistic Regression	~0.99	     ~0.99
DistilBERT	                    ~1.00	     ~1.00
```

⚠️ **Note:** High scores are partly due to the curated and balanced dataset. Real-world performance may be lower with noisier data.

## Example Prediction

Input:
`Thank you so much for resolving my issue so quickly. I really appreciate your help. `

Output:
```
Intent: praise 
Confidence: 0.98
```

## Running the API

Start the FastAPI server:
`uvicorn app.api.main:app --reload`

Open interactive API documentation:
`http://127.0.0.1:8000/docs`

Example request:
```
POST /predict

{
  "text": "URGENT: our payment system is down."
}
```

## Running the Demo

Launch the Streamlit interface:
`streamlit run app/demo/streamlit_app.py`

This provides a simple UI where users can paste an email and see the predicted intent and confidence.

## Running Tests
`pytest`

Tests validate the inference module and ensure predictions return expected outputs.

## ⚠️ Limitations

While the models perform well on the curated dataset, several limitations remain:
- dataset size is relatively small
- some classes include synthetic examples
- model confidence may drop on out-of-distribution emails
- phishing or ambiguous messages may be misclassified

Future improvements could include:
- larger and more diverse datasets
- additional evaluation sets
- improved spam detection features
- active learning for continuous dataset improvement


## 🛠 Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- FastAPI
- Streamlit
- Pandas
- PyTest



