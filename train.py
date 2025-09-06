# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import re

DATA_PATH = "data/train.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_pipeline.joblib")

def basic_clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<[^>]+>", " ", s)          # strip HTML
    s = re.sub(r"http\S+|www\.\S+", " ", s) # strip URLs
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)   # punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def main():
    df = pd.read_csv(DATA_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    df["text"] = df["text"].fillna("").map(basic_clean)

    X = df["text"].values
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
        # stratify for balanced split
    )

    # Handle potential class imbalance
    classes = sorted(set(y))
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {c: w for c, w in zip(classes, cw)}

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),       # unigrams + bigrams
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=200,
            n_jobs=None,
            class_weight=class_weight
        ))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, preds))
    print("\nConfusion matrix:\n", confusion_matrix(y_val, preds))
    print("\nClassification report:\n", classification_report(y_val, preds, digits=4))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
