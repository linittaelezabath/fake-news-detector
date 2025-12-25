# src/predict.py

import sys
import joblib

MODEL_PATH = "models/fake_news_pipeline.joblib"

def predict(text: str) -> int:
    model = joblib.load(MODEL_PATH)
    return int(model.predict([text])[0])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"your news text here\"")
        sys.exit(1)
    text = " ".join(sys.argv[1:])
    label = predict(text)
    print("FAKE" if label == 1 else "REAL")
