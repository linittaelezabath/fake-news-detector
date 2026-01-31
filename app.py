from fastapi import FastAPI
from pydantic import BaseModel
import joblib

MODEL_PATH = "models/fake_news_pipeline.joblib"

app = FastAPI(title="Fake News Detector API", version="1.0.0")
model = None

class Item(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(item: Item):
    pred = model.predict([item.text])[0]
    return {"label": int(pred), "label_name": "FAKE" if int(pred) == 1 else "REAL"}
