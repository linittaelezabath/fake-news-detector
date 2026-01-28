### Fake News Detection API

A machine learning–based Fake News Detection system built using Python, scikit-learn, and FastAPI.
The project trains a text classification model to identify whether a news article is FAKE or REAL, exposes it as a REST API, and also provides a CLI prediction script.

### Features

* Train a Fake News classifier using TF-IDF + Logistic Regression

* Basic text cleaning (HTML, URLs, punctuation removal)

* Handles class imbalance using class_weight

*REST API built with FastAPI

* Command-line prediction support

* Smoke tests for model and CLI

* Model persistence using joblib

### Project Structure
.
├── data/
│   ├── raw/
│   │   ├── true.csv
│   │   └── fake.csv
│   └── train.csv
│
├── models/
│   └── fake_news_pipeline.joblib
│
├── src/
│   ├── app.py        # FastAPI application
│   ├── train.py      # Model training script
│   ├── predict.py    # CLI prediction script
│
├── tests/
│   └── test_smoke.py # Basic smoke tests
│
├── README.md
└── requirements.txt

### Dataset

The dataset consists of two CSV files:

true.csv → Real news articles

fake.csv → Fake news articles

Both files should be placed inside:

data/raw/

Dataset Processing

Articles from both datasets are combined

Labels:

0 → REAL

1 → FAKE

title and text columns are merged into a single text field

### Installation
1️, Clone the Repository
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector

2️, Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

3️, Install Dependencies
pip install -r requirements.txt

### Prepare Training Data

Combine the raw datasets into a single training file:

python data/prepare_data.py


This will generate:

data/train.csv

### Train the Model
python src/train.py

Output:

Accuracy, confusion matrix, classification report

Saved model:

models/fake_news_pipeline.joblib

## Command-Line Prediction

Predict directly from the terminal:

python src/predict.py "Breaking news: Scientists discover water on Mars"

Output:
REAL

or

FAKE

### Run the API

Start the FastAPI server:

uvicorn src.app:app --reload

API Docs (Swagger UI)

Open in browser:

http://127.0.0.1:8000/docs

### API Usage
Endpoint
POST /predict

Request Body
{
  "text": "Your news article text here"
}

Response
{
  "label": 1,
  "label_name": "FAKE"
}

## Run Tests
pytest

Tests Include:

Model file existence check

CLI prediction smoke test

## Tech Stack

Python

scikit-learn

FastAPI

Pandas

Joblib

Pytest

## Model Details

Vectorizer: TF-IDF (unigrams + bigrams)

Classifier: Logistic Regression

Evaluation:

Accuracy

Confusion Matrix

Precision / Recall / F1-Score

Class Imbalance Handling: class_weight="balanced"

### Future Improvements

Add deep learning models (LSTM, BERT)

Dockerize the application

Add authentication to the API

Deploy on cloud (AWS / Azure / Render)

Add confidence score in predictions

### Author

Linitta Elezabath 
B.Tech CSE (AI & ML)
Graduating 2027
