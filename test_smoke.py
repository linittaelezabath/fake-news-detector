# tests/test_smoke.py
import os
import subprocess
import sys

def test_model_file_exists():
    assert os.path.exists("models/fake_news_pipeline.joblib"), \
        "Run `python src/train.py` first to train and save the model."

def test_cli_prediction_runs():
    # Quick smoke test; expects exit code 0
    result = subprocess.run(
        [sys.executable, "src/predict.py", "This is a sample headline"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.strip() in {"FAKE", "REAL"}
