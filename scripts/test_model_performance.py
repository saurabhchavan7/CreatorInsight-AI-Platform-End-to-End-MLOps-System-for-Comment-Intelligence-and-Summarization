import pytest
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("http://ec2-54-172-186-220.compute-1.amazonaws.com:5000/")

@pytest.mark.parametrize(
    "model_name, stage, test_data_path",
    [
        ("creatorinsight_sentiment_pipeline", "Staging", "data/processed/test_processed.csv"),
    ],
)
def test_model_performance(model_name, stage, test_data_path):
    """
    Validates that the staged model meets minimum performance thresholds
    using holdout test data.
    """

    # 1) Load model (sklearn flavor)
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])
    assert versions, f"No model found in {stage} stage for {model_name}"

    model_uri = f"models:/{model_name}/{versions[0].version}"
    model = mlflow.sklearn.load_model(model_uri)

    # 2) Load holdout test data
    df = pd.read_csv(test_data_path)

    # IMPORTANT: ensure pure string list (no ints, no NaNs)
    X_test = df["clean_comment"].fillna("").astype(str).tolist()
    y_true = df["category"].to_numpy()

    # 3) Predict
    y_pred = model.predict(X_test)

    # ✅ Guardrail: avoid silent “1 prediction” bugs forever
    assert len(y_pred) == len(y_true), f"Prediction count mismatch: y_pred={len(y_pred)} y_true={len(y_true)}"

    # 4) Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # 5) Threshold gate
    MIN_ACCURACY = 0.40
    MIN_PRECISION = 0.40
    MIN_RECALL = 0.40
    MIN_F1 = 0.40

    assert accuracy >= MIN_ACCURACY, f"Accuracy {accuracy:.3f} below {MIN_ACCURACY}"
    assert precision >= MIN_PRECISION, f"Precision {precision:.3f} below {MIN_PRECISION}"
    assert recall >= MIN_RECALL, f"Recall {recall:.3f} below {MIN_RECALL}"
    assert f1 >= MIN_F1, f"F1 {f1:.3f} below {MIN_F1}"
