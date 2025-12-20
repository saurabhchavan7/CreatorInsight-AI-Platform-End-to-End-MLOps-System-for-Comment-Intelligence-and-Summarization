import mlflow
import pytest
import numpy as np
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-54-172-186-220.compute-1.amazonaws.com:5000")

@pytest.mark.parametrize("model_name, stage", [
    ("creatorinsight_sentiment_pipeline", "Staging"),
])
def test_model_signature(model_name, stage):
    client = MlflowClient()

    # Get latest model version in stage
    versions = client.get_latest_versions(model_name, stages=[stage])
    assert versions, f"No model found in {stage} stage for {model_name}"

    version = versions[0].version
    model_uri = f"models:/{model_name}/{version}"

    # Load model using sklearn flavor
    model = mlflow.sklearn.load_model(model_uri)

    # Dummy raw-text input (matches pipeline contract)
    test_input = [
        "this video is amazing",
        "worst content ever",
        "average experience"
    ]

    # Run prediction
    predictions = model.predict(test_input)

    # Assertions (SIGNATURE VALIDATION)
    assert len(predictions) == len(test_input), "Output length mismatch"
    assert all(
        isinstance(p, (np.integer, np.floating))
        for p in predictions
    ), "Invalid output type"


    print(f"Model {model_name} v{version} passed signature test")
