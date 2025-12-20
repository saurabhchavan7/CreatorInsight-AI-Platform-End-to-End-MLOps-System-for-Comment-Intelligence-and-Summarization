import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    logger.debug('Data loaded and NaNs filled from %s', file_path)
    return df


def load_pipeline(pipeline_path: str):
    with open(pipeline_path, 'rb') as file:
        pipeline = pickle.load(file)
    logger.debug('Pipeline loaded from %s', pipeline_path)
    return pipeline


def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    logger.debug('Parameters loaded from %s', params_path)
    return params


def evaluate_model(pipeline, X_test_text: np.ndarray, y_test: np.ndarray):
    y_pred = pipeline.predict(X_test_text)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    logger.debug('Model evaluation completed')
    return report, cm


def log_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    model_info = {'run_id': run_id, 'model_path': model_path}
    with open(file_path, 'w') as file:
        json.dump(model_info, file, indent=4)
    logger.debug('Model info saved to %s', file_path)


def main():
    mlflow.set_tracking_uri("http://ec2-54-172-186-220.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment('dvc-pipeline-runs')

    with mlflow.start_run() as run:
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # log params (keep your behavior)
            for key, value in params.items():
                mlflow.log_param(key, value)

            # load pipeline
            pipeline = load_pipeline(os.path.join(root_dir, 'models/sentiment_pipeline.pkl'))

            # load test data
            test_data = load_data(os.path.join(root_dir, 'data/processed/test_processed.csv'))
            X_test_text = test_data['clean_comment'].values
            y_test = test_data['category'].values

            # signature inference (raw text in -> preds out)
            input_example = list(X_test_text[:5])
           # signature = infer_signature(input_example, pipeline.predict(input_example))

            # log pipeline model (ONE artifact)
            mlflow.sklearn.log_model(
            pipeline,
            artifact_path="sentiment_pipeline"
            )


            # Save model info for registration step
            save_model_info(run.info.run_id, "sentiment_pipeline", "experiment_info.json")

            # Evaluate + log metrics
            report, cm = evaluate_model(pipeline, X_test_text, y_test)

            report_path = "classification_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
            mlflow.log_artifact(report_path)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            log_confusion_matrix(cm, "Test Data")

            mlflow.set_tag("model_type", "TFIDF + LightGBM Pipeline")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube/Reddit Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")


if __name__ == '__main__':
    main()




# # -----------------------------------------------------------------------
# # logging configuration
# logger = logging.getLogger('model_evaluation')
# logger.setLevel('DEBUG')

# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')

# file_handler = logging.FileHandler('model_evaluation_errors.log')
# file_handler.setLevel('ERROR')

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)


# def load_data(file_path: str) -> pd.DataFrame:
#     """Load data from a CSV file."""
#     try:
#         df = pd.read_csv(file_path)
#         df.fillna('', inplace=True)  # Fill any NaN values
#         logger.debug('Data loaded and NaNs filled from %s', file_path)
#         return df
#     except Exception as e:
#         logger.error('Error loading data from %s: %s', file_path, e)
#         raise


# def load_model(model_path: str):
#     """Load the trained model."""
#     try:
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file)
#         logger.debug('Model loaded from %s', model_path)
#         return model
#     except Exception as e:
#         logger.error('Error loading model from %s: %s', model_path, e)
#         raise


# def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
#     """Load the saved TF-IDF vectorizer."""
#     try:
#         with open(vectorizer_path, 'rb') as file:
#             vectorizer = pickle.load(file)
#         logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
#         return vectorizer
#     except Exception as e:
#         logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
#         raise


# def load_params(params_path: str) -> dict:
#     """Load parameters from a YAML file."""
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logger.debug('Parameters loaded from %s', params_path)
#         return params
#     except Exception as e:
#         logger.error('Error loading parameters from %s: %s', params_path, e)
#         raise


# def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
#     """Evaluate the model and log classification metrics and confusion matrix."""
#     try:
#         # Predict and calculate classification metrics
#         y_pred = model.predict(X_test)
#         report = classification_report(y_test, y_pred, output_dict=True)
#         cm = confusion_matrix(y_test, y_pred)
        
#         logger.debug('Model evaluation completed')

#         return report, cm
#     except Exception as e:
#         logger.error('Error during model evaluation: %s', e)
#         raise


# def log_confusion_matrix(cm, dataset_name):
#     """Log confusion matrix as an artifact."""
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Confusion Matrix for {dataset_name}')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')

#     # Save confusion matrix plot as a file and log it to MLflow
#     cm_file_path = f'confusion_matrix_{dataset_name}.png'
#     plt.savefig(cm_file_path)
#     mlflow.log_artifact(cm_file_path)
#     plt.close()

# def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
#     """Save the model run ID and path to a JSON file."""
#     try:
#         # Create a dictionary with the info you want to save
#         model_info = {
#             'run_id': run_id,
#             'model_path': model_path
#         }
#         # Save the dictionary as a JSON file
#         with open(file_path, 'w') as file:
#             json.dump(model_info, file, indent=4)
#         logger.debug('Model info saved to %s', file_path)
#     except Exception as e:
#         logger.error('Error occurred while saving the model info: %s', e)
#         raise


# def main():
#     mlflow.set_tracking_uri("http://ec2-3-94-145-211.compute-1.amazonaws.com:5000/")

#     mlflow.set_experiment('dvc-pipeline-runs')
    
#     with mlflow.start_run() as run:
#         try:
#             # Load parameters from YAML file
#             root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
#             params = load_params(os.path.join(root_dir, 'params.yaml'))

#             # Log parameters
#             for key, value in params.items():
#                 mlflow.log_param(key, value)
            
#             # Load model and vectorizer
#             model = load_model(os.path.join(root_dir, 'models/lgbm_model.pkl'))
#             vectorizer = load_vectorizer(os.path.join(root_dir, 'models/tfidf_vectorizer.pkl'))

#             # Load test data for signature inference
#             test_data = load_data(os.path.join(root_dir, 'data/processed/test_processed.csv'))

#             # Prepare test data
#             X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
#             y_test = test_data['category'].values

#             # Create a DataFrame for signature inference (using first few rows as an example)
#             input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())  # <--- Added for signature

#             # Infer the signature
#             signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))  # <--- Added for signature

#             # Log model with signature
#             mlflow.sklearn.log_model(
#                 model,
#                 "lgbm_model",
#                 signature=signature,  # <--- Added for signature
#                 input_example=input_example  # <--- Added input example
#             )

#             # Save model info

#             # artifact_uri = mlflow.get_artifact_uri()
#             # model_path = f"{artifact_uri}/lgbm_model"
            
#             model_path = "lgbm_model"
#             save_model_info(run.info.run_id, model_path, 'experiment_info.json')

#             # Log the vectorizer as an artifact
#             mlflow.log_artifact(os.path.join(root_dir, 'models/tfidf_vectorizer.pkl'))

#             # Evaluate model and get metrics
#             report, cm = evaluate_model(model, X_test_tfidf, y_test)

#             # ---------------------------------------------------------
#             # SAVE CLASSIFICATION REPORT AS JSON + LOG TO MLFLOW
#             # ---------------------------------------------------------
#             report_path = "classification_report.json"
#             with open(report_path, "w") as f:
#                 json.dump(report, f, indent=4)

#             mlflow.log_artifact(report_path)
#             logger.debug("Saved and logged classification_report.json")


#             # Log classification report metrics for the test data
#             for label, metrics in report.items():
#                 if isinstance(metrics, dict):
#                     mlflow.log_metrics({
#                         f"test_{label}_precision": metrics['precision'],
#                         f"test_{label}_recall": metrics['recall'],
#                         f"test_{label}_f1-score": metrics['f1-score']
#                     })

#             # Log confusion matrix
#             log_confusion_matrix(cm, "Test Data")

#             # Add important tags
#             mlflow.set_tag("model_type", "LightGBM")
#             mlflow.set_tag("task", "Sentiment Analysis")
#             mlflow.set_tag("dataset", "YouTube Comments")

#         except Exception as e:
#             logger.error(f"Failed to complete model evaluation: {e}")
#             print(f"Error: {e}")

# if __name__ == '__main__':
#     main()