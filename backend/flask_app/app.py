# app.py

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
import matplotlib.dates as mdates

# NLTK imports (safe-guarded usage below)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = "http://ec2-54-172-186-220.compute-1.amazonaws.com:5000/"

MODEL_NAME = "creatorinsight_sentiment_pipeline"
MODEL_VERSION = "1"  # you can switch to "Production" later if you use stages


# -----------------------------
# Text Preprocessing (safe)
# -----------------------------
# Define the preprocessing function
def preprocess_comment(comment) -> str:
    """
    Safe preprocessing:
    - Handles int, None, float, etc.
    - Never crashes the pipeline
    """
    try:
        # Convert everything to string safely
        if comment is None:
            return ""

        comment = str(comment)

        comment = comment.lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        try:
            sw = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
        except Exception:
            sw = set()

        words = [w for w in comment.split() if w not in sw]
        comment = " ".join(words)

        try:
            lemmatizer = WordNetLemmatizer()
            comment = " ".join(lemmatizer.lemmatize(word) for word in comment.split())
        except Exception:
            pass

        return comment

    except Exception as e:
        print(f"Preprocessing failed for comment: {comment}, error: {e}")
        return str(comment)



# -----------------------------
# Load MLflow pipeline model (ONE artifact)
# -----------------------------
def load_pipeline_from_registry(model_name: str, model_version: str):
    """
    Loads a single MLflow-registered pipeline model (TF-IDF + LGBM).
    This replaces separate loading of vectorizer + model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/{model_version}"
    return mlflow.sklearn.load_model(model_uri)


# Load once at startup
try:
    pipeline = load_pipeline_from_registry(MODEL_NAME, MODEL_VERSION)
    print(f"Loaded MLflow pipeline model: {MODEL_NAME} v{MODEL_VERSION}")
except Exception as e:
    pipeline = None
    print(f"Failed to load MLflow pipeline model: {e}")


def ensure_pipeline_loaded():
    """Fail fast if model isn't loaded (better error for debugging)."""
    if pipeline is None:
        return False, jsonify({"error": "Model pipeline not loaded. Check MLflow URI/model name/version."}), 500
    return True, None, None


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return "Welcome to our flask api"


@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    ok, resp, code = ensure_pipeline_loaded()
    if not ok:
        return resp, code

    data = request.json
    comments_data = data.get("comments")

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item.get("text", "") for item in comments_data]
        timestamps = [item.get("timestamp") for item in comments_data]

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Pipeline expects raw text inputs (list of strings)
        preds = pipeline.predict(preprocessed_comments)


        # normalize output to list[str]
        if isinstance(preds, (np.ndarray, list)):
            predictions = [str(p) for p in list(preds)]
        else:
            predictions = [str(preds)]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


@app.route("/predict", methods=["POST"])
def predict():
    ok, resp, code = ensure_pipeline_loaded()
    if not ok:
        return resp, code

    data = request.json
    comments = data.get("comments")

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        preds = pipeline.predict(preprocessed_comments)



        if isinstance(preds, (np.ndarray, list)):
            predictions = [str(p) for p in list(preds)]
        else:
            predictions = [str(preds)]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)


@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get("sentiment_counts")

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ["#2F6FED", "#9AA4B2", "#E5484D"]  # subtle + premium

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "#111827"},
        )
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get("comments")

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = " ".join(preprocessed_comments)

        # Safe stopwords
        try:
            sw = set(stopwords.words("english"))
        except Exception:
            sw = set()

        wordcloud = WordCloud(
            width=900,
            height=450,
            background_color="white",
            colormap="Blues",
            stopwords=sw,
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get("sentiment_data")

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))

        colors = {-1: "#E5484D", 0: "#9AA4B2", 1: "#2F6FED"}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker="o",
                linestyle="-",
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
