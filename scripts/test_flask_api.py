import requests

BASE_URL = "http://localhost:5000"


def test_home_endpoint():
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200
    assert isinstance(r.text, str)


def test_predict_endpoint():
    payload = {
        "comments": [
            "This is a great product!",
            "Not worth the money.",
            "It's okay."
        ]
    }

    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 3

    for item in data:
        assert "comment" in item
        assert "sentiment" in item


def test_predict_with_timestamps_endpoint():
    payload = {
        "comments": [
            {"text": "This is fantastic!", "timestamp": "2024-10-25 10:00:00"},
            {"text": "Could be better.", "timestamp": "2024-10-26 14:00:00"},
        ]
    }

    r = requests.post(f"{BASE_URL}/predict_with_timestamps", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 2

    for item in data:
        assert "comment" in item
        assert "sentiment" in item
        assert "timestamp" in item


def test_generate_chart_endpoint():
    payload = {
        "sentiment_counts": {"1": 5, "0": 3, "-1": 2}
    }

    r = requests.post(f"{BASE_URL}/generate_chart", json=payload)
    assert r.status_code == 200
    assert "image/png" in r.headers.get("Content-Type", "")
    assert len(r.content) > 0


def test_generate_wordcloud_endpoint():
    payload = {
        "comments": [
            "Love this!",
            "Not so great.",
            "Absolutely amazing!",
            "Horrible experience."
        ]
    }

    r = requests.post(f"{BASE_URL}/generate_wordcloud", json=payload)
    assert r.status_code == 200
    assert "image/png" in r.headers.get("Content-Type", "")
    assert len(r.content) > 0


def test_generate_trend_graph_endpoint():
    payload = {
        "sentiment_data": [
            {"timestamp": "2024-10-01", "sentiment": 1},
            {"timestamp": "2024-10-02", "sentiment": 0},
            {"timestamp": "2024-10-03", "sentiment": -1},
        ]
    }

    r = requests.post(f"{BASE_URL}/generate_trend_graph", json=payload)
    assert r.status_code == 200
    assert "image/png" in r.headers.get("Content-Type", "")
    assert len(r.content) > 0
