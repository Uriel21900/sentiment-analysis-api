"""
Tests for the Sentiment Analysis API.
"""
import pytest
import requests
import json


BASE_URL = "http://localhost:8000"


@pytest.fixture
def client():
    """Create a test client."""
    from api.main import app
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "I love this product! It's amazing and wonderful!"


@pytest.fixture
def sample_text_negative():
    """Sample negative text for testing."""
    return "Terrible experience! Worst product ever!"


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_info(self, client):
        """Test that root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Sentiment Analysis API"
        assert "docs" in data
        assert "endpoints" in data

    def test_root_content_type(self, client):
        """Test that root returns JSON content."""
        response = client.get("/")
        assert "application/json" in response.headers.get("content-type", "")


class TestSentimentAnalyze:
    """Tests for sentiment analysis endpoint."""

    def test_analyze_positive_text(self, client, sample_text):
        """Test analyzing positive text."""
        response = client.post(
            "/api/sentiment/analyze",
            json={"text": sample_text}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] in ["positive", "negative"]
        assert data["confidence"] >= 0.5
        assert "scores" in data

    def test_analyze_negative_text(self, client, sample_text_negative):
        """Test analyzing negative text."""
        response = client.post(
            "/api/sentiment/analyze",
            json={"text": sample_text_negative}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] in ["positive", "negative"]
        assert "scores" in data

    def test_analyze_empty_text(self, client):
        """Test analyzing empty text."""
        response = client.post(
            "/api/sentiment/analyze",
            json={"text": ""}
        )
        # Empty text returns 422 (Pydantic validation) or 400 (custom error)
        assert response.status_code in [400, 422]

    def test_analyze_long_text(self, client):
        """Test analyzing long text."""
        long_text = "I love " + " this " * 1000
        response = client.post(
            "/api/sentiment/analyze",
            json={"text": long_text}
        )
        assert response.status_code == 200


class TestBatchAnalysis:
    """Tests for batch analysis endpoint."""

    def test_batch_analysis(self, client):
        """Test batch analysis."""
        texts = [
            "Great product!",
            "Terrible experience.",
            "Okay, nothing special."
        ]
        response = client.post(
            "/api/sentiment/batch",
            json={"texts": texts}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 3

    def test_batch_empty_list(self, client):
        """Test batch with empty list."""
        response = client.post(
            "/api/sentiment/batch",
            json={"texts": []}
        )
        assert response.status_code == 400

    def test_batch_over_limit(self, client):
        """Test batch exceeding limit."""
        texts = [f"Text {i}" for i in range(101)]
        response = client.post(
            "/api/sentiment/batch",
            json={"texts": texts}
        )
        assert response.status_code == 422  # Validation error


class TestTextEndpoints:
    """Tests for text processing endpoints."""

    def test_clean_text(self, client):
        """Test text cleaning endpoint."""
        text = "Check out http://example.com product @user! #amazing!!!"
        response = client.post(
            "/api/text/clean",
            json={"text": text}
        )
        assert response.status_code == 200
        data = response.json()
        assert "original_length" in data
        assert "cleaned_length" in data
        assert "removed_tokens" in data

    def test_tokenize_text(self, client):
        """Test tokenization endpoint."""
        text = "Hello world this is a test"
        response = client.post(
            "/api/text/tokenize",
            json={"text": text}
        )
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert len(data["tokens"]) > 0

    def test_summarize_text(self, client):
        """Test summarization endpoint."""
        text = " ".join(["word"] * 100)
        response = client.post(
            "/api/text/summarize",
            json={"text": text}
        )
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert len(data["summary"]) < len(text)


class TestMetricsEndpoints:
    """Tests for metrics endpoints."""

    def test_overview(self, client):
        """Test overview endpoint."""
        response = client.get("/api/metrics/overview")
        assert response.status_code == 200
        data = response.json()
        assert "total_analyses" in data
        assert "positive_count" in data
        assert "negative_count" in data
        assert "average_score" in data

    def test_trend(self, client):
        """Test trend endpoint."""
        response = client.get("/api/metrics/trend?days=7")
        assert response.status_code == 200
        data = response.json()
        assert "days" in data
        assert "positive_counts" in data
        assert "negative_counts" in data

    def test_trend_with_30_days(self, client):
        """Test trend with 30 days."""
        response = client.get("/api/metrics/trend?days=30")
        assert response.status_code == 200

    def test_performance(self, client):
        """Test model performance endpoint."""
        response = client.get("/api/metrics/model/performance")
        assert response.status_code == 200
        data = response.json()
        assert "accuracy" in data
        assert "f1_score" in data


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        response = client.get("/", headers={"Origin": "http://example.com"})
        assert "Access-Control-Allow-Origin" in response.headers


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/sentiment/analyze",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]

    def test_missing_text_field(self, client):
        """Test handling of missing text field."""
        response = client.post(
            "/api/sentiment/analyze",
            json={}
        )
        assert response.status_code in [400, 422]

    def test_text_too_long(self, client):
        """Test handling of text exceeding limit."""
        text = "a" * 10001
        response = client.post(
            "/api/sentiment/analyze",
            json={"text": text}
        )
        assert response.status_code in [400, 422]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
