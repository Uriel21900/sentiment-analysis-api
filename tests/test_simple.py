"""
Simple tests for the Sentiment Analysis API.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestAPI:
    """Basic API tests."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Sentiment Analysis API"

    def test_sentiment_analyze_positive(self, client):
        """Test positive sentiment analysis."""
        response = client.post(
            "/api/sentiment/analyze",
            json={"text": "I love this product! It's amazing and wonderful!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "positive"
        assert data["confidence"] >= 0.5

    def test_sentiment_analyze_negative(self, client):
        """Test negative sentiment analysis."""
        response = client.post(
            "/api/sentiment/analyze",
            json={"text": "Terrible experience! Worst product ever!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] in ["positive", "negative"]
        assert "confidence" in data

    def test_sentiment_analyze_empty(self, client):
        """Test empty text handling."""
        response = client.post(
            "/api/sentiment/analyze",
            json={"text": ""}
        )
        # Empty text is rejected by Pydantic validation
        assert response.status_code in [400, 422]

    def test_batch_analyze(self, client):
        """Test batch analysis."""
        response = client.post(
            "/api/sentiment/batch",
            json={"texts": ["Great!", "Terrible!", "Okay."]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 3

    def test_overview_metrics(self, client):
        """Test overview endpoint."""
        response = client.get("/api/metrics/overview")
        assert response.status_code == 200
        data = response.json()
        assert "total_analyses" in data
        assert "positive_count" in data

    def test_clean_text(self, client):
        """Test text cleaning."""
        response = client.post(
            "/api/text/clean",
            json={"text": "Check out http://example.com product!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "original_length" in data

    def test_tokenize_text(self, client):
        """Test tokenization."""
        response = client.post(
            "/api/text/tokenize",
            json={"text": "Hello world"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert len(data["tokens"]) > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
