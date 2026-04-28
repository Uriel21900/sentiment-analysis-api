"""
SDK for the Sentiment Analysis API.
Provides easy-to-use methods for sentiment analysis.
"""
import requests
import json
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    text: str
    sentiment: str
    confidence: float
    scores: dict


class SentimentAnalyzer:
    """
    Client for the Sentiment Analysis API.

    Usage:
        from sdk.sdk import SentimentAnalyzer

        client = SentimentAnalyzer(base_url="http://localhost:8000")

        # Analyze text
        result = client.analyze("I love this product!")
        print(result.sentiment)

        # Batch analyze
        results = client.batch_analyze(["Great!", "Terrible!"])
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the analyzer client.

        Args:
            base_url: Base URL for the API
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult object

        Example:
            result = client.analyze("I love this!")
            print(f"Sentiment: {result.sentiment}")
        """
        response = self.session.post(
            f"{self.base_url}/api/sentiment/analyze",
            json={"text": text}
        )
        response.raise_for_status()
        data = response.json()

        return SentimentResult(
            text=text,
            sentiment=data["sentiment"],
            confidence=data["confidence"],
            scores=data["scores"]
        )

    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts in batch.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult objects

        Example:
            results = client.batch_analyze(["Great!", "Bad!"])
            for result in results:
                print(result.sentiment)
        """
        response = self.session.post(
            f"{self.base_url}/api/sentiment/batch",
            json={"texts": texts}
        )
        response.raise_for_status()
        data = response.json()

        return [
            SentimentResult(
                text=txt,
                sentiment=result.sentiment,
                confidence=result.confidence,
                scores=result.scores
            )
            for txt, result in zip(texts, data["results"])
        ]

    def get_overview(self) -> dict:
        """
        Get overall sentiment metrics.

        Returns:
            Dictionary with metrics

        Example:
            overview = client.get_overview()
            print(f"Total analyses: {overview['total_analyses']}")
        """
        response = self.session.get(f"{self.base_url}/api/metrics/overview")
        response.raise_for_status()
        return response.json()

    def get_trend(self, days: int = 7) -> dict:
        """
        Get sentiment trend over time.

        Args:
            days: Number of days of data

        Returns:
            Dictionary with trend data

        Example:
            trend = client.get_trend(days=30)
        """
        response = self.session.get(
            f"{self.base_url}/api/metrics/trend?days={days}"
        )
        response.raise_for_status()
        return response.json()

    def clean_text(self, text: str) -> dict:
        """
        Clean and normalize text.

        Args:
            text: Text to clean

        Returns:
            Dictionary with cleaning results

        Example:
            cleaned = client.clean_text("Check http://example.com!")
        """
        response = self.session.post(
            f"{self.base_url}/api/text/clean",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()

    def tokenize(self, text: str) -> dict:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            Dictionary with tokenization results

        Example:
            tokens = client.tokenize("Hello world")
        """
        response = self.session.post(
            f"{self.base_url}/api/text/tokenize",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()

    def summarize(self, text: str) -> dict:
        """
        Summarize text.

        Args:
            text: Text to summarize

        Returns:
            Dictionary with summarization results

        Example:
            summary = client.summarize(" ".join(["word"] * 100))
        """
        response = self.session.post(
            f"{self.base_url}/api/text/summarize",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()

    def get_performance(self) -> dict:
        """
        Get model performance metrics.

        Returns:
            Dictionary with performance metrics

        Example:
            perf = client.get_performance()
            print(f"Accuracy: {perf['accuracy']}")
        """
        response = self.session.get(f"{self.base_url}/api/metrics/model/performance")
        response.raise_for_status()
        return response.json()


# Create default client instance
_analyzer = None

def get_client(base_url: str = "http://localhost:8000") -> SentimentAnalyzer:
    """Get the analyzer client instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer(base_url=base_url)
    return _analyzer


def analyze(text: str) -> SentimentResult:
    """Analyze sentiment of text."""
    return get_client().analyze(text)


def batch_analyze(texts: List[str]) -> List[SentimentResult]:
    """Batch analyze texts."""
    return get_client().batch_analyze(texts)
