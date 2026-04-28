"""
Sentiment Analysis SDK

A Python client library for the Sentiment Analysis API.

Usage:
    from sdk import SentimentAnalyzer

    client = SentimentAnalyzer()

    # Analyze text
    result = client.analyze("I love this!")
    print(result.sentiment)

    # Batch analyze
    results = client.batch_analyze(["Great!", "Bad!"])
"""

from sdk.sdk import SentimentAnalyzer, SentimentResult, get_client, analyze, batch_analyze

__all__ = [
    "SentimentAnalyzer",
    "SentimentResult",
    "get_client",
    "analyze",
    "batch_analyze",
]

__version__ = "1.0.0"
