"""
Data pipeline for collecting and processing sentiment data.
"""
import os
import time
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from collections import deque

from api.models import sentiment_model
from api.endpoints import sentiment


class SentimentPipeline:
    """Pipeline for collecting and processing sentiment data."""

    def __init__(self, cache_path: str = "./data/cache.pkl"):
        """Initialize the pipeline."""
        self.cache_path = cache_path
        self.cache = deque(maxlen=1000)
        self.data_dir = "./data"
        os.makedirs(self.data_dir, exist_ok=True)

    def load_sample_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Load or generate sample data for processing."""
        # Generate synthetic social media-style data
        positive_phrases = [
            "Love this! Amazing product!",
            "Best purchase ever! Highly recommend!",
            "Fantastic quality and service!",
            "Absolutely wonderful experience!",
            "Perfect! Exactly what I needed!",
            "Outstanding results!",
            "Superb quality! Will buy again!",
            "Incredible! Exceeded expectations!",
            "Great value for money!",
            "Brilliant! Highly satisfied!",
        ]

        negative_phrases = [
            "Terrible! Waste of money.",
            "Horrible experience! Never again!",
            "Disappointed with the quality.",
            "Awful service! Very disappointed!",
            "Poor quality! Not worth it.",
            "Horrible product! Regret buying!",
            "Broke after one use!",
            "Not as advertised! Very angry!",
            "Overpriced garbage!",
            "Useless! Complete waste!",
        ]

        neutral_phrases = [
            "Okay, nothing special.",
            "Average product.",
            "Could be better.",
            "Nothing to complain.",
            "As expected.",
            "Standard quality.",
            "Meh, not impressed.",
            "Just fine.",
            "Alright.",
            "Nothing special.",
        ]

        # Generate data
        phrases = positive_phrases * 3 + negative_phrases * 3 + neutral_phrases * 2
        texts = [phrase for phrase in phrases for _ in range(n_samples // len(phrases))]
        while len(texts) < n_samples:
            texts.append(phrases[len(texts) % len(phrases)])

        labels = [sentiment.predict(t)[0] for t in texts]

        return pd.DataFrame({
            "text": texts,
            "sentiment": labels,
            "timestamp": [datetime.now().isoformat() for _ in texts],
            "source": "sample"
        })

    def collect_social_data(self, days: int = 1) -> pd.DataFrame:
        """
        Collect simulated social media data.

        Args:
            days: Number of days of data to collect

        Returns:
            DataFrame with collected data
        """
        print(f"Collecting {days * 24} hours of simulated data...")

        # Generate realistic trend data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = [
            start_time + timedelta(minutes=i * 5)
            for i in range(days * 24 * 12)  # Every 5 minutes
        ]

        data = []
        for i, ts in enumerate(timestamps):
            # Simulate sentiment with some randomness
            base_positive = 55 + np.sin(i / 50) * 5  # Slight trend
            noise = np.random.normal(0, 3)
            positive = max(30, min(70, base_positive + noise))
            negative = max(30, min(70, 70 - base_positive - noise))

            topic = ["Technology", "Business", "Entertainment"][i % 3]

            data.append({
                "text": self._generate_text(positive > negative),
                "sentiment": 1 if positive > negative else 0,
                "timestamp": ts.isoformat(),
                "source": "social_media",
                "topic": topic
            })

        return pd.DataFrame(data)

    def _generate_text(self, positive: bool) -> str:
        """Generate sample text for data collection."""
        if positive:
            templates = [
                "Loving this! Absolutely {adj}!",
                "{adv} impressed with the quality!",
                "Best {noun} ever! {adj} experience!",
                "Truly {adj} product! Will recommend!",
            ]
        else:
            templates = [
                "Hate this! {adj} quality!",
                "{adv} disappointed with the product!",
                "Worst {noun} ever! {adj} experience!",
                "Terrible! {adj} and {adj}!",
            ]

        adjs = ["amazing", "terrible", "great", "awful", "excellent", "bad"]
        advs = ["really", "truly", "absolutely", "highly"]
        nouns = ["product", "service", "experience"]

        template = np.random.choice(templates if positive else templates)
        text = template.format(
            adj=np.random.choice(adjs),
            adv=np.random.choice(advs),
            noun=np.random.choice(nouns)
        )
        return text

    def process_batch(self, texts: List[str], batch_size: int = 50) -> Dict[str, Any]:
        """
        Process a batch of texts.

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing

        Returns:
            Processing results
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Process each text
            for text in batch:
                pred, probs = sentiment_model.predict(text)
                sentiment = sentiment_model.get_model().model_type

                results.append({
                    "text": text,
                    "sentiment": pred,
                    "confidence": float(np.max(probs)),
                    "timestamp": datetime.now().isoformat()
                })

                # Add to cache
                sentiment_model.add_to_cache((pred, probs), text)

            # Periodically save cache
            if len(results) % 100 == 0:
                self._save_cache()

        return {
            "processed": len(results),
            "positive": sum(1 for r in results if r["sentiment"] == 1),
            "negative": sum(1 for r in results if r["sentiment"] == 0),
            "average_confidence": np.mean([r["confidence"] for r in results])
        }

    def _save_cache(self):
        """Save cache to file."""
        with open(self.cache_path, "wb") as f:
            pickle.dump(list(self.cache), f)


def run_pipeline():
    """Run the data pipeline."""
    pipeline = SentimentPipeline()

    # Load sample data
    print("Loading sample data...")
    df = pipeline.load_sample_data(n_samples=100)
    df.to_csv("./data/sample_processed.csv", index=False)
    print(f"Saved {len(df)} samples to data/sample_processed.csv")

    # Process batch
    texts = [
        "This is amazing! Love it!",
        "Terrible product. Waste of money.",
        "Great quality and fast shipping.",
        "Poor service. Very disappointed.",
        "Excellent! Highly recommend.",
    ]

    print("\nProcessing batch...")
    results = pipeline.process_batch(texts)
    print(f"Processed {results['processed']} texts")
    print(f"Positive: {results['positive']}, Negative: {results['negative']}")


if __name__ == "__main__":
    run_pipeline()
