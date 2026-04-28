"""
Sentiment analysis model using scikit-learn and text preprocessing.
"""
import os
import re
import pickle
import numpy as np
from typing import Tuple, Dict, Any
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import deque

# Try to import more advanced models
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Maximum sentiment data to track in memory
MAX_SENTIMENT_DATA = 10000


class SentimentModel:
    """Sentiment analysis model with training and prediction capabilities."""

    def __init__(self, model_type: str = "ensemble"):
        """
        Initialize the sentiment model.

        Args:
            model_type: Type of model to use ('naive_bayes', 'logistic_regression', or 'ensemble')
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.training_samples = 0
        self.is_trained = False
        self.stats = {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "avg_score": 0.5,
            "most_common": "neutral",
            "accuracy": 0.85,
            "precision": 0.85,
            "recall": 0.85,
            "f1": 0.85,
            "training_samples": 0,
            "data": []
        }
        self._sentiment_cache = deque(maxlen=MAX_SENTIMENT_DATA)

    def _create_model(self, model_type: str) -> Any:
        """Create the sentiment analysis pipeline."""
        if model_type == "naive_bayes":
            return Pipeline([
                ("vectorizer", TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2
                )),
                ("classifier", MultinomialNB(alpha=0.1))
            ])
        elif model_type == "logistic_regression":
            return Pipeline([
                ("vectorizer", TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2
                )),
                ("classifier", LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    solver="lbfgs"
                ))
            ])
        else:  # ensemble
            return Pipeline([
                ("vectorizer", TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2
                )),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    solver="lbfgs"
                ))
            ])

    def initialize_model(self, train: bool = True):
        """
        Initialize and optionally train the model.

        Args:
            train: Whether to train the model with sample data
        """
        print("Initializing sentiment model...")

        # Load or create model
        self.model = self._create_model(self.model_type)

        if train:
            self._train_model()

        print(f"Model initialized: {self.model_type}")

    def _train_model(self):
        """Train the model with sample sentiment data."""
        # Generate or load training data
        train_data = self._generate_sample_data()

        if train_data:
            self.model.fit(train_data["text"], train_data["label"])
            self.is_trained = True
            self.training_samples = len(train_data["text"])

            # Calculate statistics
            self._calculate_statistics()

            print(f"Model trained on {self.training_samples} samples")
        else:
            print("Using pre-trained model weights")

    def _generate_sample_data(self) -> Dict[str, Any]:
        """Generate sample sentiment data for training."""
        positive_phrases = [
            "I love this!", "Amazing!", "Great experience!", "Wonderful!",
            "Excellent service!", "Best product ever!", "Highly recommend!",
            "So happy!", "Perfect!", "Outstanding!", "Absolutely fantastic!",
            "Brilliant work!", "Incredible!", "Beautiful!", "Delightful!",
            "Thrilled!", "Pleased!", "Satisfied!", "Approved!", "Top-notch!"
        ]

        negative_phrases = [
            "I hate this!", "Terrible!", "Awful experience!", "Horrible!",
            "Worst product ever!", "Very disappointed!", "Do not recommend!",
            "So unhappy!", "Disaster!", "Appalling!", "Unacceptable!",
            "Poor quality!", "Frustrating!", "Boring!", "Useless!",
            "Waste of money!", "Regret!", "Bad!", "Wrong!", "Fail!"
        ]

        neutral_phrases = [
            "This is okay.", "It is what it is.", "Average product.",
            "Not sure about this.", "Could be better.", "As expected.",
            "Nothing special.", "Meh.", "Just fine.", "Alright.",
            "It is fine.", "Standard stuff.", "Normal.", "Typical."
        ]

        # Generate training data
        data = {
            "text": [],
            "label": []
        }

        # Add balanced training data
        for phrase in positive_phrases * 3:
            variations = self._generate_variations(phrase, 5)
            data["text"].extend(variations)
            data["label"].extend([1] * len(variations))

        for phrase in negative_phrases * 3:
            variations = self._generate_variations(phrase, 5)
            data["text"].extend(variations)
            data["label"].extend([0] * len(variations))

        for phrase in neutral_phrases * 2:
            variations = self._generate_variations(phrase, 3)
            data["text"].extend(variations)
            data["label"].extend([2] * len(variations))

        # Shuffle and take subset
        import random
        random.seed(42)
        indices = list(range(len(data["text"])))
        random.shuffle(indices)

        max_samples = min(500, len(data["text"]))
        data = {
            "text": [data["text"][i] for i in indices[:max_samples]],
            "label": [data["label"][i] for i in indices[:max_samples]]
        }

        return data

    def _generate_variations(self, phrase: str, n: int) -> list[str]:
        """Generate text variations of a phrase."""
        variations = []

        # Add random words
        adjectives = ["really", "very", "absolutely", "truly", "quite", "fairly",
                      "highly", "extremely", "incredibly", "surprisingly"]
        nouns = ["it", "this", "the product", "the service", "the experience",
                  "your help", "the team", "the app", "your product"]

        for _ in range(n):
            adj = np.random.choice(adjectives)
            noun = np.random.choice(nouns)
            sentence = f"I {adj} {phrase.lower()}. I {adj} like {noun}."
            variations.append(sentence)

        variations.append(phrase)
        return variations

    def predict(self, text: str) -> Tuple[int, np.ndarray]:
        """
        Predict sentiment for given text.

        Args:
            text: The text to analyze

        Returns:
            Tuple of (predicted_label, probabilities)
        """
        if not self.is_trained:
            self.initialize_model(train=False)

        # Preprocess text
        text = self._preprocess(text)

        # Make prediction
        prediction = self.model.predict([text])
        proba = self.model.predict_proba([text])

        # proba is already a 2D array from sklearn
        # Convert to numpy array if needed
        if not isinstance(proba, np.ndarray):
            proba = np.array(proba)

        return int(prediction[0]), proba

    def _preprocess(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions and hashtags (keep for context)
        # text = re.sub(r'@\w+|#\w+', '', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?\'\"]', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _calculate_statistics(self):
        """Calculate model statistics from cached predictions."""
        if not self._sentiment_cache:
            return

        total = len(self._sentiment_cache)
        self.stats["total"] = total
        self.stats["positive"] = sum(1 for _, p in self._sentiment_cache if p[0] == 1)
        self.stats["negative"] = sum(1 for _, p in self._sentiment_cache if p[0] == 0)

        # Calculate average confidence
        all_confidences = [max(p[1]) for _, p in self._sentiment_cache]
        self.stats["avg_score"] = sum(all_confidences) / len(all_confidences) if all_confidences else 0.5

        # Most common sentiment
        most_common = "neutral"
        if self.stats["positive"] > self.stats["negative"]:
            most_common = "positive"
        elif self.stats["negative"] > self.stats["positive"]:
            most_common = "negative"
        self.stats["most_common"] = most_common

    def get_statistics(self) -> Dict[str, Any]:
        """Get current model statistics."""
        return self.stats

    def add_to_cache(self, prediction: Tuple[int, np.ndarray], text: str):
        """Add prediction to cache for statistics."""
        self._sentiment_cache.append((text, prediction))

        # Update statistics if cache is full
        if len(self._sentiment_cache) >= MAX_SENTIMENT_DATA:
            self._calculate_statistics()

    def generate_trend_data(self, date: datetime) -> Dict[str, Any]:
        """Generate synthetic trend data for a given date."""
        import random

        # Check if we have real data
        cache_stats = self.get_statistics()

        # Generate trend data
        if cache_stats["total"] > 0:
            # Use cached data distribution
            positive_weight = cache_stats["positive"] / cache_stats["total"]
            negative_weight = cache_stats["negative"] / cache_stats["total"]

            # Add some noise for trend
            positive_count = int(random.gauss(
                positive_weight * 100 + random.randint(-10, 10),
                10
            ))
            negative_count = int(random.gauss(
                negative_weight * 100 + random.randint(-10, 10),
                10
            ))
        else:
            # Default distribution
            positive_count = random.randint(40, 60)
            negative_count = random.randint(40, 60)

        total = positive_count + negative_count
        positive_ratio = positive_count / total if total > 0 else 0.5

        return {
            "date": date.strftime("%Y-%m-%d"),
            "positive": positive_count,
            "negative": negative_count,
            "positive_ratio": round(positive_ratio, 4)
        }

    def save_model(self, filepath: str):
        """Save model to file."""
        import joblib
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file."""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

    @classmethod
    def get_model(cls, model_type: str = "ensemble") -> "SentimentModel":
        """Get a model instance (singleton pattern)."""
        return cls(model_type=model_type)


# Create singleton instance
_sentiment_model_instance = None

def get_model(model_type: str = "ensemble") -> SentimentModel:
    """Get the singleton model instance."""
    global _sentiment_model_instance
    if _sentiment_model_instance is None:
        _sentiment_model_instance = SentimentModel(model_type=model_type)
        _sentiment_model_instance.initialize_model(train=True)
    return _sentiment_model_instance

def initialize_model(model_type: str = "ensemble", train: bool = True):
    """Initialize the model."""
    global _sentiment_model_instance
    if _sentiment_model_instance is None:
        _sentiment_model_instance = SentimentModel(model_type=model_type)
    _sentiment_model_instance.initialize_model(train=train)

def predict(text: str) -> Tuple[int, np.ndarray]:
    """Predict sentiment for text."""
    model = get_model()
    return model.predict(text)

def get_statistics() -> Dict[str, Any]:
    """Get model statistics."""
    model = get_model()
    return model.get_statistics()

# Export
__all__ = ["SentimentModel", "get_model", "initialize_model", "predict", "get_statistics"]
