"""
Sentiment analysis API endpoints.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np

# BaseModel is already imported from pydantic


from api.models import sentiment_model

router = APIRouter()


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis."""
    text: str = Field(..., min_length=1, max_length=10000,
                      description="Text to analyze")


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    sentiment: str
    confidence: float
    scores: dict

    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": 0.9823,
                "scores": {"positive": 0.9823, "negative": 0.0177}
            }
        }


class BatchSentimentRequest(BaseModel):
    """Request model for batch sentiment analysis."""
    texts: list[str] = Field(..., min_items=1, max_items=100)


class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment analysis."""
    results: list[SentimentResponse]
    total_processed: int


@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment of a single text.

    Analyzes the sentiment of the provided text using a pre-trained
    machine learning model and returns the sentiment label and confidence score.

    **Parameters:**
        - text (str): The text to analyze (1-10000 characters)

    **Returns:**
        Sentiment analysis result with sentiment label and confidence scores
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Get sentiment prediction
    pred, probs = sentiment_model.predict(request.text)

    sentiment = "positive" if pred == 1 else "negative"
    confidence = float(np.max(probs))

    # Flatten probs to ensure proper array handling
    probs_flat = probs.flatten()

    # Handle both 2-class and binary classification cases
    # For 2-class: probs = [negative_prob, positive_prob]
    # For binary (single class): probs = [positive_prob]
    if len(probs_flat) >= 2:
        scores = {
            "positive": float(probs_flat[1]),
            "negative": float(probs_flat[0])
        }
    else:
        # Single class - assume it's positive
        scores = {
            "positive": float(probs_flat[0]),
            "negative": 1.0 - float(probs_flat[0])
        }

    return SentimentResponse(
        sentiment=sentiment,
        confidence=confidence,
        scores=scores
    )


@router.post("/batch", response_model=BatchSentimentResponse)
async def batch_analyze(request: BatchSentimentRequest):
    """
    Analyze sentiment of multiple texts in batch.

    Processes multiple texts efficiently in a single request. Ideal
    for processing large batches of text data.

    **Parameters:**
        - texts (list[str]): List of texts to analyze (1-100 texts)

    **Returns:**
        List of sentiment analysis results for each text
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="At least one text is required")

    results = []
    for text in request.texts:
        if not text.strip():
            continue

        pred, probs = sentiment_model.predict(text)
        sentiment = "positive" if pred == 1 else "negative"
        confidence = float(np.max(probs))

        # Flatten probs to ensure proper array handling
        probs_flat = probs.flatten()

        # Handle both 2-class and binary classification cases
        if len(probs_flat) >= 2:
            scores = {
                "positive": float(probs_flat[1]),
                "negative": float(probs_flat[0])
            }
        else:
            # Single class - assume it's positive
            scores = {
                "positive": float(probs_flat[0]),
                "negative": 1.0 - float(probs_flat[0])
            }

        results.append(SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            scores=scores
        ))

    return BatchSentimentResponse(
        results=results,
        total_processed=len(results)
    )
