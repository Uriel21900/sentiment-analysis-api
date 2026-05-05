"""
Text processing endpoints for cleaning and preprocessing.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import re
from collections import Counter

# BaseModel and Field already imported from pydantic


router = APIRouter()


class TextCleanRequest(BaseModel):
    """Request model for text cleaning."""
    text: str = Field(..., min_length=1, max_length=10000)


class TextCleanResponse(BaseModel):
    """Response model for text cleaning."""
    original_length: int
    cleaned_length: int
    removed_count: int
    removed_tokens: list[str]


class TokenizeRequest(BaseModel):
    """Request model for tokenization."""
    text: str = Field(..., min_length=1, max_length=10000)


class TokenizeResponse(BaseModel):
    """Response model for tokenization."""
    tokens: list[str]
    token_count: int
    unique_tokens: int


class SummarizeRequest(BaseModel):
    """Request model for text summarization."""
    text: str = Field(..., min_length=10, max_length=10000)


class SummarizeResponse(BaseModel):
    """Response model for text summarization."""
    original_length: int
    summary: str
    compression_ratio: float


@router.post("/clean", response_model=TextCleanResponse)
async def clean_text(request: TextCleanRequest):
    """
    Clean and normalize text.

    Removes special characters, normalizes whitespace, and
    handles common text preprocessing tasks.

    **Parameters:**
        - text (str): The text to clean

    **Returns:**
        Cleaning statistics and cleaned text
    """
    original_length = len(request.text)

    # Remove URLs
    text = re.sub(
        r'http\S+|www\S+|https\S+',
        '',
        request.text,
        flags=re.MULTILINE)

    # Remove mentions and hashtags (optional - can keep for sentiment)
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?\'\"]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    cleaned_length = len(text)
    removed_count = original_length - cleaned_length
    removed_tokens = ['URLs', 'mentions', 'hashtags', 'special_chars']

    return TextCleanResponse(
        original_length=original_length,
        cleaned_length=cleaned_length,
        removed_count=removed_count,
        removed_tokens=removed_tokens
    )


@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize_text(request: TokenizeRequest):
    """
    Tokenize text into individual words/tokens.

    **Parameters:**
        - text (str): The text to tokenize

    **Returns:**
        Token list with statistics
    """
    # Basic tokenization
    tokens = request.text.lower().split()

    return TokenizeResponse(
        tokens=tokens,
        token_count=len(tokens),
        unique_tokens=len(set(tokens))
    )


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """
    Create a summary of the input text.

    Uses a simple keyword-based summarization approach.

    **Parameters:**
        - text (str): The text to summarize

    **Returns:**
        Summary and compression ratio
    """
    if len(request.text) < 10:
        raise HTTPException(status_code=400,
                            detail="Text too short for summarization")

    original_length = len(request.text)

    # Extract sentences
    sentences = request.text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]

    # Simple keyword-based summary
    words = Counter(' '.join(sentences).lower().split())
    total_words = len(words)

    # Get top 20% of important words
    if total_words > 0:
        top_words = [word for word,
                     _ in words.most_common(int(total_words * 0.2))]
    else:
        top_words = []

    # Reconstruct summary
    summary = ' '.join(top_words[:50])

    return SummarizeResponse(
        original_length=original_length,
        summary=summary,
        compression_ratio=(
            len(summary) / original_length
        ) if original_length > 0 else 0
    )
