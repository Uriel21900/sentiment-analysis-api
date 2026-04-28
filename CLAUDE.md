# Sentiment Analysis API

A FastAPI-based sentiment analysis service with text processing capabilities.

## Project Summary

This project provides a REST API for sentiment analysis using scikit-learn with the following features:

- **Sentiment Analysis**: Classifies text as positive or negative
- **Batch Processing**: Analyze multiple texts in a single request
- **Text Cleaning**: Remove URLs, special characters, normalize whitespace
- **Tokenization**: Split text into tokens for NLP preprocessing
- **Summarization**: Keyword-based text summarization
- **Metrics**: Overview and trend statistics

## Current State

- All dependencies installed (`pip install -r requirements.txt`)
- API server running on `http://localhost:8000`
- All 8 tests passing in `tests/test_simple.py`
- API documentation available at `http://localhost:8000/docs`

## Quick Start

```bash
# Install dependencies
cd C:\Users\joseu\sentiment-analysis
pip install -r requirements.txt

# Run tests
pytest tests/test_simple.py -v

# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/api/sentiment/analyze` | POST | Analyze single text |
| `/api/sentiment/batch` | POST | Batch analyze multiple texts |
| `/api/text/clean` | POST | Clean text |
| `/api/text/tokenize` | POST | Tokenize text |
| `/api/text/summarize` | POST | Summarize text |
| `/api/metrics/overview` | GET | Overall metrics |
| `/api/metrics/trend` | GET | Sentiment trend |

## Key Files

- `api/main.py` - FastAPI app and route mounting
- `api/endpoints/sentiment.py` - Sentiment analysis endpoints
- `api/endpoints/text.py` - Text processing endpoints
- `api/endpoints/metrics.py` - Metrics endpoints
- `api/models/sentiment_model.py` - ML model for sentiment analysis
- `tests/test_simple.py` - API tests

## Notes

- The sentiment model is trained with 500 samples on first use
- Empty text returns 422 (validation error)
- Text must be 1-10000 characters for sentiment analysis
- Batch requests support 1-100 texts