# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-04-28

### Added
- Initial release of Sentiment Analysis Dashboard
- FastAPI REST API with sentiment analysis endpoints
- Streamlit interactive dashboard
- Pre-trained sentiment model using TF-IDF + Logistic Regression
- Sample data for demonstration
- Docker support for easy deployment
- Batch sentiment analysis capability
- Text preprocessing utilities
- Model metrics and performance endpoints
- Time-series trend visualization
- Topic distribution charts

### Features
- `/api/sentiment/analyze` - Single text analysis
- `/api/sentiment/batch` - Batch text analysis
- `/api/metrics/overview` - Overall metrics
- `/api/metrics/trend` - Sentiment trend over time
- `/api/text/clean` - Text cleaning
- `/api/text/tokenize` - Tokenization
- `/api/text/summarize` - Text summarization

### Documentation
- Comprehensive README.md
- API documentation via FastAPI docs
- Inline code documentation
- Usage examples in SDK

### Data
- Sample data CSV for testing
- Jupyter notebook for EDA
- Model training notebook
- Data processing pipeline
