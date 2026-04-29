# 🚀 Sentiment Analysis API

A production-ready sentiment analysis system built with FastAPI that provides real-time text analysis, sentiment classification, and text processing capabilities.

## ✨ Features

- 📊 **Sentiment Analysis**: Classify text as positive/negative with confidence scores
- 🤖 **ML-Powered**: Uses scikit-learn with TF-IDF and logistic regression
- 🔌 **REST API**: FastAPI endpoints with Pydantic validation
- 📈 **Metrics**: Track sentiment trends and model performance
- 🧹 **Text Processing**: Cleaning, tokenization, summarization
- 📁 **CI/CD Ready**: Automated testing with GitHub Actions
- 🐳 **Docker**: Container-ready deployment

## 🛠️ Tech Stack

- **FastAPI** - REST API framework
- **Scikit-learn** - Machine learning models
- **NumPy/Pandas** - Data manipulation
- **Matplotlib/Plotly** - Data visualization

## 📁 Project Structure

```
sentiment-analysis/
├── api/
│   ├── endpoints/
│   │   ├── sentiment.py
│   │   ├── text.py
│   │   └── metrics.py
│   ├── models/
│   │   └── sentiment_model.py
│   └── main.py
├── tests/
│   ├── test_api.py
│   └── test_simple.py
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
```

### Run the API

```bash
# Start the FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# API documentation will be available at:
# - http://localhost:8000/docs
# - http://localhost:8000/redoc
```

### Run the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run dashboards/app.py

# Dashboard will be available at:
# http://localhost:8501
```

### Run Data Pipeline

```bash
# Process real-time data from APIs
python data/pipeline.py
```

## 📊 API Endpoints

### POST /api/sentiment/analyze
Analyze sentiment of text

```bash
curl -X POST "http://localhost:8000/api/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product! It changed my life!"}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.98,
  "scores": {
    "positive": 0.98,
    "negative": 0.02
  }
}
```

### GET /api/metrics/overview
Get overall sentiment metrics

```bash
curl "http://localhost:8000/api/metrics/overview"
```

### GET /api/metrics/trend?days=7
Get sentiment trend over time

```bash
curl "http://localhost:8000/api/metrics/trend?days=7"
```

## 📈 Dashboard Features

- Real-time sentiment scores
- Trending topics visualization
- Topic distribution charts
- Historical sentiment analysis
- Word cloud of most discussed topics
- User engagement metrics

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t sentiment-analysis .

# Run the container
docker run -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -e API_KEY=your_api_key \
  sentiment-analysis
```

## 📝 Usage Examples

### Python SDK

```python
from sentiment_api import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Analyze single text
result = analyzer.analyze("This is amazing! Love it!")
print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.confidence}")

# Batch analysis
texts = ["Great product!", "Terrible experience"]
results = analyzer.batch_analyze(texts)
```

### Jupyter Notebook Analysis

```python
import pandas as pd
from sentiment_api import SentimentAnalyzer

# Load data
df = pd.read_csv('data/social_media.csv')

# Analyze sentiments
analyzer = SentimentAnalyzer()
df['sentiment'] = df['text'].apply(analyzer.analyze)

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df.groupby('date')['sentiment'].mean().plot()
plt.title('Sentiment Trend')
plt.tight_layout()
plt.show()
```

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) first.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Your Name
- GitHub: [@Uriel21900](https://github.com/Uriel21900)
- Email: Joseurodriguez@gmail.com

## 👤 Author

Uriel21900
- GitHub: [@Uriel21900](https://github.com/Uriel21900)
- Email: uriel21900@example.com

## 🔧 GitHub Actions Compatibility

This repository uses:
- `actions/checkout@v4` - Latest version with Node.js 24 support
- `actions/setup-python@v5` - Latest Python setup with Node.js 24 support

These versions ensure compatibility with GitHub's upcoming Node.js 24 default.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
