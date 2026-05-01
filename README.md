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
cd sentiment-analysis-api
.
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
└── .python-version
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Uriel21900/sentiment-analysis-api.git
cd sentiment-analysis-api

# Create and activate virtual environment
python -m venv venv

# MacOS/Linux:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run the API

```bash
# Start the FastAPI server locally
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# API documentation will be available at:
# - http://localhost:8000/docs
# - http://localhost:8000/redoc
```

## 📊 API Endpoints

### POST /api/sentiment/analyze

Analyze sentiment of text

```bash
curl -X POST "http://localhost:8000/api/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "The new update is absolutely fantastic and works perfectly!"}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.984,
  "processed_text": "new update absolutely fantastic works perfectly"
}
```

### GET /api/metrics/overview
{
  "total_analyses": 1250,
  "average_confidence": 0.89,
  "sentiment_distribution": {
    "positive": 850,
    "negative": 400
  }
}
Get overall sentiment metrics

```bash
curl "http://localhost:8000/api/metrics/overview"
```

### GET /api/metrics/trend?days=7

Get sentiment trend over time

```bash
curl "http://localhost:8000/api/metrics/trend?days=7"
```

## 🧪 Running Tests

```bash
pytest
```

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t sentiment-analysis .

# Run the container
docker run -p 8000:8000 sentiment-analysis
```

## 🔧 GitHub Actions

This repository uses Node.js 24 compatible actions:
- `actions/checkout@v4` - Latest version with Node.js 24 support
- `actions/setup-python@v5` - Latest Python setup with Node.js 24 support

These versions ensure compatibility with GitHub's upcoming Node.js 24 default.

### CI Workflow

The CI pipeline:
1. ✅ Runs tests with pytest
2. ✅ Validates API structure
3. ✅ Runs linting with flake8
4. ✅ Reports coverage (optional)

Workflow file: `.github/workflows/ci.yml`

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Uriel21900
- GitHub: [@Uriel21900](https://github.com/Uriel21900)
- Email: uriel2190@gmail.com
