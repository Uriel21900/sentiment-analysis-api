"""
FastAPI application for sentiment analysis API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.endpoints import sentiment, text, metrics

# Create FastAPI app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup app resources."""
    # Initialize models
    from api.models import sentiment_model
    sentiment_model.initialize_model()
    yield
    # Cleanup code here

app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyze sentiment of text using machine learning models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/api/sentiment/analyze - Analyze sentiment of text",
            "/api/text/batch - Batch analyze multiple texts",
            "/api/metrics/overview - Get overall metrics",
            "/api/metrics/trend - Get sentiment trend"
        ]
    }

# Include API routers
app.include_router(
    sentiment.router,
    prefix="/api/sentiment",
    tags=["Sentiment"])
app.include_router(text.router, prefix="/api/text", tags=["Text"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
