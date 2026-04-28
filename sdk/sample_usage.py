"""
Sample usage scripts for the Sentiment Analysis API.
"""

# =====================================================
# Example 1: Basic Usage with the API
# =====================================================


def example_api_usage():
    """Example of using the API directly with requests."""
    import requests
    import json

    # Analyze a single text
    url = "http://localhost:8000/api/sentiment/analyze"
    payload = {
        "text": "I absolutely love this product! Best purchase ever!"
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Scores: {result['scores']}")
    else:
        print(f"Error: {response.status_code}")


# =====================================================
# Example 2: Batch Analysis
# =====================================================


def example_batch_analysis():
    """Example of batch sentiment analysis."""
    import requests
    import json

    url = "http://localhost:8000/api/sentiment/batch"

    texts = [
        "This is amazing! Love it so much!",
        "Terrible product, waste of money.",
        "Okay, nothing special but acceptable.",
        "Incredible! Exceeded my expectations!",
        "Horrible experience, will never buy again."
    ]

    payload = {
        "texts": texts
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        results = response.json()
        for result in results['results']:
            print(f"\nText: {result['text']}")
            print(f"  Sentiment: {result['sentiment']}")
            print(f"  Confidence: {result['confidence']:.2%}")
    else:
        print(f"Error: {response.status_code}")


# =====================================================
# Example 3: Get Metrics
# =====================================================


def example_get_metrics():
    """Example of getting overall metrics."""
    import requests
    import json

    # Get overview
    url = "http://localhost:8000/api/metrics/overview"
    response = requests.get(url)

    if response.status_code == 200:
        metrics = response.json()
        print("=== Overview Metrics ===")
        print(f"Total Analyses: {metrics['total_analyses']}")
        print(f"Positive Count: {metrics['positive_count']}")
        print(f"Negative Count: {metrics['negative_count']}")
        print(f"Average Score: {metrics['average_score']:.2f}")
        print(f"Most Common: {metrics['most_common_sentiment']}")

    # Get trend
    url = "http://localhost:8000/api/metrics/trend?days=7"
    response = requests.get(url)

    if response.status_code == 200:
        trend = response.json()
        print("\n=== 7-Day Trend ===")
        for day in trend['days']:
            print(f"{day}: Positive={trend['positive_counts'][trend['days'].index(day)]} "
                  f"Negative={trend['negative_counts'][trend['days'].index(day)]}")


# =====================================================
# Example 4: Text Processing Endpoints
# =====================================================


def example_text_processing():
    """Example of text cleaning and processing."""
    import requests
    import json

    url = "http://localhost:8000/api/text/clean"
    payload = {
        "text": "Check out this http://example.com product @username! #amazing!!! "
                "What a terrible experience!!! So disappointed!!!!!!"
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("=== Text Cleaning Result ===")
        print(f"Original Length: {result['original_length']}")
        print(f"Cleaned Length: {result['cleaned_length']}")
        print(f"Removed Tokens: {result['removed_tokens']}")


# =====================================================
# Example 5: Visualize with Plotly
# =====================================================


def example_visualization():
    """Example of creating visualizations with Plotly."""
    import requests
    import pandas as pd
    import plotly.express as px

    # Get trend data
    response = requests.get("http://localhost:8000/api/metrics/trend?days=7")
    trend_data = response.json()

    # Convert to DataFrame
    days = trend_data['days']
    positive_counts = trend_data['positive_counts']
    negative_counts = trend_data['negative_counts']

    df = pd.DataFrame({
        'date': days,
        'positive': positive_counts,
        'negative': negative_counts
    })

    # Create visualization
    fig = px.bar(
        df,
        x='date',
        y=['positive', 'negative'],
        barmode='group',
        title='Sentiment Trend Over 7 Days',
        labels={'positive': 'Count', 'variable': 'Sentiment'}
    )

    # Show or save
    fig.show()
    # fig.write_image("sentiment_trend.png", scale=2)


# =====================================================
# Example 6: Streamlit App Integration
# =====================================================


def example_streamlit_integration():
    """Example of integrating with Streamlit."""
    import streamlit as st
    import requests

    st.title("Sentiment Analysis Demo")

    # Text input
    text = st.text_area("Enter text to analyze:", height=150)

    if st.button("Analyze"):
        if not text:
            st.warning("Please enter some text!")
        else:
            # Analyze sentiment
            response = requests.post(
                "http://localhost:8000/api/sentiment/analyze",
                json={"text": text}
            )

            if response.status_code == 200:
                result = response.json()
                st.success(f"Sentiment: {result['sentiment'].capitalize()}")
                st.write(f"Confidence: {result['confidence']:.2%}")

                # Visualize scores
                scores = result['scores']
                st.write(
                    f"**Scores**: Positive: {scores['positive']:.2%}, "
                    f"Negative: {scores['negative']:.2%}"
                )


# =====================================================
# Run Examples
# =====================================================

if __name__ == "__main__":
    # Uncomment to run examples
    # example_api_usage()
    # example_batch_analysis()
    # example_get_metrics()
    # example_text_processing()
    # example_visualization()
    # example_streamlit_integration()

    print("Run individual examples from this file.")
    print("Make sure the API is running on http://localhost:8000")
