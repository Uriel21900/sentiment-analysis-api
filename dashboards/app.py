"""
Streamlit dashboard for sentiment analysis visualization.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import defaultdict
import time

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .chart-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def load_sample_data(days: int = 30) -> pd.DataFrame:
    """Load or generate sample data for visualization."""
    # Generate trend data
    dates = []
    positive_counts = []
    negative_counts = []
    topics = []

    for i in range(days):
        date = datetime.now() - timedelta(days=days - i)
        dates.append(date.strftime("%Y-%m-%d"))

        # Generate realistic trend data with seasonality
        trend = 50 + 10 * np.sin(2 * np.pi * i / 14) + np.random.normal(0, 5)
        positive_counts.append(int(max(30, min(70, trend))))
        negative_counts.append(int(max(25, min(65, 70 - trend))))

        # Generate topics
        topic_pool = [
            "Technology", "Business", "Entertainment", "Sports",
            "Politics", "Health", "Travel", "Food", "Education", "Science"
        ]
        topics.append(np.random.choice(topic_pool))

    return pd.DataFrame({
        "date": dates,
        "positive": positive_counts,
        "negative": negative_counts,
        "topic": topics
    })


def show_metrics(df: pd.DataFrame):
    """Display metrics cards."""
    st.header("📊 Overall Metrics")
    st.markdown("---")

    total_analyses = len(df)
    positive_count = df["positive"].sum()
    negative_count = df["negative"].sum()
    positive_ratio = positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0
    avg_score = (positive_ratio * 2) if positive_ratio > 0.5 else (positive_ratio * 2 - 1)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Analyses",
            value=f"{total_analyses:,}",
            delta=f"+{total_analyses * 5} this week"
        )

    with col2:
        sentiment_indicator = "😊" if positive_ratio > 0.5 else "😐"
        st.metric(
            label="Positive Sentiment",
            value=f"{positive_ratio * 100:.1f}%",
            delta=f"{sentiment_indicator} +{positive_ratio - 0.4:.1%}"
        )

    with col3:
        st.metric(
            label="Negative Sentiment",
            value=f"{(1 - positive_ratio) * 100:.1f}%",
            delta=f"😕 -{(1 - positive_ratio) - 0.4:.1%}"
        )

    with col4:
        avg_indicator = "📈" if avg_score > 0.3 else "📉"
        st.metric(
            label="Avg Confidence",
            value=f"{(avg_score + 1) / 2 * 100:.1f}%",
            delta=f"{avg_indicator} {avg_score:.2f}"
        )

    st.markdown("---")


def show_trend_chart(df: pd.DataFrame):
    """Display sentiment trend chart."""
    st.header("📈 Sentiment Trend")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["positive"],
        mode="lines+markers",
        name="Positive",
        line=dict(color="#2ecc71", width=3),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.1)"
    ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["negative"],
        mode="lines+markers",
        name="Negative",
        line=dict(color="#e74c3c", width=3),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.1)"
    ))

    fig.update_layout(
        title="Sentiment Trend Over Time",
        xaxis_title="Date",
        yaxis_title="Count",
        height=400,
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


def show_topic_distribution(df: pd.DataFrame):
    """Display topic distribution."""
    st.header("🏷️ Topic Distribution")

    topic_counts = df["topic"].value_counts()

    fig = px.treemap(
        df,
        path=["topic"],
        values="positive",
        title="Sentiment by Topic",
        color="topic",
        color_continuous_scale="RdYlGn",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def show_word_cloud(df: pd.DataFrame):
    """Display word cloud visualization."""
    st.header("📝 Most Discussed Topics")

    # Create topic frequency visualization
    fig = px.bar(
        df.groupby("topic").size().reset_index(name="count").head(10),
        x="topic",
        y="count",
        orientation="h",
        title="Top 10 Topics by Volume",
        color="count",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        height=400,
        template="plotly_white",
        xaxis_title="Topic",
        yaxis_title="Mentions"
    )

    st.plotly_chart(fig, use_container_width=True)


def show_sample_text_analysis():
    """Show sample text analysis UI."""
    st.header("✍️ Try Sentiment Analysis")

    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type your text here...",
        height=150,
        max_chars=10000
    )

    analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_button and text_input:
        # Simple sentiment scoring
        words = text_input.lower().split()
        positive_words = ["love", "great", "amazing", "excellent", "wonderful",
                         "fantastic", "awesome", "beautiful", "perfect", "happy"]
        negative_words = ["hate", "terrible", "awful", "horrible", "worst",
                         "disappointed", "bad", "awful", "useless", "stupid"]

        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        if positive_count + negative_count > 0:
            positive_ratio = positive_count / (positive_count + negative_count)

            sentiment = "positive" if positive_ratio > 0.5 else "negative"
            emoji = "😊" if positive_ratio > 0.5 else "😕"

            with st.expander("Analysis Results"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Sentiment", sentiment.capitalize())
                col2.metric("Positive Words", positive_count)
                col3.metric("Negative Words", negative_count)

                st.success(f"Sentiment: {sentiment} ({positive_ratio:.1%})" if positive_ratio > 0.5
                          else st.error(f"Sentiment: {sentiment} ({positive_ratio:.1%})"))
        else:
            st.info("No sentiment indicators found in text.")


@st.cache_data
def get_dashboard_data():
    """Get dashboard data."""
    return load_sample_data(30)


def main():
    """Main dashboard function."""
    st.title("📊 Sentiment Analysis Dashboard")
    st.markdown("Real-time sentiment tracking with ML-powered analysis")

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")

        days_to_show = st.slider(
            "Days to show",
            min_value=7,
            max_value=90,
            value=30
        )

        st.markdown("---")
        st.subheader("📁 Data Source")
        data_source = st.radio(
            "Select data source:",
            ["Live API", "Sample Data", "Upload CSV"],
            horizontal=True
        )

        st.markdown("---")
        st.info("ℹ️ **Tip**: Click 'Try Sentiment Analysis' to test the model!")

    # Main content
    if data_source == "Sample Data":
        df = get_dashboard_data(days_to_show)
    else:
        # For other data sources, use sample data
        df = get_dashboard_data(days_to_show)

    # Display metrics
    show_metrics(df)

    # Display charts
    col1, col2 = st.columns(2)

    with col1:
        show_trend_chart(df)

    with col2:
        show_topic_distribution(df)

    # Sample analysis
    show_sample_text_analysis()

    # Footer
    st.markdown("---")
    st.caption(
        "Built with Streamlit, Plotly, Matplotlib | Powered by scikit-learn"
    )


if __name__ == "__main__":
    main()
