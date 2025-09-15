# utils/sentiment_visualizer.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def sentiment_distribution(df: pd.DataFrame):
    """Donut chart of sentiment distribution"""
    if df.empty:
        return go.Figure()
    fig = px.pie(
        df, names="sentiment", hole=0.5,
        color="sentiment",
        color_discrete_map={"Positive": "lightblue", "Negative": "pink", "Neutral": "lightgrey"}
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(title="😊 Sentiment Distribution")
    return fig

def sentiment_over_time(df: pd.DataFrame):
    """Line/Scatter chart of sentiments over time"""
    if df.empty or "date" not in df.columns:
        return go.Figure()
    df["date"] = pd.to_datetime(df["date"])
    fig = px.scatter(
        df, x="date", y="sentiment", color="sentiment",
        hover_data=["text", "author"],
        color_discrete_map={"Positive": "lightblue", "Negative": "pink", "Neutral": "lightgrey"}
    )
    fig.update_layout(title="📈 Sentiment Over Time", yaxis_title="Sentiment")
    return fig
