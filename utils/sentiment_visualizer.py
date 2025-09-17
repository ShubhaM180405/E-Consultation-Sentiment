import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def sentiment_distribution(df: pd.DataFrame):
    """Donut chart of sub-sentiment distribution"""
    if df.empty:
        return go.Figure()
    fig = px.pie(
        df, names="sentiment_sub", hole=0.5,
        color="sentiment_sub",
        color_discrete_map={
            "Positive": "lightblue",
            "Negative": "red",
            "Neutral (Pure Neutral)": "grey",
            "Neutral (Dominantly Negative)": "orange",
            "Neutral (Dominantly Positive)": "green"
        }
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(title="😊 Sub-Sentiment Distribution")
    return fig

def sentiment_over_time(df: pd.DataFrame):
    """Scatter plot of sub-sentiments over time"""
    if df.empty or "date" not in df.columns:
        return go.Figure()
    df["date"] = pd.to_datetime(df["date"])
    fig = px.scatter(
        df, x="date", y="sentiment_sub", color="sentiment_sub",
        hover_data=["text"],
        color_discrete_map={
            "Positive": "lightblue",
            "Negative": "red",
            "Neutral (Pure Neutral)": "grey",
            "Neutral (Dominantly Negative)": "orange",
            "Neutral (Dominantly Positive)": "green"
        }
    )
    fig.update_layout(title="📈 Sub-Sentiment Over Time", yaxis_title="Sentiment")
    return fig
