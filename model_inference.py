import pandas as pd
from transformers import pipeline

# Load Hugging Face sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
)

# ---------------------------
# Load keywords from CSV files
# ---------------------------
def load_keywords(path: str) -> list:
    try:
        df = pd.read_csv(path)
        return [str(x).lower() for x in df["keyword"].dropna().tolist()]
    except Exception as e:
        print(f"⚠️ Could not load {path}: {e}")
        return []

NEGATIVE_KEYWORDS = load_keywords("keywords_negative.csv")
POSITIVE_KEYWORDS = load_keywords("keywords_positive.csv")
NEUTRAL_KEYWORDS  = load_keywords("keywords_neutral.csv")


# ---------------------------
# Helper for refining Neutral
# ---------------------------
def adjust_sentiment(text: str, sentiment_main: str) -> str:
    """
    Refine Neutral into Neutral (Dominantly Positive/Negative)
    using keyword matches. If no match, stays Neutral.
    """
    text_lower = text.lower()

    if sentiment_main == "Neutral":
        if any(word in text_lower for word in NEUTRAL_KEYWORDS):
            return "Neutral"
        if any(word in text_lower for word in NEGATIVE_KEYWORDS):
            return "Neutral (Dominantly Negative)"
        if any(word in text_lower for word in POSITIVE_KEYWORDS):
            return "Neutral (Dominantly Positive)"
    return sentiment_main


# ---------------------------
# Single comment analysis
# ---------------------------
def analyze_sentiment(text: str) -> dict:
    """Analyze a single comment and return both main & refined labels."""
    result = sentiment_pipeline(text)[0]
    label = result["label"]

    # Map base labels
    if label == "LABEL_0":
        sentiment_main = "Negative"
    elif label == "LABEL_1":
        sentiment_main = "Neutral"
    else:
        sentiment_main = "Positive"

    # Refine Neutral into sub category
    sentiment_sub = adjust_sentiment(text, sentiment_main)

    return {
        "text": text,
        "sentiment_main": sentiment_main,
        "sentiment_sub": sentiment_sub,
        "score": round(result["score"], 3),
    }


# ---------------------------
# Batch analysis
# ---------------------------
def analyze_batch(comments: list) -> list:
    """Analyze multiple comments with both main & sub sentiment labels."""
    results = []
    for comment in comments:
        if isinstance(comment, dict) and "text" in comment:
            text = comment["text"]
        else:
            text = str(comment)
        results.append(analyze_sentiment(text))
    return results
