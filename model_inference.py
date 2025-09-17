import pandas as pd
from transformers import pipeline

# Load model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
)

# Load keywords from CSV
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

def adjust_sentiment(text: str, sentiment: str, score: float) -> tuple:
    """
    Refine sentiment using keyword counts + model confidence.
    Returns (main_sentiment, sub_sentiment).
    """
    text_lower = text.lower()

    pos_hits = sum(word in text_lower for word in POSITIVE_KEYWORDS)
    neg_hits = sum(word in text_lower for word in NEGATIVE_KEYWORDS)
    neu_hits = sum(word in text_lower for word in NEUTRAL_KEYWORDS)

    main_sentiment = sentiment
    sub_sentiment = sentiment

    # Neutral refinement
    if sentiment == "Neutral":
        if neg_hits > pos_hits:
            sub_sentiment = "Neutral (Dominantly Negative)"
        elif pos_hits > neg_hits:
            sub_sentiment = "Neutral (Dominantly Positive)"
        else:
            sub_sentiment = "Neutral (Pure Neutral)"

    # Positive overridden if negatives dominate
    elif sentiment == "Positive":
        if neg_hits > pos_hits and score < 0.95:
            sub_sentiment = "Neutral (Dominantly Negative)"

    # Negative overridden if positives dominate
    elif sentiment == "Negative":
        if pos_hits > neg_hits and score < 0.95:
            sub_sentiment = "Neutral (Dominantly Positive)"

    return main_sentiment, sub_sentiment


def analyze_sentiment(text: str) -> dict:
    """Analyze single comment with refined Neutral handling."""
    result = sentiment_pipeline(text)[0]
    label = result["label"]

    if label == "LABEL_0":
        sentiment = "Negative"
    elif label == "LABEL_1":
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    main_sentiment, sub_sentiment = adjust_sentiment(text, sentiment, result["score"])

    return {
        "text": text,
        "sentiment_main": main_sentiment,
        "sentiment_sub": sub_sentiment,
        "score": round(result["score"], 3)
    }


def analyze_batch(comments: list) -> list:
    """Analyze batch of comments with refined Neutral handling."""
    results = []
    for comment in comments:
        if isinstance(comment, dict) and "text" in comment:
            text = comment["text"]
        else:
            text = str(comment)
        results.append(analyze_sentiment(text))
    return results
