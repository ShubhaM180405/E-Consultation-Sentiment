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

def adjust_sentiment(text: str, sentiment: str, score: float) -> str:
    """Refine sentiment using keyword counts + model confidence."""
    text_lower = text.lower()

    pos_hits = sum(word in text_lower for word in POSITIVE_KEYWORDS)
    neg_hits = sum(word in text_lower for word in NEGATIVE_KEYWORDS)
    neu_hits = sum(word in text_lower for word in NEUTRAL_KEYWORDS)

    # If base is Neutral → refine normally
    if sentiment == "Neutral":
        if neg_hits > pos_hits:
            return "Neutral (Dominantly Negative)"
        elif pos_hits > neg_hits:
            return "Neutral (Dominantly Positive)"
        else:
            return "Neutral"

    # If base is Positive but negatives dominate strongly
    if sentiment == "Positive":
        if neg_hits > pos_hits:
            if score < 0.95:  # only override if model isn't extremely confident
                return "Neutral (Dominantly Negative)"

    # If base is Negative but positives dominate strongly
    if sentiment == "Negative":
        if pos_hits > neg_hits:
            if score < 0.95:
                return "Neutral (Dominantly Positive)"

    return sentiment

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

    # Apply keyword adjustment
    sentiment = adjust_sentiment(text, sentiment, result["score"])

    return {"text": text, "sentiment": sentiment, "score": round(result["score"], 3)}

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
