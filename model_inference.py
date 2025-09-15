# model_inference.py
from transformers import pipeline

# Load Hugging Face pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text: str):
    """Analyze single comment sentiment"""
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']

    if label == "POSITIVE":
        sentiment = "Positive"
    elif label == "NEGATIVE":
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {"text": text, "sentiment": sentiment, "score": round(score, 3)}

def analyze_batch(comments: list):
    """Analyze multiple comments"""
    results = []
    for comment in comments:
        analysis = analyze_sentiment(comment["text"])
        analysis["author"] = comment.get("author", "Anonymous")
        analysis["date"] = comment.get("date", None)
        results.append(analysis)
    return results
