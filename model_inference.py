from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load Cardiff NLP 3-class sentiment model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Mapping HuggingFace labels -> Human-readable
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def analyze_sentiment(text: str):
    """Analyze single comment sentiment (3-class)"""
    result = sentiment_pipeline(text, truncation=True)[0]
    label = label_map.get(result["label"], result["label"])
    score = round(result["score"], 3)

    return {"text": text, "sentiment": label, "score": score}

def analyze_batch(comments: list):
    """Analyze multiple comments (expects list of dicts with 'text')"""
    results = []
    for comment in comments:
        analysis = analyze_sentiment(comment["text"])
        analysis["author"] = comment.get("author", "Anonymous")
        analysis["date"] = comment.get("date", None)
        results.append(analysis)
    return results
