# utils/data_handler.py
import pandas as pd

def load_comments(uploaded_file):
    """Read comments from uploaded CSV and normalize column names robustly"""
    df = pd.read_csv(uploaded_file)

    # Normalize column names: lowercase + strip spaces
    df.columns = df.columns.str.strip().str.lower()

    # --- Handle Comment/Text column ---
    text_aliases = ["text", "comment", "comments", "feedback", "review", "message"]
    text_col = next((c for c in df.columns if c in text_aliases), None)
    if text_col:
        df.rename(columns={text_col: "text"}, inplace=True)
    else:
        raise ValueError("❌ Uploaded file must have a column like 'text', 'comment', 'comments', 'feedback', or 'review'")

    # --- Handle Author column ---
    author_aliases = ["author", "user", "username", "name"]
    author_col = next((c for c in df.columns if c in author_aliases), None)
    if author_col:
        df.rename(columns={author_col: "author"}, inplace=True)
    else:
        df["author"] = "Anonymous"

    # --- Handle Date column ---
    date_aliases = ["date", "created_at", "timestamp", "time"]
    date_col = next((c for c in df.columns if c in date_aliases), None)
    if date_col:
        df.rename(columns={date_col: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    else:
        df["date"] = pd.Timestamp.now().date()

    # Keep only relevant columns
    df = df[["text", "author", "date"]]

    return df

def save_results_to_csv(df, filename="sentiment_results.csv"):
    """Save results to CSV"""
    df.to_csv(filename, index=False)
    return filename
me
