# utils/data_handler.py
import pandas as pd

def load_comments(uploaded_file):
    """Read comments from uploaded CSV and normalize column names"""
    df = pd.read_csv(uploaded_file)

    # Standardize column names (lowercase, strip spaces)
    df.columns = df.columns.str.strip().str.lower()

    # Try to map to expected names
    if "comment" in df.columns:
        df.rename(columns={"comment": "text"}, inplace=True)
    elif "comments" in df.columns:
        df.rename(columns={"comments": "text"}, inplace=True)
    elif "feedback" in df.columns:
        df.rename(columns={"feedback": "text"}, inplace=True)
    elif "review" in df.columns:
        df.rename(columns={"review": "text"}, inplace=True)

    # Add defaults if missing
    if "text" not in df.columns:
        raise ValueError("Uploaded file must have a column like 'comment' or 'text'")

    if "author" not in df.columns:
        df["author"] = "Anonymous"
    if "date" not in df.columns:
        df["date"] = pd.Timestamp.now().date()

    return df

def save_results_to_csv(df, filename="sentiment_results.csv"):
    """Save results to CSV"""
    df.to_csv(filename, index=False)
    return filename
