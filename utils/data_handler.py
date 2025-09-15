# utils/data_handler.py
import pandas as pd

def load_comments(uploaded_file):
    """Read comments from uploaded CSV"""
    df = pd.read_csv(uploaded_file)
    return df

def save_results_to_csv(df, filename="sentiment_results.csv"):
    """Save results to CSV"""
    df.to_csv(filename, index=False)
    return filename
