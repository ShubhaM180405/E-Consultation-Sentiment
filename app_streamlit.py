# app_streamlit.py
import streamlit as st
import pandas as pd
from datetime import date

from model_inference import analyze_sentiment, analyze_batch
from utils import sentiment_visualizer as viz
from utils import data_handler as dh

st.set_page_config(page_title="E-Consultation Sentiment Analysis", layout="wide")

# --- SESSION STATE ---
if "comments" not in st.session_state:
    st.session_state["comments"] = []

# --- SIDEBAR ---
st.sidebar.title("➕ Add Comments")
input_method = st.sidebar.radio("Choose input method:", ["Single Comment", "Multiple Comments", "Upload File"])

if input_method == "Single Comment":
    text = st.sidebar.text_area("Comment text:")
    author = st.sidebar.text_input("Author (optional):")
    date_input = st.sidebar.date_input("Date", date.today())
    if st.sidebar.button("Add Comment"):
        st.session_state["comments"].append({"text": text, "author": author, "date": str(date_input)})

elif input_method == "Multiple Comments":
    texts = st.sidebar.text_area("Enter multiple comments (one per line):")
    author = st.sidebar.text_input("Author (optional):")
    date_input = st.sidebar.date_input("Date", date.today())
    if st.sidebar.button("Add Comments"):
        for t in texts.split("\n"):
            if t.strip():
                st.session_state["comments"].append({"text": t.strip(), "author": author, "date": str(date_input)})

# app_streamlit.py (excerpt)
elif input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV / XLSX / TXT with comments", 
        type=["csv", "txt", "xlsx", "xls"]
    )
    if uploaded_file is not None:
        try:
            df = dh.load_comments(uploaded_file)
            st.session_state["comments"].extend(df.to_dict("records"))
            st.success(f"✅ Loaded {len(df)} comments from file")
        except Exception as e:
            st.error(f"❌ Could not load file: {e}")



if st.sidebar.button("Clear All Comments"):
    st.session_state["comments"] = []

# --- MAIN APP ---
st.title("💬 E-Consultation Sentiment Analysis")
st.write("Analyze public sentiment from consultation comments with AI-powered insights")

tabs = st.tabs(["📊 Dashboard", "📈 Analytics", "💬 Comments View", "🔍 Insights", "📑 Reports"])

# TAB 1: Dashboard
with tabs[0]:
    st.subheader("Overview")
    if st.button("Analyze All Comments"):
        results = analyze_batch(st.session_state["comments"])
        st.session_state["results"] = results
        st.success("✅ Analysis completed!")

    if "results" in st.session_state:
        df = pd.DataFrame(st.session_state["results"])

        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        total = len(df)
        positive = len(df[df["sentiment"] == "Positive"])
        negative = len(df[df["sentiment"] == "Negative"])
        neutral = len(df[df["sentiment"] == "Neutral"])

        col1.metric("😊 Positive", positive, f"{positive/total:.1%}")
        col2.metric("😞 Negative", negative, f"{negative/total:.1%}")
        col3.metric("😐 Neutral", neutral, f"{neutral/total:.1%}")
        col4.metric("📊 Total", total)

        # Charts with dark theme
        c1, c2 = st.columns(2)
        fig_dist = viz.sentiment_distribution(df)
        fig_time = viz.sentiment_over_time(df)

        # Apply dark background
        for fig in [fig_dist, fig_time]:
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(color="white")
            )

        c1.plotly_chart(fig_dist, use_container_width=True)
        c2.plotly_chart(fig_time, use_container_width=True)

# TAB 2: Analytics
with tabs[1]:
    if "results" in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state["results"]).describe(include="all"))

# TAB 3: Comments View
with tabs[2]:
    if "results" in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state["results"]))

# TAB 4: Insights
with tabs[3]:
    st.info("✨ Future: Add word clouds, key phrases, etc.")

# TAB 5: Reports
with tabs[4]:
    if "results" in st.session_state:
        df = pd.DataFrame(st.session_state["results"])
        st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="sentiment_results.csv", mime="text/csv")
