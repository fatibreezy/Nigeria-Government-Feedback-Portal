import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import os
from textblob import TextBlob
from llama_cpp import Llama

@st.cache_resource
def load_model():
    return Llama(model_path="Nous-Hermes-2-Mistral-7B.Q4_K_M.gguf", n_ctx=2048)

llm = load_model()

st.set_page_config(page_title="Nigeria Gov Feedback Portal", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f4f6f8; }
    .stButton>button { background-color: #007BFF; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🇳🇬 Nigeria Government Feedback Portal")
st.subheader("📢 A citizen feedback system powered by AI")

tab1, tab2, tab3 = st.tabs(["🗨️ Chat with AI", "📊 Sentiment Dashboard", "📋 Submit Feedback"])

# --- CHATBOT ---
with tab1:
    st.header("Ask Government-Related Questions")
    user_input = st.text_input("Enter your question:")
    if user_input:
        prompt = f"[INST] {user_input} [/INST]"
        response = llm(prompt, max_tokens=200)
        answer = response['choices'][0]['text'].strip()
        st.markdown(f"**AI Response:** {answer}")

# --- SENTIMENT ANALYSIS ---
feedback_data_file = "feedback_data.csv"
if os.path.exists(feedback_data_file):
    df = pd.read_csv(feedback_data_file)
else:
    df = pd.DataFrame(columns=["timestamp", "feedback", "sentiment"])

with tab2:
    st.header("📈 Real-Time Sentiment Analysis")
    if not df.empty:
        fig = px.histogram(df, x="sentiment", color="sentiment", title="Feedback Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.tail(10))
        st.download_button("📥 Download Feedback", df.to_csv(index=False), "feedback_data.csv")
    else:
        st.info("No feedback submitted yet.")

# --- FEEDBACK SUBMISSION ---
with tab3:
    st.header("📝 Share Your Thoughts")
    feedback = st.text_area("Your feedback", placeholder="Type your suggestions or concerns...")
    name = st.text_input("Optional: Your name")
    if st.button("Submit"):
        if feedback:
            sentiment = TextBlob(feedback).sentiment.polarity
            label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
            new_entry = pd.DataFrame([[datetime.datetime.now(), feedback, label]], columns=["timestamp", "feedback", "sentiment"])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(feedback_data_file, index=False)
            st.success("✅ Thank you! Your feedback has been recorded.")
        else:
            st.error("Feedback cannot be empty!")

# --- ADMIN PANEL ---
with st.sidebar:
    st.title("🔐 Admin Panel")
    pwd = st.text_input("Enter admin password", type="password")
    if pwd == "nigeria2025":
        st.success("Access granted.")
        st.write("Latest feedback data:")
        st.dataframe(df)
        st.download_button("📥 Download All Feedback", df.to_csv(index=False), "all_feedback.csv")
    else:
        st.warning("Enter password to access admin panel.")
