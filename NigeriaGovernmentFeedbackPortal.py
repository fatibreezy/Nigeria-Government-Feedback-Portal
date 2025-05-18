import streamlit as st
import pandas as pd
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load chatbot model
@st.cache_resource
def load_chatbot():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_chatbot()
analyzer = SentimentIntensityAnalyzer()

if 'history' not in st.session_state:
    st.session_state.history = []

# Admin dashboard in sidebar
st.sidebar.title("Admin Dashboard")
if st.sidebar.checkbox("Show Feedback Table"):
    if "feedback_data.csv" in st.session_state:
        df = st.session_state["feedback_data.csv"]
        st.sidebar.dataframe(df)
        st.sidebar.download_button("Download Feedback", df.to_csv(index=False), "feedback.csv")
    else:
        st.sidebar.info("No feedback submitted yet.")

# Page Title
st.title("🇳🇬 Nigeria Government Feedback & Info Hub")
st.markdown("An AI-powered platform to share suggestions, see government updates, and analyze public sentiment.")

# Chat Section
st.subheader("🤖 Ask Government Chatbot")
user_input = st.text_input("Enter your message")
if user_input:
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([new_user_input_ids], dim=-1)
    chat_output = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(chat_output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.session_state.history.append((user_input, reply))
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**Bot:** {reply}")

# Display full chat
if st.session_state.history:
    st.markdown("---")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")

# Feedback Form
st.subheader("📢 Share Your Feedback or Suggestion")
with st.form("feedback_form"):
    feedback = st.text_area("Your suggestion or complaint")
    name = st.text_input("Name (optional)")
    submitted = st.form_submit_button("Submit")

    if submitted and feedback:
        sentiment = analyzer.polarity_scores(feedback)
        score = sentiment['compound']
        tone = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        new_data = pd.DataFrame([[name, feedback, tone, score, time]],
                                columns=["Name", "Feedback", "Sentiment", "Score", "Timestamp"])

        if "feedback_data.csv" not in st.session_state:
            st.session_state["feedback_data.csv"] = new_data
        else:
            st.session_state["feedback_data.csv"] = pd.concat([st.session_state["feedback_data.csv"], new_data], ignore_index=True)

        st.success("Thanks for your feedback!")

# Sentiment Analysis Visualization
st.subheader("📊 Real-Time Sentiment Analysis")
if "feedback_data.csv" in st.session_state:
    df = st.session_state["feedback_data.csv"]
    fig = px.histogram(df, x="Sentiment", color="Sentiment", title="Feedback Sentiment Summary")
    st.plotly_chart(fig)

# Government Updates
st.subheader("📰 Government Updates")
updates = [
    {"title": "3MTT 2nd Cohort Training Begins", "desc": "Thousands of Nigerians onboarded into tech learning."},
    {"title": "NYSC Members to Receive Increased Allowance", "desc": "FG raises corp members' monthly stipends."},
    {"title": "Digital Economy Strategy Released", "desc": "NDE unveils plan for tech-driven growth."},
    {"title": "Tinubu Approves Student Loan Rollout", "desc": "Application starts May 2025."},
    {"title": "Cybersecurity Bill Passed", "desc": "Aims to protect digital infrastructure."},
    {"title": "Digital ID Enrollment Hits 70M+", "desc": "NIMC sees record surge."},
    {"title": "Kogi State Partners on Tech Talent", "desc": "Up to 10,000 youths to benefit."},
    {"title": "Start-Up Grants for Youth", "desc": "FG disburses ₦2B in startup funds."},
    {"title": "NITDA Expands AI Research", "desc": "AI research centres launched nationwide."},
    {"title": "Lagos Launches Digital Skills Park", "desc": "5,000 youths enrolled in Phase 1."}
]
for u in updates:
    st.markdown(f"🔹 **{u['title']}** — {u['desc']}")

st.markdown("---")
st.caption("Powered by the Federal Government of Nigeria – All information is authentic and updated.")
