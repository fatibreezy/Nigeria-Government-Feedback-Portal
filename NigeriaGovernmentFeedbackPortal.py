import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datetime

# Load chatbot model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model

tokenizer, model = load_model()

# Function to generate chatbot response
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]

# Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0: return 'Positive'
    elif polarity < 0: return 'Negative'
    else: return 'Neutral'

# Load or create CSV
def load_feedback():
    try:
        return pd.read_csv("feedback.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Name", "Email", "Location", "Message", "Sentiment", "Timestamp"])

def save_feedback(df):
    df.to_csv("feedback.csv", index=False)

def load_suggestions():
    try:
        return pd.read_csv("suggestions.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Suggestion", "Timestamp"])

def save_suggestions(df):
    df.to_csv("suggestions.csv", index=False)

# UI
st.set_page_config(page_title="Nigeria Gov Feedback", layout="wide")
st.title("🇳🇬 Nigeria Government Feedback & Updates Portal")

menu = st.sidebar.selectbox("Choose a section", ["Chat with AI", "Submit Feedback", "Suggestions", "Government Updates", "Admin Dashboard"])

# Chatbot
if menu == "Chat with AI":
    st.subheader("Ask our AI Assistant anything")
    user_input = st.text_input("Your question:")
    if st.button("Ask"):
        if user_input:
            response = generate_response(user_input)
            st.success(response)

# Feedback
elif menu == "Submit Feedback":
    st.subheader("Submit Your Feedback")
    name = st.text_input("Name (Optional)")
    email = st.text_input("Email (Optional)")
    location = st.text_input("Location (Optional)")
    message = st.text_area("Your Message")

    if st.button("Submit Feedback"):
        if message:
            sentiment = analyze_sentiment(message)
            timestamp = datetime.datetime.now()
            new_data = {"Name": name, "Email": email, "Location": location, "Message": message,
                        "Sentiment": sentiment, "Timestamp": timestamp}
            df = load_feedback()
            df = df.append(new_data, ignore_index=True)
            save_feedback(df)
            st.success("Feedback submitted!")

# Suggestions
elif menu == "Suggestions":
    st.subheader("Suggest Something to the Government")
    suggestion = st.text_area("Enter your suggestion here")
    if st.button("Submit Suggestion"):
        if suggestion:
            timestamp = datetime.datetime.now()
            df = load_suggestions()
            df = df.append({"Suggestion": suggestion, "Timestamp": timestamp}, ignore_index=True)
            save_suggestions(df)
            st.success("Suggestion submitted!")

# Government Updates
elif menu == "Government Updates":
    st.subheader("Latest Government Updates")
    updates = [
        {"title": "3MTT Skills Training Launch", "desc": "Massive tech training rollout across Nigeria"},
        {"title": "NYSC Reforms", "desc": "New allowances and improved digital skills curriculum"},
        {"title": "Education Budget Increased", "desc": "Government increases education sector allocation"},
        {"title": "Health Insurance Plan", "desc": "New health insurance for informal workers"},
        {"title": "Youths in Agritech Program", "desc": "Funding and tools provided for smart farming"},
        {"title": "Public WiFi Initiative", "desc": "Free WiFi in markets and parks"},
        {"title": "Data Protection Bill", "desc": "Improved data privacy for Nigerians"},
        {"title": "Digital Census Launch", "desc": "Smart census to improve accuracy"},
        {"title": "Tech Export Drive", "desc": "Promoting Nigerian software exports"},
        {"title": "SME Grants", "desc": "New round of capital for small businesses"},
    ]

    for update in updates:
        st.markdown(f"**{update['title']}**")
        st.write(update["desc"])
        st.markdown("---")

# Admin Dashboard
elif menu == "Admin Dashboard":
    st.subheader("Admin Dashboard")
    st.info("Download or analyze feedback and suggestions.")

    df_feedback = load_feedback()
    df_suggestions = load_suggestions()

    st.markdown("### Feedback Analysis")
    if not df_feedback.empty:
        sentiment_counts = df_feedback["Sentiment"].value_counts()
        st.bar_chart(sentiment_counts)
        st.dataframe(df_feedback.tail(10))
        st.download_button("Download Feedback CSV", df_feedback.to_csv(index=False), "feedback.csv")

    st.markdown("### Suggestions Overview")
    if not df_suggestions.empty:
        st.dataframe(df_suggestions.tail(10))
        st.download_button("Download Suggestions CSV", df_suggestions.to_csv(index=False), "suggestions.csv")
