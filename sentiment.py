import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from textblob import TextBlob
from collections import Counter
import nltk
from nltk.corpus import stopwords
from transformers import pipeline

# Force transformers to use PyTorch instead of TensorFlow
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", framework="pt", top_k=1)

def get_sentiment(text):
    """
    Perform sentiment analysis on the given text.
    """
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

def detect_emotion(text):
    """
    Perform emotion classification on the given text.
    """
    try:
        result = emotion_classifier(text)[0][0]['label']
    except:
        result = "Unknown"
    return result

def plot_sentiment_analysis(df):
    """
    Analyze and plot sentiment and emotion distribution of social media posts.
    """
    df['sentiment'] = df['Cleaned_Text'].apply(get_sentiment)
    sentiment_counts = df['sentiment'].value_counts()
    
    df['emotion'] = df['Cleaned_Text'].apply(detect_emotion)
    emotion_counts = df['emotion'].value_counts()
    
    # Display sentiment and emotion counts in Streamlit
    st.subheader("📊 Sentiment Analysis of Social Media Posts")
    st.dataframe(sentiment_counts)
    
    st.subheader("🎭 Emotion Analysis of Social Media Posts")
    st.dataframe(emotion_counts)
    
    # Plot sentiment distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Analysis of Social Media Posts")
    st.pyplot(fig)
    
    # Plot emotion distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(emotion_counts.index, emotion_counts.values)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.set_title("Emotion Analysis of Social Media Posts")
    ax.set_xticklabels(emotion_counts.index, rotation=45)
    st.pyplot(fig)

    # Convert timestamp to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    # Aggregate emotion counts over time (monthly)
    df['month'] = df['Timestamp'].dt.to_period("M")
    emotion_trends = df.groupby(['month', 'emotion']).size().unstack().fillna(0)
    
    # Plot psychological analysis trends over time
    fig, ax = plt.subplots(figsize=(12, 6))
    for emotion in emotion_trends.columns:
        ax.plot(emotion_trends.index.astype(str), emotion_trends[emotion], label=emotion, marker='o')
    
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Number of Mentions")
    ax.set_title("Psychological Emotion Trends Over Time")
    ax.legend()
    ax.set_xticks(range(len(emotion_trends.index)))  # Ensure fixed ticks
    ax.set_xticklabels(emotion_trends.index.astype(str), rotation=45, ha='right')
    st.pyplot(fig)