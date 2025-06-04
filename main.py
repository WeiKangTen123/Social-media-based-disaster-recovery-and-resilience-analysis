import streamlit as st
import pandas as pd
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from scraper import scrape_flood_posts
from data_preprocessor import preprocess_flood_data
from model import train_random_forest
from geo_spatial import plot_disaster_post_distribution
from sentiment import plot_sentiment_analysis
from time_series import run_time_series_analysis
from network_analysis import extract_top_hashtags_mentions, build_network_graphs
from topic_modeling import lda_topic_modeling
from cohere_summary import generate_insight_from_accuracy
import google.generativeai as genai

# File paths
SCRAPED_DATA_FILE = "Datasets/social_media_data.csv"
PREPROCESSED_DATA_FILE = "Datasets/preprocessed_flood_data_test.csv"
SHAPEFILE_PATH = "shapefile/CTYUA_MAY_2023_UK_BGC.shp"
MAP_IMAGE_PATH = "shapefile/disaster_post_distribution.png"

# API
cohere_api = "Your-API"

# ------------------------
# Streamlit UI
# ------------------------

# Set page config
st.set_page_config(page_title="Disaster Resilience Analysis", layout="wide")

# Navigation Bar
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Home", "Scraping", "Preprocessing", "Classification", "Analysis"]
)

# Home Page
if menu == "Home":
    st.title("🌊 Disaster Recovery and Resilience Analysis System")
    st.markdown("Welcome to the Social Media Disaster Monitoring Dashboard!")
    st.write("This system scrapes Reddit, processes text, classifies posts, and analyzes emotional and geographic trends related to flood events.")

# Scraping Page
elif menu == "Scraping":
    st.title("🌐 Scrape Reddit Data")

    if st.button("Start Scraping Reddit"):
        st.write("⏳ Scraping Reddit posts from UK...")
        scrape_flood_posts(["UnitedKingdom"], limit=100, output_file=SCRAPED_DATA_FILE)
        st.success("✅ Scraping completed! Data saved.")

# Preprocessing Page
elif menu == "Preprocessing":
    st.title("🛠️ Preprocess Data")

    if st.button("Start Preprocessing"):
        if os.path.exists(SCRAPED_DATA_FILE):
            st.write("⏳ Preprocessing data...")
            preprocess_flood_data(SCRAPED_DATA_FILE, PREPROCESSED_DATA_FILE)
            st.success("✅ Preprocessing completed! Data is ready for classification.")
        else:
            st.warning("⚠️ No scraped data found. Please run Scraping first.")

# Classification Page
elif menu == "Classification":
    st.title("🔍 Classification and AI Insight")

    if os.path.exists(PREPROCESSED_DATA_FILE):
        if st.button("Run Classification Report"):
            df = pd.read_csv(PREPROCESSED_DATA_FILE)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

            st.write("🚀 Running Classification Report on the Full Dataset...")
            model, vectorizer, accuracy, classification_rep, df = train_random_forest(df)

            st.subheader(f"🎯 Model Accuracy: {accuracy:.2%}")

            st.subheader("🧠 AI-Generated Insight")
            with st.spinner("Generating insight using Cohere..."):
                try:
                    insight = generate_insight_from_accuracy(accuracy, cohere_api)
                    st.write(insight)
                except Exception as e:
                    st.error("❌ Failed to generate insight.")
                    st.exception(e)
    else:
        st.warning("⚠️ No preprocessed data found. Please run Preprocessing first.")

# Analysis Page
elif menu == "Analysis":
    st.title("📈 Disaster Psychological and Geospatial Analysis")

    if os.path.exists(PREPROCESSED_DATA_FILE):
        df = pd.read_csv(PREPROCESSED_DATA_FILE)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])

        # Time Series Analysis Section
        st.subheader("📅 Time Series Analysis")
        run_time_series_analysis(PREPROCESSED_DATA_FILE)

        # Analysis Option Section
        st.subheader("🧠 Psychological Analysis")

        analysis_option = st.radio("Select an analysis type:", ["Geospatial Analysis", "Sentiment Analysis", "Network Analysis", "Topic Modeling"])

        min_year = int(df['Timestamp'].dt.year.min())
        max_year = int(df['Timestamp'].dt.year.max())
        valid_years = list(range(min_year, max_year + 1))
        valid_months = list(range(1, 13))

        st.info(f"📝 Data available from {min_year} to {max_year}")

        col1, col2 = st.columns(2)
        with col1:
            start_year = st.selectbox("Start Year", valid_years)
            start_month = st.selectbox("Start Month", valid_months)
        with col2:
            end_year = st.selectbox("End Year", valid_years, index=len(valid_years)-1)
            end_month = st.selectbox("End Month", valid_months, index=11)

        if st.button("Run Selected Analysis"):
            start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
            end_date = pd.Timestamp(year=end_year, month=end_month, day=28)

            df_filtered = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]

            if df_filtered.empty:
                st.warning("⚠️ No data for the selected time range.")
            else:
                if analysis_option == "Geospatial Analysis":
                    if os.path.exists(SHAPEFILE_PATH):
                        st.write("🗺️ Generating geospatial map...")
                        plot_disaster_post_distribution(df_filtered, SHAPEFILE_PATH, save_path=MAP_IMAGE_PATH)
                    else:
                        st.warning("⚠️ Shapefile not found.")

                elif analysis_option == "Sentiment Analysis":
                    st.write("📊 Performing sentiment and emotion analysis...")
                    plot_sentiment_analysis(df_filtered)

                elif analysis_option == "Network Analysis":
                    st.subheader("🔥 Hashtag and Mention Networks")
                    hashtag_df, mention_df = extract_top_hashtags_mentions(df_filtered)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("📌 **Top Hashtags**")
                        st.dataframe(hashtag_df)
                    with col2:
                        st.write("👥 **Top Mentions**")
                        st.dataframe(mention_df)

                    st.subheader("🔗 Visualizing Networks")
                    build_network_graphs(df_filtered)

                elif analysis_option == "Topic Modeling":
                    st.subheader("📝 Topic Modeling Results")
                    topics_df, df_filtered, lda_model = lda_topic_modeling(df_filtered)

                    st.write("📌 **Extracted Topics**")
                    st.dataframe(topics_df)

                    st.write("📋 **Topic Assignments for Posts**")
                    st.dataframe(df_filtered[["Cleaned_Text", "Topic", "Topic Meaning"]])
    else:
        st.warning("⚠️ No preprocessed data found. Please run Preprocessing first.")
