# 🌊 Social Media-Based Disaster Recovery and Resilience Analysis

This project analyzes social media data to monitor disaster-related events—particularly floods—and assess public sentiment, geospatial impact, and topic trends to support resilience and recovery efforts.

## 🔧 Features

* **Data Scraping**: Collect Reddit posts based on flood-related keywords and extract geolocation info.
* **Preprocessing**: Clean, tokenize, lemmatize, and extract hashtags/mentions from posts.
* **Classification**: Classify posts as flood-related using a Random Forest classifier with SMOTE balancing.
* **AI Insight**: Automatically generate public-friendly model accuracy explanations using Cohere AI.
* **Sentiment & Emotion Analysis**: Identify sentiment (positive, neutral, negative) and emotions (anger, fear, joy, etc.).
* **Time Series Analysis**: Visualize post trends over months/years.
* **Geospatial Mapping**: Plot a choropleth map of post density by UK region using shapefiles.
* **Topic Modeling**: Extract and trend key discussion topics using LDA.
* **Network Analysis**: Visualize co-occurrence networks of top hashtags and mentions.

## 🗂️ Project Structure

```bash
├── main.py                      # Streamlit dashboard app
├── scraper.py                  # Reddit data scraper using PRAW
├── data_preprocessor.py        # Text cleaning, tokenization, lemmatization
├── model.py                    # Flood classification model (Random Forest)
├── sentiment.py                # Sentiment & emotion analysis
├── time_series.py              # Yearly and monthly time-series visualization
├── geo_spatial.py              # Choropleth mapping using UK shapefiles
├── topic_modeling.py           # LDA-based topic modeling and visualization
├── network_analysis.py         # Mention & hashtag co-occurrence graphs
├── cohere_summary.py           # Natural language explanation of model accuracy
├── Datasets/                   # Directory for raw and preprocessed data
└── shapefile/                  # Shapefiles for geospatial analysis
```

## 🛠️ Setup Instructions

1. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Download Spacy Model**:

   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Set up Reddit API** in `scraper.py`:

   ```python
   REDDIT_CLIENT_ID = "your_client_id"
   REDDIT_CLIENT_SECRET = "your_client_secret"
   REDDIT_USER_AGENT = "your_username"
   ```

4. **Add your Cohere API key** in `main.py`:

   ```python
   cohere_api = "Your-API"
   ```

5. **Run the App**:

   ```bash
   streamlit run main.py
   ```

## 📊 Visual Output Examples

* Sentiment and emotion bar charts
* Choropleth maps of post concentration
* Network graphs of mentions and hashtags
* Time-series and topic trend line charts

## 📍 Use Case

This system helps disaster response teams and researchers:

* Monitor flood-related discourse on social platforms
* Track changes in public emotion over time
* Identify heavily impacted areas
* Extract urgent or recurring concerns through topic modeling

## 👨‍💻 Author

Developed by a Data Science graduate from TARUMT as part of a final-year project on social media-based disaster resilience monitoring.