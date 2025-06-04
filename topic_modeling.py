from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def lda_topic_modeling(df, text_column='Cleaned_Text', timestamp_column='Timestamp', n_topics=5, max_features=1000):
    """
    Applies LDA topic modeling to extract key topics from text data and visualizes topic distribution and trends over time in Streamlit.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing text data.
        text_column (str): The column name containing cleaned text.
        timestamp_column (str): The column containing timestamps for trend analysis.
        n_topics (int): Number of topics to extract.
        max_features (int): Maximum number of features for vectorization.
    
    Returns:
        tuple: A DataFrame of topic words, an updated DataFrame with topic assignments, and an LDA model.
    """
    # Vectorize text data
    vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(df[text_column].dropna())
    
    # Apply LDA for topic modeling
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    # Get the top words for each topic
    words = vectorizer.get_feature_names_out()
    topics = {}
    for i, topic in enumerate(lda.components_):
        top_words = [words[j] for j in topic.argsort()[-10:]]
        topics[f"Topic {i+1}"] = top_words
    
    # Convert topics into a DataFrame
    topics_df = pd.DataFrame(topics)
    
    # Define topic meanings based on extracted words
    topic_meanings = {
        "Topic 1": "Disaster Relief & Needs",
        "Topic 2": "Evacuation & Assistance",
        "Topic 3": "Stranded People & Transport Issues",
        "Topic 4": "Live Reporting & Rescue Efforts",
        "Topic 5": "Personal Reactions to the Disaster"
    }
    
    # Add topic meanings to the topics DataFrame
    topics_df.loc[len(topics_df)] = topic_meanings.values()
    
    # Assign topic labels to the main dataframe
    topic_assignments = lda.transform(X).argmax(axis=1)  # Get dominant topic for each text
    df = df.dropna(subset=[text_column]).reset_index(drop=True)  # Ensure index alignment
    df["Topic"] = topic_assignments + 1  # Convert zero-based index to human-friendly topic number
    df["Topic Meaning"] = df["Topic"].map(lambda x: topic_meanings.get(f"Topic {x}", "Unknown"))
    
    # Display topic distribution in Streamlit
    topic_counts = df['Topic'].value_counts()
    topic_labels = [topic_meanings.get(f"Topic {topic}", f"Topic {topic}") for topic in topic_counts.index]
    
    st.subheader("📊 Distribution of Topics in Social Media Posts")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=topic_labels, y=topic_counts.values, palette="Blues_d", ax=ax)
    plt.xlabel("Topic Name")
    plt.ylabel("Count")
    plt.title("Distribution of Topics in Social Media Posts")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    
    # Convert timestamp to datetime format if not already
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
    
    # Aggregate topic counts over time
    df['month'] = df[timestamp_column].dt.to_period('M')
    topic_trends = df.groupby(['month', 'Topic']).size().unstack().fillna(0)
    
    # Display topic trends in Streamlit
    st.subheader("📈 Topic Trends Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    for topic in topic_trends.columns:
        ax.plot(topic_trends.index.astype(str), topic_trends[topic], label=topic_meanings.get(f"Topic {topic}", f"Topic {topic}"))
    
    plt.xlabel("Time (Months)")
    plt.ylabel("Number of Mentions")
    plt.title("Topic Trends Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    return topics_df, df, lda

# Example usage:
# topics_df, updated_df, lda_model = lda_topic_modeling(df)
# print(topics_df)
# print(updated_df[["cleaned_text", "Topic", "Topic Meaning"]].head())
