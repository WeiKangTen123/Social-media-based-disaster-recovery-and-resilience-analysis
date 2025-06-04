from collections import Counter
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

def extract_top_hashtags_mentions(df, hashtag_column='Hashtag', mention_column='Mention', top_n=10):
    """
    Extracts and counts the most used hashtags and most mentioned users from a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing social media data.
        hashtag_column (str): The column name containing hashtags.
        mention_column (str): The column name containing mentions.
        top_n (int): Number of top hashtags and mentions to return.
    
    Returns:
        tuple: Two DataFrames (hashtags, mentions) with the top used hashtags and mentioned users.
    """
    # Extract and count most used hashtags
    hashtag_data = df.dropna(subset=[hashtag_column])[hashtag_column]
    hashtags = [tag for sublist in hashtag_data.str.split() for tag in sublist]  # Flatten list
    hashtag_counts = Counter(hashtags).most_common(top_n)  # Get top N hashtags

    # Extract and count most mentioned users
    mention_data = df.dropna(subset=[mention_column])[mention_column]
    mentions = [mention for sublist in mention_data.str.split() for mention in sublist]  # Flatten list
    mention_counts = Counter(mentions).most_common(top_n)  # Get top N mentions

    # Convert to DataFrame for display
    hashtag_df = pd.DataFrame(hashtag_counts, columns=['Hashtag', 'Count'])
    mention_df = pd.DataFrame(mention_counts, columns=['Mention', 'Count'])

    return hashtag_df, mention_df

def build_network_graphs(df, hashtag_column='Hashtag', mention_column='Mention', text_column='Text', top_n=5):
    """
    Builds mention and hashtag co-occurrence networks from a DataFrame and displays in Streamlit.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing social media data.
        hashtag_column (str): The column name containing hashtags.
        mention_column (str): The column name containing mentions.
        text_column (str): The column containing tweet/text data.
        top_n (int): Number of top hashtags and mentions to consider in the networks.
    
    Returns:
        None (Displays the generated network graphs in Streamlit).
    """
    df[hashtag_column] = df[hashtag_column].fillna("")
    df[mention_column] = df[mention_column].fillna("")

    hashtags = [tag for sublist in df[hashtag_column].dropna().str.split() for tag in sublist]
    mentions = [mention for sublist in df[mention_column].dropna().str.split() for mention in sublist]
    
    hashtag_counts = Counter(hashtags).most_common(top_n)
    mention_counts = Counter(mentions).most_common(top_n)
    
    top_hashtags = [hashtag for hashtag, _ in hashtag_counts]
    top_mentions = [mention for mention, _ in mention_counts]
    
    # Build mention network
    mention_graph_filtered = nx.DiGraph()
    mention_edges_filtered = [(row[text_column][:30], mention) for _, row in df.iterrows()
                              for mention in row[mention_column].split() if mention in top_mentions]
    mention_graph_filtered.add_edges_from(mention_edges_filtered)
    
    # Apply layout and draw mention network
    mention_pos_filtered = nx.spring_layout(mention_graph_filtered, k=1.7)
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw(mention_graph_filtered, mention_pos_filtered, with_labels=True,
            node_size=300, font_size=9, edge_color="gray", alpha=0.6, node_color="skyblue", ax=ax)
    plt.title("Optimized Mention Network (Top 5 Users Only)")
    st.pyplot(fig)
    
    # Build hashtag co-occurrence network
    hashtag_graph_filtered = nx.Graph()
    hashtag_edges_filtered = [(h1, h2) for _, row in df.iterrows()
                              for h1 in row[hashtag_column].split() for h2 in row[hashtag_column].split()
                              if h1 != h2 and h1 in top_hashtags and h2 in top_hashtags]
    hashtag_graph_filtered.add_edges_from(hashtag_edges_filtered)
    
    # Apply layout and draw hashtag network
    hashtag_pos_filtered = nx.spring_layout(hashtag_graph_filtered, k=0.8)
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw(hashtag_graph_filtered, hashtag_pos_filtered, with_labels=True,
            node_size=300, font_size=9, edge_color="gray", alpha=0.6, node_color="lightcoral", ax=ax)
    plt.title("Hashtag Co-occurrence Network (Top 5 Hashtags Only)")
    st.pyplot(fig)

# Example usage in Streamlit:
# if st.button("Run Network Analysis"):
#     build_network_graphs(df)
