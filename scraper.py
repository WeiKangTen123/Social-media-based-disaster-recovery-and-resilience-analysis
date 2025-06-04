import praw
import pandas as pd
import os
import spacy
from datetime import datetime

# 🔑 Reddit API Credentials
REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_client_secret"
REDDIT_USER_AGENT = "your_username"

# Authenticate with Reddit API
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Flood-related keywords
FLOOD_KEYWORDS = ["flood", "heavy rain", "flash flood", "water level rise", "flooding", "storm", "landslide"]

def classify_flood_label(text):
    """Classify post as flood-related (1) or not (0)."""
    return 1 if any(keyword in text.lower() for keyword in FLOOD_KEYWORDS) else 0

def extract_location(text):
    """Extract location using Spacy NLP (Geopolitical Entity)."""
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return locations[0].lower() if locations else None  # Return first detected location (lowercase)

def scrape_flood_posts(locations=["UnitedKingdom"], limit=100, output_file="Datasets/flood_reddit_posts.csv"):
    """
    Scrape flood-related Reddit posts and save them to a CSV file with lowercase column names.
    
    :param locations: List of subreddit locations to scrape.
    :param limit: Number of posts to retrieve per subreddit.
    :param output_file: File path to save the scraped data.
    """
    posts_data = []

    for location in locations:
        try:
            subreddit_instance = reddit.subreddit(location)

            for post in subreddit_instance.search("flood OR heavy rain OR flooding", limit=limit):
                flood_label = classify_flood_label(post.title)
                detected_location = extract_location(post.title + " " + post.selftext)

                final_location = detected_location if detected_location else location.lower()  # Convert to lowercase

                posts_data.append({
                    "text": post.title.lower(),  # Convert to lowercase
                    "timestamp": datetime.utcfromtimestamp(post.created_utc),  # Keep timestamp format
                    "label": flood_label,  # No need for lowercase (already numeric)
                    "location": final_location  # Convert to lowercase
                })
        
        except Exception as e:
            print(f"⚠️ Could not fetch data from r/{location}: {e}")

    posts_df = pd.DataFrame(posts_data)

    if posts_df.empty:
        print("⚠️ No posts found.")
        return

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Check if file exists
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            existing_df = pd.read_csv(output_file)

            if "text" not in existing_df.columns:  # Check for lowercase column name
                print(f"⚠️ Error: '{output_file}' does not have expected columns. Creating a new file.")
                existing_df = pd.DataFrame(columns=["text", "timestamp", "label", "location"])

            # Convert existing data to lowercase for consistency
            existing_df.columns = ["text", "timestamp", "label", "location"]

            # Filter duplicates
            existing_texts = set(existing_df["text"])
            posts_df = posts_df[~posts_df["text"].isin(existing_texts)]

            if not posts_df.empty:
                combined_df = pd.concat([existing_df, posts_df], ignore_index=True)
                combined_df.to_csv(output_file, index=False)
                print(f"✅ Added {len(posts_df)} new posts.")
            else:
                print("⚠️ No new unique posts found.")

        except pd.errors.EmptyDataError:
            print(f"⚠️ Warning: '{output_file}' is empty. Creating a new file.")
            posts_df.to_csv(output_file, index=False)
            print(f"✅ Saved {len(posts_df)} new posts.")

    else:
        print(f"File '{output_file}' is empty or not found. Creating a new file.")
        posts_df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(posts_df)} new posts.")

# Standalone execution for testing
if __name__ == "__main__":
    scrape_flood_posts(["UnitedKingdom"], limit=100, output_file="Datasets/flood_reddit_posts.csv")
