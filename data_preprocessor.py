import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


def preprocess_flood_data(input_file, output_file):
    """
    Preprocess the flood data from a given CSV file.

    :param input_file: Path to the raw CSV file.
    :param output_file: Path where preprocessed data will be saved.
    """
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file {input_file} does not exist.")
        return

    print(f"🔄 Loading dataset from {input_file}...")
    data = pd.read_csv(input_file)

    # Handle missing values
    data.dropna(inplace=True)

    # Remove duplicates
    data = data.drop_duplicates()

    # Extract mentions (@username) from text
    def extract_mentions(text):
        mentions = re.findall(r'@\w+', str(text))
        return ', '.join(mentions) if mentions else None

    # Extract hashtags (#hashtag) from text
    def extract_hashtags(text):
        hashtags = re.findall(r'#\w+', str(text))
        return ', '.join(hashtags) if hashtags else None

    data["mention"] = data["text"].apply(extract_mentions)
    data["hashtag"] = data["text"].apply(extract_hashtags)

    # Text Cleaning Function
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
        text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text

    data['cleaned_text'] = data['text'].apply(clean_text)

    # Tokenization
    data['tokens'] = data['cleaned_text'].apply(lambda x: x.split())

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    data['tokens'] = data['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    data['tokens'] = data['tokens'].apply(lambda x: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in x])

    # Ensure lowercase column names
    data.rename(columns={
        "text": "Text",
        "timestamp": "Timestamp",
        "label": "Label",
        "location": "Location",
        "mention": "Mention",
        "hashtag": "Hashtag",
        "cleaned_text": "Cleaned_Text",
        "tokens": "Tokens"
    }, inplace=True)



    # Save preprocessed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    write_header = not os.path.exists(output_file)
    data.to_csv(output_file, mode='a', index=False, header=write_header)

    print(f"✅ Preprocessing completed! Data saved to {output_file}")

# Run standalone for testing
if __name__ == "__main__":
    preprocess_flood_data("flood_reddit_posts.csv", "Datasets/preprocessed_flood_data_test.csv")
