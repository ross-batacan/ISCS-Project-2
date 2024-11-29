from dotenv import load_dotenv

import traceback
import os
import time
import json
import praw
import logging
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


# Configure logging
logging.basicConfig(
    filename='etl_process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


def load_stopwords():
    """Load all stopwords from NLTK, custom, and Tagalog files."""
    try:
        # Start with NLTK English stopwords
        all_stopwords = set(stopwords.words('english'))
        
        # Load Tagalog stopwords
        with open('stopwords-tl.json', 'r') as file:
            tagalog_stopwords = set(json.load(file))
        
        # Load custom stopwords
        with open('stopwords-custom.json', 'r') as file:
            custom_stopwords = set(json.load(file))
        
        # Combine all stopwords
        all_stopwords.update(tagalog_stopwords)
        all_stopwords.update(custom_stopwords)
        
        logging.info("Stopwords loaded successfully.")
        return list(all_stopwords)
    except FileNotFoundError as e:
        logging.error(f"Stopwords file not found: {e}")
        raise FileNotFoundError("One or more stopwords files were not found.")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON stopwords file: {e}")
        raise ValueError("Error decoding JSON stopwords file.")

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def initialize_reddit():
    """Initializes Reddit client using credentials from .env."""
    try:
        # Load variables from .env
        load_dotenv()

        # Access the necessary environment variables
        client_id = os.getenv("CLIENT_ID")
        client_secret = os.getenv("CLIENT_SECRET")
        user_agent = os.getenv("USER_AGENT")

        # Validate that all required variables are present
        if not all([client_id, client_secret, user_agent]):
            raise KeyError("Missing one or more required environment variables.")

        # Initialize the Reddit client
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

        logging.info("Reddit client initialized successfully.")
        return reddit
    except KeyError as e:
        logging.error(f"KeyError: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during Reddit initialization: {e}")
        raise

def get_subreddit(reddit, subreddit_name):
    """Fetches the subreddit object."""
    try:
        subreddit = reddit.subreddit(subreddit_name)
        logging.info(f"Subreddit '{subreddit_name}' initialized successfully.")
        return subreddit
    except Exception as e:
        logging.error(f"An error occurred when initializing the subreddit: {e}")
        raise e

def scrape_from_subreddit(subreddit, limit, csv_file):
    """
    Scrapes posts and comments from a subreddit and saves them to a CSV file.

    Parameters:
        subreddit (praw.models.Subreddit): The subreddit object to scrape from.
        limit (int): Number of posts to scrape.
        csv_file (str): The filename for the CSV to store data.
    """
    all_data = []  # List to store all post and comment data

    try:
        # Scraping posts and comments
        for post in subreddit.new(limit=limit):  # Iterate through the newest posts
            print(f"Working on Post ID: {post.id}")

            # Append the post data
            all_data.append({
                'Type': 'Post',
                'Post_id': post.id,
                'Title': post.title,
                'Author': post.author.name if post.author else 'Unknown',
                'Timestamp': pd.to_datetime(post.created_utc, unit='s'),
                'Text': post.selftext,
                'Score': post.score,
                'Total_comments': post.num_comments,
                'Post_URL': post.url
            })

            # Check if the post has comments
            if post.num_comments > 0:
                # Scraping comments for each post
                post.comments.replace_more(limit=0)  # Fetch all comments
                for comment in post.comments.list():
                    all_data.append({
                        'Type': 'Comment',
                        'Post_id': post.id,
                        'Title': post.title,
                        'Author': comment.author.name if comment.author else 'Unknown',
                        'Timestamp': pd.to_datetime(comment.created_utc, unit='s'),
                        'Text': comment.body,
                        'Score': comment.score,
                        'Total_comments': 0,  # Comments don't have this attribute
                        'Post_URL': None  # Comments don't have this attribute
                    })

            time.sleep(2)  # Pause for 2 seconds between requests

        # Create a pandas DataFrame for all posts and comments
        reddit_data = pd.DataFrame(all_data)

        # Save the data to a CSV file
        reddit_data.to_csv(csv_file, index=False)
        print(f"Total posts and comments scraped: {len(all_data)}")
        logging.info(f"Data saved to {csv_file} successfully with {len(all_data)} records.")

    except Exception as e:
        logging.error(f"An error occurred during scraping: {e}")
        traceback.print_exc()
        raise e

def clean_stopwords(stopwords_list):
    """
    Cleans the stopwords list by removing special characters and empty strings. This is done
    because text scraped from the subreddit we're cleaned.
    """
    cleaned_stopwords = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in stopwords_list]
    # Remove any empty strings that may result from the cleaning
    return [word for word in cleaned_stopwords if word]

def clean_text(df, text_column, stopwords):
    """
    Cleans the text in the specified column of the DataFrame by performing several steps:
    - Replaces line breaks with spaces
    - Removes special characters
    - Converts text to lowercase
    - Removes stopwords
    - Lemmatizes words
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the text data.
        text_column (str): The name of the column to clean.
        stopwords (list): A list of stopwords to remove from the text.
    
    Returns:
        pd.DataFrame: The DataFrame with the cleaned text.
    """
    # Ensure that all values in the text column are strings
    df[text_column] = df[text_column].apply(lambda x: str(x) if isinstance(x, str) else '')

    # Replace line breaks with spaces
    df[text_column] = df[text_column].apply(lambda row: re.sub(r'[\n\r]+', ' ', row))

    # Remove special characters
    df[text_column] = df[text_column].apply(lambda row: re.sub(r'[^a-zA-Z0-9\s]', '', row))

    # Convert to lowercase
    df[text_column] = df[text_column].str.lower()

    # Remove stopwords
    df[text_column] = df[text_column].apply(
        lambda row: ' '.join([word for word in row.split() if word not in stopwords])
    )

    # Lemmatize words
    df[text_column] = df[text_column].apply(
        lambda row: ' '.join([lemmatizer.lemmatize(word) for word in row.split()])
    )

    # Remove leading and trailing whitespace
    df[text_column] = df[text_column].str.strip()

    return df


# def preprocess_data(reddit_data, year, stopwords_list):
def preprocess_data(reddit_data, stopwords_list):
    """
    Preprocess the scraped Reddit data to filter posts and comments by the provided year.
    Also cleans the text in the 'Text' column using the provided stopwords list.
    
    Parameters:
        reddit_data (pd.DataFrame): The DataFrame containing the scraped data.
        year (int): The year to filter the posts and comments by (e.g., 2022).
        stopwords_list (list): A list of stopwords to remove from the text.
    
    Returns:
        pd.DataFrame: A cleaned DataFrame with posts and comments for the given year.
    """
    try:
        # Step 1: Drop the column 'Author' 
        reddit_data_cleaned = reddit_data.drop(columns=['Author'])

        # Step 2: Convert 'Timestamp' to datetime format
        reddit_data_cleaned['Timestamp'] = pd.to_datetime(reddit_data_cleaned['Timestamp'])

        # Step 4: Clean the text by removing stopwords and lemmatizing
        cleaned_data = clean_text(reddit_data_cleaned, 'Text', stopwords_list)

        # logging.info(f"Data successfully filtered for the year {year}.")
        logging.info("Data successfully filtered.")
        print(f"Filtered data contains {len(cleaned_data)} records.")
        
        load_dotenv()
        subreddit_name = os.getenv("SUBREDDIT")
        csv_file=(f"{subreddit_name}-clean.csv")

        print(f"Saved to {csv_file}.csv")
        cleaned_data.to_csv(f"{subreddit_name}-clean.csv", index=False)
        
        return cleaned_data
    
    except Exception as e:
        logging.error(f"An error occurred during data preprocessing: {e}")
        raise e
    
def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))

def train_lda_model(dataframe, num_topics_range=range(2, 6), output_dir="lda_visualizations"):
    # Convert the 'Text' column to a list of documents
    x = dataframe['Text'].tolist()
    docs = list(sent_to_words(x))   

    # Check the structure of docs
    print(f"Sample docs: {docs[:3]}")  # Print a few sample documents for debugging

    # Create bigrams and trigrams
    bigram = gensim.models.Phrases(docs, min_count=5)
    trigram = gensim.models.Phrases(bigram[docs])

    # Add bigrams and trigrams to docs
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                docs[idx].append(token)
        for token in trigram[docs[idx]]:
            if '_' in token:
                docs[idx].append(token)

    # Create dictionary and corpus
    gsim_dict = Dictionary(docs)
    gsim_dict.filter_extremes(no_below=5, no_above=0.80)

    # Check dictionary before corpus creation
    print(f"Sample dictionary: {list(gsim_dict.items())[:10]}")  # Print a few sample dictionary items

    corpus = [gsim_dict.doc2bow(doc) for doc in docs]


    # Train LDA models for each number of topics in the range
    lda_models = {}
    for num_topics in num_topics_range:
        print(f"Training LDA model with {num_topics} topics...")
        gsim_lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=gsim_dict, passes=50, iterations=100)
        lda_models[num_topics] = gsim_lda

        # Visualize and save the model output
        vis = gensimvis.prepare(gsim_lda, corpus, gsim_dict)
        output_path = os.path.join(output_dir, f"lda_vis_{num_topics}_topics.html")
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save visualization to HTML file
        pyLDAvis.save_html(vis, output_path)
        print(f"Visualization saved for {num_topics} topics at {output_path}")

    return lda_models
