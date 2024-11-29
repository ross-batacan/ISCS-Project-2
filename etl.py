import os
import logging
import pandas as pd
import json
import traceback
from functions import *
from dotenv import load_dotenv

# Configure logging for the ETL process
logging.basicConfig(
    filename='etl_process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ETLProcess:
    def __init__(self):
        self.reddit_data = None
        self.cleaned_data = None
        self.lda_models = None

    def load_env(self):
        """Load the configuration from the .env file."""
        try:
            logging.info("Loading configuration from .env...")
            load_dotenv()

            # Check that the required environment variables are set
            required_vars = ["CLIENT_ID", "CLIENT_SECRET", "USER_AGENT", "SUBREDDIT"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise KeyError(f"Missing required environment variables: {', '.join(missing_vars)}")

            logging.info("Configuration loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise

    def extract_data(self):
        """Extract data by scraping from the subreddit."""
        try:
            logging.info("Initializing Reddit client...")
            reddit = initialize_reddit()

            # Get subreddit name from the environment variables
            subreddit_name = os.getenv("SUBREDDIT")
            if not subreddit_name:
                raise ValueError("SUBREDDIT environment variable is not set.")

            subreddit = get_subreddit(reddit, subreddit_name)

            logging.info(f"Starting to scrape data from {subreddit_name}...")
            scrape_from_subreddit(subreddit, limit=500, csv_file=f"{subreddit_name}-raw.csv")

            # Load the scraped data from CSV
            logging.info(f"Loading data from {subreddit_name}-raw.csv...")
            self.reddit_data = pd.read_csv(f"{subreddit_name}-raw.csv")
            logging.info("Data extracted successfully.")
        except Exception as e:
            logging.error(f"Error during data extraction: {e}")
            raise

    def transform_data(self):
        """Transform the extracted data (e.g., preprocess, train LDA)."""
        try:

            logging.info("Loading stopwords...")
            stopwords_list = load_stopwords()

            logging.info("Preprocessing data...")
            self.cleaned_data = preprocess_data(self.reddit_data, stopwords_list)

            logging.info("Data transformation completed.")
        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise

    def load_data(self):
        """Load transformed data (e.g., visualize LDA topics)."""
        try:
            logging.info("Training LDA models...")
            self.lda_models = train_lda_model(self.cleaned_data, num_topics_range=range(2, 6))
        except Exception as e:
            logging.error(f"Error during data loading: {e}")
            raise

    def run_etl(self):
        """Wrapper function that wraps the entire ETL process."""
        try:
            self.load_env()       # Load the config
            self.extract_data()      # Extract data
            self.transform_data()    # Transform data
            self.load_data()         # Load data (or visualize)
            logging.info("ETL process completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during the ETL process: {e}")
            logging.error(f"Error details: {traceback.format_exc()}")
            print(f"An error occurred: {e}")
            print(f"Error details: {traceback.format_exc()}")

if __name__ == "__main__":
    etl_process = ETLProcess()
    etl_process.run_etl()
