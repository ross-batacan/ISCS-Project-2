# Intitalizing Virtual Environment
The etl script needs to be run within a virtual environment. 
To initialize a virtual environment, run the following command in your terminal
from within the project directory.

`` python -m venv <your_env_name>``

Project Directory Structure

```
project_directory
.
├── Lda_visualizations/          # File folder
├── .env                         # Environment variable file
├── etl.py                       # Python source file
├── functions.py                 # Python source file
├── main.py                      # Python source file
├── requirements.txt             # Text file for project dependencies
├── stopwords-custom.json        # JSON source file
└── stopwords-tl.json            # JSON source file
```

# Install Dependencies

To install the needed dependencies for the ETL Pipeline,
run the following command in your terminal

``pip install -r requirements.txt``

# .env File

The ETL Script requires that you create a .env file containing 
your CLIENT_ID, CLIENT_SECRET, USER_AGENT, and the SUBREDDIT you 
would like to scrape from. 

https://www.reddit.com/prefs/apps

