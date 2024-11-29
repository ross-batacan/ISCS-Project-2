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

To create an app and get the credentials, do the following steps:
1. https://www.reddit.com/prefs/apps
2. Enter a name for your app
3. Select 'script'
4. Set redirect uri to: http://localhost:8080
5. Create the application
6. Once created, create the .env file with the details below

```
CLIENT_ID=[personal use script]
CLIENT_SECRET=[the secret key]
USER_AGENT=[name of the application]
SUBREDDIT=Coronavirus_PH [or any subreddit you'd like]
```

# Start ETL Pipeline
To start the ETL Pipeline, run the following command in your terminal

```python main.py```

# Note
Running the etl pipelines gives three outputs
1. ISCS-Reddit-raw.csv - contains the raw data extracted using PRAW stored in a data frame
2. ISCS-Reddit-clean.csv - contains the preprocessed data stored in a data frame with the 'Text' column to be used for the LDA model
3. LDA Visualizations (stored in lda visualizations directory) - contains four html files of IDMs
