# Udacity - Disaster Response Pipeline

## Project Overview
Figure Eight (formerly Crowdflower) crowdsourced the tagging and translation of messages to apply artificial
intelligence to disaster response relief. In this project, a data pipeline was prepared to take in the message data from natural disasters and apply machine learning to categorize emergency text messages based on the need communicated by the sender.

## Installation
The following was used for the environment setup:
- numpy 
- pandas 
- sqlalchemy (create_engine)
- nltk
- nltk.download(['punkt', 'wordnet'])
- nltk.tokenize (word_tokenize)
- sklearn.multioutput (MultiOutputClassifier)
- sklearn.ensemble (RandomForestClassifier)
- sklearn.model_selection (train_test_split)
- re
- nltk.stem (WordNetLemmatizer)
- sklearn.feature_extraction.text (CountVectorizer, TfidfTransformer)
- sklearn.grid_search (GridSearchCV)
- sklearn.pipeline (Pipeline)
- sklearn.metrics (classification_report)
- sklearn.neighbors (KNeighborsClassifier)
- pickle
- matplotlib 

## File Descriptions
The following files are located in this directory:
- messages.csv: Origin dataset of messages (features)
- categories.csv: Origin dataset of categories (labels)
- README.md: This is the file you are reading now!
- ML Pipeline Preparation.ipynb: Jupyter notebook for training and predicting model
- ETL Pipeline Preparation.ipynb: Jupyter notebook used for processing the data
- DisasterResponse.db: Saved database that is created after processing the dataset and training a model based off of that dataset
- app/run.py: Executes and visualizes the dataset
- data/process_data.py: Data cleaning pipeline that Loads the messages and categories datasets, Merges the two datasets, Cleans the data, Stores it in a SQLite database
- model/train_classifier.py: Machine learning pipeline that, Loads data from the SQLite database, Splits the dataset into training and test sets, Builds a text processing and machine learning pipeline, Trains and tunes a model using GridSearchCV, Outputs results on the test set, Exports the final model as a pickle file
