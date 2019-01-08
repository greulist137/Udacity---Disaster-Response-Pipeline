# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:12:52 2019

@author: greul
"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle

# load data from database
def load_data():
    engine = create_engine('sqlite:///DisasterResponse.db')
    df =  pd.read_sql_table('DisasterResponse', engine)
    X = df.message.values
    y = df.iloc[:,5:]
    return X, y

X, y = load_data()


def tokenize(text):

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# train classifier
pipeline.fit(X_train, y_train)


def display_results(cv, y_test, y_pred):
    print(classification_report(y_test, y_pred, target_names=y_test.keys()))
    
    
# predict on test data
y_pred = pipeline.predict(X_test)

# display results
display_results(pipeline, y_test, y_pred)


parameters = {  
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__max_features': ['log2', 'auto', 'sqrt', None],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_depth': [None, 25, 50, 100, 150, 200],
    }


cv = GridSearchCV(estimator=pipeline, param_grid=parameters)


cv.fit(X_train,y_train)

clf = cv.predict(X_test)

# display results
display_results(clf, y_test, y_pred)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))