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

# load messages dataset
messages = pd.read_csv("messages.csv")
print(messages.head())

# load categories dataset
categories = pd.read_csv("categories.csv")
print(categories.head())

# merge datasets
df = pd.merge(messages, categories)
df_temp_id = df['id']
print(df.head())

# create a dataframe of the 36 individual category columns
categories =  df['categories'].str.split(';', expand=True).add_prefix('categories_')
print(categories.head())

# select the first row of the categories dataframe
row = categories.iloc[0]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing

category_colnames = list()
for x in row:
    print(x[0:-2])
    category_colnames.append(x[0:-2])
print(category_colnames)

# rename the columns of `categories`
categories.columns = category_colnames
print(categories.head())

for column in categories:
    # set each value to be the last character of the string
    categories[column] =  categories[column].str[-1]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
print(categories.head())

# drop the original categories column from `df`
df.drop(['categories'], axis=1, inplace = True)
print(df.head())

# concatenate the original dataframe with the new `categories` dataframe
categories['id'] = df_temp_id
df = pd.merge(messages, categories)
df.head()

# check number of duplicates
print(df.duplicated().sum())

# drop duplicates
df.drop_duplicates(inplace = True)

# check number of duplicates
print(df.duplicated().sum())

'''MAKE SURE TO UNCOMMENT THIS LATER'''
#engine = create_engine('sqlite:///DisasterResponse.db')
#df.to_sql('DisasterResponse', engine, index=False)

# load data from database
def load_data():
    engine = create_engine('sqlite:///DisasterResponse.db')
    df =  pd.read_sql_table('DisasterResponse', engine)
    X = df.message.values
    y = df.iloc[:,5:]
    return X, y

X, y = load_data()

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

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
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__max_features': ['log2', 'auto'],
        'clf__estimator__n_estimators': [100, 250],
}

cv = GridSearchCV(estimator=pipeline, param_grid=parameters)


cv.fit(X_train,y_train)    


clf = cv.predict(X_test, y_test)

# display results
display_results(clf, y_test, y_pred)