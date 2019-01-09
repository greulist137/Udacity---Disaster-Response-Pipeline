import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT 
        database_filepath - Filepath used for importing the database     
    OUTPUT
        Returns the following variables:
        X - Returns the input features.  Specifically, this is returning the messages column from the dataset
        Y - Returns the categories of the dataset.  This will be used for classification based off of the input X
        y.keys - Just returning the columns of the Y columns
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)
    df_temp_id = df['id']
    return df, df_temp_id

def clean_data(df, df_temp_id):
    '''
    INPUT 
        df: Dataframe to be cleaned by the method
        df_temp_id: the id that is to be used when merging the messages and classifications together based off of the common id
    OUTPUT
        df: Returns a cleaned dataframe Returns the following variables:
    '''
    categories =  df['categories'].str.split(';', expand=True).add_prefix('categories_')
    messages = df[['message', 'genre', 'id']]
    row = categories.iloc[0]
    category_colnames = list()
    for x in row:
        #print(x[0:-2])
        category_colnames.append(x[0:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    categories['id'] = df['id']

    df = pd.merge(messages, categories)
    # check number of duplicates
    print(df.duplicated().sum())
    # drop duplicates
    df.drop_duplicates(inplace = True)
    # check number of duplicates
    print(df.duplicated().sum())
    return df
    
def save_data(df, database_filename):
    '''
    INPUT 
        df: Dataframe to be saved
        database_filepath - Filepath used for saving the database     
    OUTPUT
        Saves the database
    '''
    engine = create_engine('sqlite:///data//DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, df_temp_index = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, df_temp_index)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()