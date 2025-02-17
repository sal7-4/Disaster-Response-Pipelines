import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''This function loads data from messages_filepath and categories_filepath and creates a dataframe df contains both file.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages,categories,how='left',on=['id','id'])
    return df

def clean_data(df):
    '''This function cleans the data df
    '''
    categories = df['categories'].str.split(";", expand=True)
    categories.columns = categories.iloc[0]
    
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2] )
    categories.columns = category_colnames    
    
    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = categories[column].astype(int)

    df=df.drop(['categories'], axis=1)
    df = pd.concat([df,categories],axis=1,join="outer")
    df= df.drop_duplicates()
    return df
    
def save_data(df, database_filename):
    '''This function save data df as a database DisasterResponse
    '''
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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