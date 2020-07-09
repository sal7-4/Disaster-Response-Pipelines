import sys
import pandas as pd
import pickle
import re

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''This function loads data to the database DisasterResponse, read the table, and return features 
    '''
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    return X, Y, Y.columns
 

def tokenize(text):
    '''This function tokenizes text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''This model build pipeline for the model, use Grid Search to find the best params
    '''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf',TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {
        'clf__estimator__n_estimators': [10, 20, 40],
    }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs= -1)
    return cv

def evaluate_model(model, X, y, category_names):
    '''This function trains the model and get prediction
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    numofcolumns = len(category_names)

    for i in range(numofcolumns):
        print(y_test.columns[i])
        print(classification_report(y_test.iloc[:, 0], y_pred[:, 0]))

    return model

def save_model(model, model_filepath):
    '''This function save model to model_filepath as pickle file
    '''
        with open(model_filepath, 'wb') as file:
            pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()