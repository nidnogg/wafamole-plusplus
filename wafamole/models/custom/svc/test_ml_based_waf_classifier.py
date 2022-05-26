
import shutup
shutup.please()
import datetime
from sre_parse import Tokenizer
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
shutup.jk()


def trainModel():
    """This function trains a Non-Linear SVM model with the SQL Injection Dataset from Kaggle.
    WARNING: Only works with conda env set to mlWaf (provided in /deps/conda_envs).
    """
    # Input data 
    data_df = pd.read_json('./SQLiV3.json' , orient='records')

    # Count entries for plotting 
    counts = data_df['type'].value_counts()
    counts.plot.pie(figsize=(5, 5))

    # Do string conversion to prepare for pipelines
    X = data_df['pattern'].to_numpy().astype(str)
    y = data_df['type'].to_numpy().astype(str)

    print(len(X))
    print(len(y))

    # Split into training and testing labels
    trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify = y)

    # Vectorizer transformer -> SVC pipeline
    # pipe = make_pipeline(TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024, ngram_range = (1, 2)), SVC(C = 10, kernel = 'rbf', probability=True))
    # For scikit 0.21.1
    pipe = make_pipeline(TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024, ngram_range = (1, 2)), SVC(C = 10, kernel = 'rbf', probability=True, gamma='scale'))


    print("Fitting pipeline")
    print(pipe.fit(trainX, trainY))

    print("Scoring pipeline")
    print(pipe.score(testX, testY))

    preds = pipe.predict(testX)

    # Glitched in scikit 0.21.1
    # print(classification_report(testY, preds))

    # Glitched in scikit 0.21.1
    # plot_confusion_matrix(pipe, testX, testY)

    # Final Steps
    joblib.dump(pipe, 'svc_trained_test{}.dump'.format(datetime.datetime.now()))
    print('printing final output:')
    print(joblib.load('test_threat_classifier.dump'))

def testModel(query):
    """This function tests a Non-Linear SVM model with any SQL query provided as a string.
    It returns the probability of it being an SQL injection, and how the classifier labels such query.
    WARNING: Only works with conda env set to mlWaf (provided in /deps/conda_envs).
    """
    data_df = pd.read_json('./SQLiV3.json' , orient='records')
    X = data_df['pattern'].to_numpy().astype(str)
    y = data_df['type'].to_numpy().astype(str)


    # clf is pipe
    clf = joblib.load('svc_trained.dump')
    print("Testing query for SQL Injection - {}".format(query))
    print("type assumed: {}".format(clf.predict([query])))
    print("probability of being an SQL injection is: {}".format(clf.predict_proba([query])[0,0]))

#trainModel()

testModel("admin' OR 1=1")
testModel("admin'   OR   5o8x4x4o0o1=0o4o2b4b4o0b60#")
