
import shutup
shutup.please()
from sre_parse import Tokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

def trainModel(clfType):
    """This function trains Model of clfType with the SQL Injection Dataset from Kaggle.
    Possible clfTypes: 
    * Non-Linear SVM classifier;
    * SGD
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

    # Split into training and testing labels
    trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify = y)

    # Prepare for GridSearchCV classification for best params based on Classifier type
    if(clfType == "SVC"):
        pipe = make_pipeline(TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024), SVC())
        param_grid = {'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 4)], 'svc__C': [1, 10], 'svc__kernel': ['linear', 'rbf']}   
    if(clfType == "SGD"):
        pipe = make_pipeline(TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024), SVC())
        param_grid = {'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 4)], 'svc__C': [1, 10], 'svc__kernel': ['linear', 'rbf']}

    # Perform GridSearch fitting and fetch best params (commented due to large processing time)
    # grid = GridSearchCV(pipe, param_grid, cv = 2, verbose = 4)
    # grid.fit(trainX, trainY)
    # print(grid.best_params_)
    # For SVC, {'svc__C': 10, 'svc__kernel': 'linear', 'tfidfvectorizer__ngram_range': (1, 4)} - Note - Seems to peform worse than nonlinear kernel.


    # Vectorizer transformer -> Classifier pipeline
    # pipe = make_pipeline(TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024, ngram_range = (1, 2)), SVC(C = 10, kernel = 'rbf', probability=True))
    # For scikit 0.21.1
    if(clfType == "SVC"):
        # Support Vector Classification
        #pipe = make_pipeline(TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024, ngram_range = (1, 2)), SVC(C = 10, kernel = 'rbf', probability=True, gamma='scale'))
        pipe = make_pipeline(TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024, ngram_range = (1, 4)), SVC(C = 10, kernel = 'linear', probability=True, gamma='scale'))

    if(clfType == "SGD"):
        # Stochastic Gradient Descent Classifier
        # Modified Huber Loss required for predict_proba
        pipe = make_pipeline(TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024, ngram_range = (1, 2)), SGDClassifier(loss="modified_huber", penalty="l2", max_iter=500))

    print("Fitting pipe")
    print(pipe.fit(trainX, trainY))

    print("Scoring pipe")
    print(pipe.score(testX, testY))

    preds = pipe.predict(testX)

    # Glitched in scikit 0.21.1
    # print(classification_report(testY, preds))

    # Glitched in scikit 0.21.1
    # plot_confusion_matrix(pipe, testX, testY)

    # Final Steps

    print('printing final output:')
    if(clfType == "SVC"):
        # Support Vector Classification
        joblib.dump(pipe, 'test_svc_classifier.dump')
        print(joblib.load('test_svc_classifier.dump'))    

    if(clfType == "SGD"):
        # Stochastic Gradient Descent Classifier
        joblib.dump(pipe, 'test_sgd_classifier.dump')
        print(joblib.load('test_sgd_classifier.dump'))

def testModel(query, type):
    """This function tests a model of type type against a provided SQL Query.
    Possible model types: 
    * Non-Linear SVM classifier;
    * SGD
    WARNING: Only works with conda env set to mlWaf (provided in /deps/conda_envs).
    """
    data_df = pd.read_json('./SQLiV3.json' , orient='records')
    X = data_df['pattern'].to_numpy().astype(str)
    y = data_df['type'].to_numpy().astype(str)


    # clf is pipe
    if(type == "SVC"):
        clf = joblib.load('test_svc_classifier.dump')
    if(type == "SGD"):
        clf = joblib.load('test_sgd_classifier.dump')
    print("Testing query {} for SQL Injection".format(query))
    print("type assumed: {}".format(clf.predict([query])))
    print("probability of being SQL injection is: {}".format(clf.predict_proba([query])[0,0]))

# trainModel("SVC")

testModel("admin' OR 1=1", "SVC")

# Vulnerable query for SVC generated with WAF-A-MoLE
testModel("admin'   OR   5o8x4x4o0o1=0o4o2b4b4o0b60#", "SVC")

# Vulnerable query for Linear SVC generated with WAF-A-MoLE
testModel("admin'          ||          9o0b101o1x0o0x0o0o7b0o1561x0b0o72=9b0b1101101o0b10b6#", "SVC")


# Vulnerable query for SGD generated with WAF-A-MoLE
#testModel("admin'  OR  0x5x6b0o0b0o5x0x7x8x0x0o2b0o0x0o0o6b0o0b0x6b0o0o0o0b0o0b0o4x0b0=6b171#", "SGD")


