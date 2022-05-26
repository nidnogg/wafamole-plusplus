
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix


data_df = pd.read_json('../Dataset/complete_clean.json', orient='records')
print(data_df)

counts = data_df['type'].value_counts()

counts.plot.pie(figsize=(5, 5))

X = data_df['pattern'].to_numpy().astype(str)
y = data_df['type'].to_numpy().astype(str)

print(len(X))
print(len(y))

trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify = y)

np.savez('dataset', trainX=trainX, testX=testX, trainY=trainY, testY=testY)

pipe = make_pipeline(TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024), SVC())


param_grid = {'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 4)], 'svc__C': [1, 10], 'svc__kernel': ['linear', 'rbf']}

grid = GridSearchCV(pipe, param_grid, cv = 2, verbose = 4)

# Perform fitting and score 
grid.fit(trainX, trainY)
grid.score(testX, testY)

preds = grid.predict(testX)

print(classification_report(testY, preds))

plot_confusion_matrix(grid, testX, testY)

print(grid.best_params_)

pipe = make_pipeline(TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024, ngram_range = (1, 2)), SVC(C = 10, kernel = 'rbf'))

pipe.fit(trainX, trainY)
pipe.score(testX, testY)

preds = pipe.predict(testX)

print(classification_report(testY, preds))

plot_confusion_matrix(pipe, testX, testY)

joblib.dump(pipe, 'predictor.dump')

print('printing final output:')
print(joblib.load('predictor.dump'))

