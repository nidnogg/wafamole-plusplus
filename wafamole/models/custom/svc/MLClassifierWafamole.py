# %% [markdown]
# # ML-based-WAF Classifier
# 
# This notebook trains a WAF model with the [SQL Injection Dataset from Kaggle](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset), based off vladan stojnic's open source [ML-based-WAF](https://github.com/vladan-stojnic/ML-based-WAF) implementation.
# 
# Tested classifier types: 
# * Non-Linear SVM classifier;
# * Stochastic Gradient Descent
# 
# WARNING: Only works with conda env set to mlWaf (provided in /deps/conda_envs).

# %% [markdown]
# #### Library imports
# 

# %%
import shutup
shutup.please()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# %%
data_df = pd.read_json('~/localdev/db/wafamole_dataset.json' , orient='records')
data_df

# %%
counts = data_df['type'].value_counts()
print(counts)
counts.plot.pie(figsize=(5, 5))

# %% [markdown]
# #### Split into training and testing labels
# 

# %%
X = pd.arrays.StringArray(data_df['pattern']).to_numpy()
y = pd.arrays.StringArray(data_df['type']).to_numpy()
trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify = y)

# %% [markdown]
# #### Prepare for GridSearchCV classification for best params based on Classifier type
# Note - Linear with ngram range [1,4] seems to peform slightly worse than rbf kernel with ngram_range [1,2].

# %%
# pipe = make_pipeline(
#     TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024), 
#     SVC())
# param_grid = {'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 4)], 
#               'svc__C': [1, 10], 'svc__kernel': ['linear', 'rbf']}  
# grid = GridSearchCV(pipe, param_grid, cv = 2, verbose = 4, n_jobs=4)
# grid.fit(trainX, trainY)
# grid.best_params_

# %% [markdown]
# #### 1. Train model with best params
# Found by experimenting with top results from GridSearchCV run

# %%
pipe = make_pipeline(
    TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024, ngram_range = (1, 2)), 
    SVC(C = 10, kernel = 'rbf', probability=True, gamma='scale'))
pipe.fit(trainX, trainY)
pipe.score(testX, testY)

# %% [markdown]
# #### 2. Train model with suggested GridSearch optimal params
# **WARNING** do not run this cell if the previous cell has been executed!
# 
# Uses best result from GridSearchCV run - however the performance is slightly worse.

# %%
pipe = make_pipeline(
    TfidfVectorizer(input = 'content', lowercase = True, analyzer = 'char', max_features = 1024, ngram_range = (1, 4)), 
    SVC(C = 10, kernel = 'linear', probability=True, gamma='scale'))
pipe.fit(trainX, trainY)
pipe.score(testX, testY)

# %%
predY = pipe.predict(testX)
print(classification_report(testY, predY))
print(confusion_matrix(testY, predY))

# %%
joblib.dump(pipe, 'test_svc_classifier.dump')

# %% [markdown]
# #### WAF-A-MoLE query testing
# For testing with and without WAF queries in dataset.

# %%
# clf = joblib.load('test_svc_classifier_extra_moled.dump')
# # query = "admin'   OR   5o8x4x4o0o1=0o4o2b4b4o0b60#"
# query = "aDMiN'!!, OR  ;](sELeCt 9)=6O8 or\{6, OR  fAlsE}ANd true Or[FALSE?anD TRUe And trUe+and TrUe Or FalSe  OR  False#7xFKSipgJo;"
# print("Testing query {} for SQL Injection".format(query))
# print("type assumed: {}".format(clf.predict([query])))
# print("probability of being SQL injection is: {}".format(clf.predict_proba([query])[0,0]))

# # %%
# clf = joblib.load('test_svc_classifier_moled.dump')
# # query = "admin'   OR   5o8x4x4o0o1=0o4o2b4b4o0b60#"
# query = "aDMiN'!!, OR  ;](sELeCt 9)=6O8 or\{6, OR  fAlsE}ANd true Or[FALSE?anD TRUe And trUe+and TrUe Or FalSe  OR  False#7xFKSipgJo;"
# print("Testing query {} for SQL Injection".format(query))
# print("type assumed: {}".format(clf.predict([query])))
# print("probability of being SQL injection is: {}".format(clf.predict_proba([query])[0,0]))


