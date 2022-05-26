
from sre_parse import Tokenizer
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
from wafamole.tokenizer.tokenizer import Tokenizer



data_df = pd.read_json('../Dataset/SQLi/SQLiV3.json' , orient='records')

counts = data_df['type'].value_counts()

counts.plot.pie(figsize=(5, 5))

X = data_df['pattern'].to_numpy().astype(str)
y = data_df['type'].to_numpy().astype(str)

print(len(X))
print(len(y))

trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify = y)

#clf = joblib.load('predictor.dump')
# From WAF-A-MoLE
tokenizer = Tokenizer() 
clf = SVC(C = 10, kernel = 'rbf', probability=True, Tokenizer=tokenizer)
#import pdb; pdb.set_trace()

clf.fit(trainX, trainY)
print(clf.score(testX, testY))

preds = clf.predict(testX)

print(classification_report(testY, preds))

plot_confusion_matrix(clf, testX, testY)

joblib.dump(clf, 'test_threat_classifier.dump')

print('printing final output:')
print(joblib.load('test_threat_classifier.dump'))

