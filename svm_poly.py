from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from generate_MFCC import gen_MFCC
import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV

'''
classical_path = os.path.join("data", "classical")
metal_path = os.path.join("data", "metal")
rap_path = os.path.join("data", "rap")

X = []
y = []

for audio_file in os.listdir(classical_path):
    X.append(gen_MFCC(os.path.join(classical_path, audio_file)).flatten().tolist())
    y.append("classical")


for audio_file in os.listdir(metal_path):
    X.append(gen_MFCC(os.path.join(metal_path, audio_file)).flatten().tolist())
    y.append("metal")


for audio_file in os.listdir(rap_path):
    X.append(gen_MFCC(os.path.join(rap_path, audio_file)).flatten().tolist())
    y.append("rap")

X = np.array(X)
print(X.shape)
'''


X = pd.read_csv('data/features.csv', header=None).as_matrix()
# Idea: standardize features
# standardize values of each feature
# mean = np.mean(X, axis=0)
# std = np.std(X, axis=0)
# X = (X - mean) / std
print('X shape: ', X.shape)
y = pd.read_csv('data/labels.csv', header=None).as_matrix()
y = y.reshape(len(y[0]), )


X_train, X_test, y_train, y_test = train_test_split(X, y)
'''
# parameters = {'C': [1, 3, 5, 10, 50, 100]}
parameters = {'kernel': ['poly'], 'C': [0.05, 0.1, 0.3, 0.5, 1, 1.5, 3, 5] ,'degree': [2, 3, 4, 5]}
modelSVM = GridSearchCV(svm.SVC(), parameters, scoring='accuracy', cv=5)
modelSVM.fit(X_train, y_train)
print('best params: ', modelSVM.best_params_)
'''

modelSVMProbs = svm.SVC(kernel='poly', C=0.05, degree=2, probability=True)
modelSVMProbs.fit(X_train, y_train)
print(modelSVMProbs.predict_proba(X_test))

scores = []

for i in range(10):
    score = cross_val_score(modelSVMProbs, X, y, cv=5)
    scores += list(score)

print scores
print np.mean(scores)
print np.std(scores)