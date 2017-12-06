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
'''

# X = []
# y = []

# X_test = pd.read_csv('data/features.csv', header=None).as_matrix()
X = pd.read_csv('data/features.csv', header=None).as_matrix()

# print('X shape: ', X.shape)
# print('type: ', type(X_test))
# y_test = pd.read_csv('data/labels.csv', header=None).as_matrix()
y = pd.read_csv('data/labels.csv', header=None).as_matrix()
y = y.reshape(len(y[0]), 1)
print(y.shape)

# y_test = y_test.reshape(len(y_test[0]), 1)
# print('y shape: ', y_test.shape)
# print(y_test)
'''
for audio_file in os.listdir(classical_path):
    X.append(gen_MFCC(os.path.join(classical_path, audio_file)).flatten().tolist())
    y.append("classical")


for audio_file in os.listdir(metal_path):
    X.append(gen_MFCC(os.path.join(metal_path, audio_file)).flatten().tolist())
    y.append("metal")


for audio_file in os.listdir(rap_path):
    X.append(gen_MFCC(os.path.join(rap_path, audio_file)).flatten().tolist())
    y.append("rap")
'''
'''
X = np.array(X)

# delete these 3 lines later

print('X shape: ', X.shape)
print('y length: ', len(y))
print('y :', y)
'''

X_train, X_test, y_train, y_test = train_test_split(X, y)

# parameters = {'C': [1, 3, 5, 10, 50, 100]}
parameters = {'kernel': ['poly'], 'C': [0.1, 0.5, 1, 1.5, 3] ,'degree': [2, 3, 4, 5]}
modelSVM = GridSearchCV(svm.SVC(), parameters, scoring='accuracy', cv=5)
modelSVM.fit(X_train, y_train)
print('best params: ', modelSVM.best_params_)

scores = []

for i in range(10):
    score = cross_val_score(modelSVM, X, y, cv=5)
    scores += list(score)

print scores
print np.mean(scores)
print np.std(scores)