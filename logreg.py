from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from generate_MFCC import gen_MFCC
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

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

X_train, X_test, y_train, y_test = train_test_split(X, y)

parameters = {'C': [0.5, 1, 5]}
modelLG = GridSearchCV(LogisticRegression(), parameters, scoring='accuracy', cv=5)
modelLG.fit(X_train, y_train)
print('best params: ', modelLG.best_params_)

scores = []

for i in range(10):
    score = cross_val_score(modelLG, X, y, cv=5)
    scores += list(score)

print scores
print np.mean(scores)
print np.std(scores)