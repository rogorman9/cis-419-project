from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from generate_MFCC import gen_MFCC
import os
import numpy as np

genres = ["classical", "metal", "rap"]

X = []
y = []

for genre in genres:
    path = os.path.join("data", genre)
    for audio_file in os.listdir(path):
        print audio_file
        X.append(gen_MFCC(os.path.join(path, audio_file)).flatten().tolist())
        y.append(genre)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = AdaBoostClassifier()

scores = []

for i in range(10):
    score = cross_val_score(model, X, y, cv=5)
    scores += list(score)

print np.mean(scores)
print np.std(scores)



