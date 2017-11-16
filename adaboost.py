from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from generate_MFCC import gen_MFCC
import os
import numpy as np

classical_path = os.path.join("data", "classical")
metal_path = os.path.join("data", "metal")

X = []
y = []

for audio_file in os.listdir(classical_path):
    print audio_file
    X.append(gen_MFCC(os.path.join(classical_path, audio_file)).flatten().tolist())
    y.append("classical")

for audio_file in os.listdir(metal_path):
    print audio_file
    X.append(gen_MFCC(os.path.join(metal_path, audio_file)).flatten().tolist())
    y.append("metal")

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = AdaBoostClassifier()

print cross_val_score(model, X, y, cv=5)



