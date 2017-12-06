from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from alex_generate_MFCC import gen_MFCC
from alex_tune_parameters import tuned_params
import os
import numpy as np

classical_path = os.path.join("data", "classical")
metal_path = os.path.join("data", "metal")

X = []
y = []

for audio_file in os.listdir(classical_path):
	test = gen_MFCC(os.path.join(classical_path, audio_file))
	print test, test.shape
	X.append(gen_MFCC(os.path.join(classical_path, audio_file)).flatten().tolist())
	y.append("classical")

for audio_file in os.listdir(metal_path):
	X.append(gen_MFCC(os.path.join(metal_path, audio_file)).flatten().tolist())
	y.append("metal")

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = AdaBoostClassifier()

scores = []

for i in range(10):
	score = cross_val_score(model, X, y, cv=5)
	scores += list(score)

print scores
print np.mean(scores)
print np.std(scores)
