from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from generate_MFCC import gen_MFCC
import os
import numpy as np
from extract_features_from_wav import extract_features_from_wav

"""

Create and test a DecisionTreeClassifier on a feature set which combines mfcc features 
with features extracted from the amplitude wave values (such as amplitude mean, std, etc)

"""


# Generate features
genres = ["classical", "metal", "rap"]

X = []
y = []

# Load in the mfcc data
for g in genres:
    path = os.path.join("data", g)
    for filename in os.listdir(path):
    	# print path, filename
        _, filetype = os.path.splitext(filename)
        if filetype == ".wav":
            X.append(gen_MFCC(os.path.join(path, filename)).flatten().tolist())
            y.append(g)

X = np.array(X)
y = np.array(y)

model = DecisionTreeClassifier()
scores = []
for _ in xrange(10):
	score = cross_val_score(model, X, y, cv=5)
	scores += list(score)
accuracy = np.mean(scores)
print "Decision Tree without amplitude data: " + str(accuracy)

# Now append the amplitude data to the features array for each song (appending 4 features to each row)
n,d = X.shape
for _ in xrange(4):
	X = np.c_[X, np.zeros((n,1))]

cntr = 0
for g in genres:
	path = os.path.join("data", g)
	for filename in os.listdir(path):
		# print path, filename
		_, filetype = os.path.splitext(filename)
		if filetype == ".wav":
			amplitude_data = extract_features_from_wav(os.path.join(path, filename))
			X[cntr, d:] = amplitude_data
			cntr += 1

model = DecisionTreeClassifier()
scores = []
for _ in xrange(10):
	score = cross_val_score(model, X, y, cv=5)
	scores += list(score)
accuracy = np.mean(scores)
print "Decision Tree with amplitude data: " + str(accuracy)

# Check how much importance the Decision Tree Classifier is placing on the four amplitude features
model = DecisionTreeClassifier()
model.fit(X, y)
importance = model.feature_importances_

for indx, val in enumerate(importance):
	if val > 0:
		print indx, val












