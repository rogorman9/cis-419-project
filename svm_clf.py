import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


data = np.loadtxt('metal_classical_data.dat', dtype='string', delimiter=",")

X = data[:, :-1]
X = X.astype(float)
y = data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13, shuffle=True)

clf = SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print accuracy




