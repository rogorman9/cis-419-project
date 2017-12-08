from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from generate_MFCC import gen_MFCC
import os
import numpy as np

if __name__ == "__main__":
    genres = ["classical", "metal", "rap", "jazz"]

    estimator_counts = [100, 150, 200, 300, 400, 500, 600]
    learning_rates = [0.1 * i for i in range(1, 12)]
    base_estimators = [DecisionTreeClassifier(max_depth=i) for i in range(1, 6)]

    param_grid = {
        "n_estimators": estimator_counts,
        "learning_rate": learning_rates,
        "base_estimator": base_estimators
    }

    X = np.loadtxt("data/features.csv", delimiter=',')
    y = np.loadtxt("data/labels.csv", delimiter=',', dtype=str)

    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), learning_rate=0.7, n_estimators=600)

    scores = []

    for i in range(2):
        score = cross_val_score(model, X, y, cv=5)
        scores += list(score)

    print np.mean(scores)
    print np.std(scores)
