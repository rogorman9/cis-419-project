import os
import sys
import csv
from generate_MFCC import gen_MFCC

genres = ["classical", "metal", "rap"]

X = []
y = []

for g in genres:
    path = os.path.join("data", g)
    for filename in os.listdir(path):
        _, filetype = os.path.splitext(filename)
        if filetype == ".wav":
            X.append(gen_MFCC(os.path.join(path, filename)).flatten().tolist())
            y.append(g)

with open("data/features.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(X)

with open("data/labels.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerow(y)
