from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from generate_MFCC import gen_MFCC
import os

classical_path = "data/classical"
metal_path = "data/metal"

X = []
y = []

for audio_file in os.listdir(classical_path):
    print gen_MFCC(os.path.join(classical_path, audio_file))
    X.append(gen_MFCC(os.path.join(classical_path, audio_file)).flatten())
    y.append("classical")

for audio_file in os.listdir(metal_path):
    X.append(gen_MFCC(os.path.join(metal_path, audio_file)).flatten())
    y.append("metal")

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = AdaBoostClassifier()
model.fit(X_train, y_train)
print model.score(X_test, y_test)



