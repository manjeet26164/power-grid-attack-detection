import json
import numpy as np
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_DIR = Path("data/preprocessed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

X_train = np.load(DATA_DIR / "X_train.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_train_occur = np.load(DATA_DIR / "y_train_occur.npy")
y_test_occur = np.load(DATA_DIR / "y_test_occur.npy")

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

strategies = ["most_frequent", "stratified", "uniform"]
results = {}

for strategy in strategies:
    clf = DummyClassifier(strategy=strategy, random_state=42)
    clf.fit(X_train_flat, y_train_occur)
    y_pred = clf.predict(X_test_flat)
    results[strategy] = {
        "accuracy": round(accuracy_score(y_test_occur, y_pred), 4),
        "precision": round(precision_score(y_test_occur, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test_occur, y_pred, average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(y_test_occur, y_pred, average="weighted", zero_division=0), 4),
    }

with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))