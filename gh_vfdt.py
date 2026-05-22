import pandas as pd
import numpy as np
import time
from river import tree
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report

train_df = pd.read_csv("htru2/train.csv")
test_df = pd.read_csv("htru2/test.csv")

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

model_gh_vfdt = tree.HoeffdingTreeClassifier(
    splitter=tree.splitter.GaussianSplitter(),
    grace_period=200 
)

print("\nGH-VFDT (Gaussian-Hoeffding Very Fast Decision Tree) ")
print("Modell betanítása mintánként")

start_time = time.time()

for x, y in zip(X_train.to_dict(orient="records"), y_train):
    model_gh_vfdt.learn_one(x, y)

train_time = time.time() - start_time
print(f"Betanítás befejezve. Idő: {train_time:.4f} másodperc.")

print("Predikciók készítése a teszt halmazon...")
y_pred = []
for x in X_test.to_dict(orient="records"):
    pred = model_gh_vfdt.predict_one(x)
    y_pred.append(pred)

y_pred = [0 if p is None else p for p in y_pred]

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print("\nEredmények:")
print(f" Accuracy: {accuracy:.4f}")
print(f" Balanced Accuracy: {balanced_accuracy:.4f}")

report_gh_vfdt = classification_report(y_test, y_pred, output_dict=True)
print(f" Recall for class 1: {report_gh_vfdt['1']['recall']:.4f}")
print(f" F1-score for class 1: {report_gh_vfdt['1']['f1-score']:.4f}")
