import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report


train_df = pd.read_csv("htru2/train.csv")
val_df = pd.read_csv("htru2/validation.csv")
test_df = pd.read_csv("htru2/test.csv")

target_column = "label"
class_distribution = train_df[target_column].value_counts()

plt.bar(class_distribution.index, class_distribution)
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(class_distribution.index, ['0','1'])
#plt.show()

#print("Train méret:", len(train_df))
#print("Validation méret:", len(val_df))
#print("Test méret:", len(test_df))

X_train = train_df.drop(columns=["label"])
x_train = train_df["label"]

X_val = val_df.drop(columns=["label"], axis=1)
x_val = val_df["label"]

Y_test = test_df.drop(columns=["label"], axis=1)
y_test = test_df["label"]


def balanced_accuracy(y_true, y_pred):
    """Compute balance accuracy using numpy"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    
    recalls = []
    for cls in classes:
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        actual_positives = np.sum(y_true == cls)
        recall = true_positives / actual_positives
        recalls.append(recall)

    return np.mean(recalls)

#model = RandomForestClassifier(random_state=42)
#model.fit(X_train, x_train)

model = svm.SVC(kernel="linear")
model.fit(X_train, x_train)

predictions = model.predict(X_val)
print(classification_report(x_val, predictions))

b_acc = balanced_accuracy(x_val, predictions)
print(f"Balanced Accuracy: {b_acc:.4f}")
