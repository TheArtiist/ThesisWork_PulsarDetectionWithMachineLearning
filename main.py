import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


column_names = [
    "ip_mean", "ip_std", "ip_kurtosis", "ip_skewness",
    "dm_mean", "dm_std", "dm_kurtosis", "dm_skewness",
    "label"
]
f = open("htru2/HTRU_2.csv" ,"r")
dataset = pd.read_csv(f, header=None, names=column_names)

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True) # null értékek törlése
    df = df.select_dtypes(include=[np.number])  #Számértékek megtartása
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep]

dataFrame = clean_dataset(dataset)

#print(dataFrame.head())
target_column = "label"
class_distribution = dataFrame[target_column].value_counts()

plt.bar(class_distribution.index, class_distribution)
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(class_distribution.index, ['0','1'])
#plt.show()

feature = dataFrame.drop(columns=['label'])
target = dataFrame.label

train_df, temp_df = train_test_split(dataFrame, test_size=0.30, random_state=42, stratify=dataFrame["label"])

val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])

#print("Train méret:", len(train_df))
#print("Validation méret:", len(val_df))
#print("Test méret:", len(test_df))

X_train = train_df.drop(columns=["label"])
x_train = train_df["label"]

X_val = val_df.drop(columns=["label"], axis=1)
x_val = val_df["label"]

Y_test = test_df.drop(columns=["label"], axis=1)
y_test = test_df["label"]


#model = RandomForestClassifier(random_state=42)
#model.fit(X_train, x_train)

#predictions = model.predict(X_val)
#print(classification_report(x_val, predictions))
