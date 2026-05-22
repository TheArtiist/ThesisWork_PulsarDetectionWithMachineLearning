import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import inbalanced

output_dir = "confusion_matrices"
os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_csv("htru2/train.csv")
test_df = pd.read_csv("htru2/test.csv")

target_column = "label"
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]
X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

txt_file = open(os.path.join(output_dir, "confusion_matrices_text.txt"), "w", encoding="utf-8")

def save_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:d}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Noise (0)', 'Pulsar (1)'])
    ax.set_yticklabels(['Noise (0)', 'Pulsar (1)'])
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.title(title, pad=20)
    
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close('all')
    
    txt_file.write(f"=== {title} ===\n")
    txt_file.write(f"TN: {cm[0,0]} | FP: {cm[0,1]}\n")
    txt_file.write(f"FN: {cm[1,0]} | TP: {cm[1,1]}\n\n")
    print(f"Saved: {filename}")

print("Training Random Forest")
model_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
save_cm(y_test, model_rf.predict(X_test), "Random Forest (Nyers adatok)", "cm_rf_original.png")

print("Training SVM")
model_svm = svm.SVC(kernel="linear")
model_svm.fit(X_train, y_train)
save_cm(y_test, model_svm.predict(X_test), "Support Vector Machine (Lineáris)", "cm_svm.png")

print("Evaluating MLP")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(8, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear5(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear6(x)
        return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp = Net().to(device)
if os.path.exists("pulsar_model.pth"):
    mlp.load_state_dict(torch.load("pulsar_model.pth", map_location=device))
    mlp.eval()
    
    test_data_tensor = torch.tensor(X_test[["ip_mean", "ip_std", "ip_kurtosis", "ip_skewness",
                                            "dm_mean", "dm_std", "dm_kurtosis", "dm_skewness"]].values.astype(np.float32)).to(device)
    with torch.no_grad():
        output = mlp(test_data_tensor)
        mlp_preds = output.argmax(dim=1).cpu().numpy()
    
    save_cm(y_test, mlp_preds, "Neurális Hálózat (MLP)", "cm_mlp.png")
else:
    print("pulsar_model.pth not found! Skipping MLP.")

methods = [
    'random_oversampling', 'smote', 'borderline_smote', 'adasyn', 
    'tomek_links', 'cluster_centroids', 'random_undersampling', 
    'nearmiss', 'condensed_nearest_neighbor', 'edited_nearest_neighbor', 
    'one_sided_selection'
]

for method in methods:
    print(f"Training RF with {method}...")
    X_res, y_res = inbalanced.get_resampled_data(method, X_train, y_train)
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf.fit(X_res, y_res)
    save_cm(y_test, clf.predict(X_test), f"Random Forest + {method}", f"cm_rf_{method}.png")

txt_file.close()
print("All confusion matrices generated successfully!")
