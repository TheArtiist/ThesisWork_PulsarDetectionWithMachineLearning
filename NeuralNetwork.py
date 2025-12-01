import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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

train_data_tensor = torch.tensor(X_train[["ip_mean", "ip_std", "ip_kurtosis", "ip_skewness",
    "dm_mean", "dm_std", "dm_kurtosis", "dm_skewness",]].values.astype(np.float32))
train_target_tensor = torch.tensor(x_train.values, dtype=torch.long)

test_data_tensor = torch.tensor(X_val[["ip_mean", "ip_std", "ip_kurtosis", "ip_skewness",
    "dm_mean", "dm_std", "dm_kurtosis", "dm_skewness",]].values.astype(np.float32))
test_target_tensor = torch.tensor(x_val.values, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_target_tensor)
test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_target_tensor)

batch_size = 64

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.5)

        self.linear1 = nn.Linear(8, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 2)

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

        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

epochs = 10
model = Net().to(device)
optimizer = optim.AdamW(model.parameters())
scheduler = StepLR(optimizer, step_size=1)

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

print("\n" + "="*50)
print("FINAL TEST SET EVALUATION:")
print("="*50)

final_test_data_tensor = torch.tensor(Y_test[["ip_mean", "ip_std", "ip_kurtosis", "ip_skewness",
    "dm_mean", "dm_std", "dm_kurtosis", "dm_skewness"]].values.astype(np.float32))
final_test_target_tensor = torch.tensor(y_test.values, dtype=torch.long)

final_test_dataset = torch.utils.data.TensorDataset(final_test_data_tensor, final_test_target_tensor)
final_test_loader = torch.utils.data.DataLoader(final_test_dataset, batch_size=batch_size, shuffle=False)

test(model, device, final_test_loader)

torch.save(model.state_dict(), "pulsar_model.pth")
print("\nModel saved to pulsar_model.pth")
print("Training complete!")
