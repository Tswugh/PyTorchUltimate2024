#%% packages
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import numpy as np
from collections import Counter
# %% data prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size = 0.2)


# %% dataset and dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# create instance of dataset
label_dataset = MultilabelDataset(X_train, y_train)
# create train loader
train_loader = DataLoader(label_dataset, batch_size=32, shuffle=True)

# %% model
# set up model class
# topology: fc1, relu, fc2
# final activation function??
class MultilabelClass(nn.Module):
    def __init__(self, in_size, hidden, out_size):
        super().__init__()
        self.lin1 = nn.Linear(in_size, hidden)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden, out_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        return x


# define input and output dim
input_dim = X_torch.shape[1]
output_dim = y_torch.shape[1]

# create a model instance
model = MultilabelClass(in_size=input_dim, hidden=20, out_size=output_dim)
model.train()

# %% loss function, optimizer, training loop
# set up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()

LR = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

losses = []
slope, bias = [], []
number_epochs = 100

#%%
# implement training loop
for epoch in range(number_epochs):
    for j, (x, y) in enumerate(train_loader):

        # optimization
        optimizer.zero_grad()
        # forward pass
        y_pred = model(x)
        # compute loss
        loss = loss_fn(y_pred, y)
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()

    losses.append(float(loss.data.detach().numpy()))

    # print epoch and loss at end of every 10th epoch
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.data}")


# %% losses
# plot losses
sns.scatterplot(x=range(len(losses)), y=losses, alpha=0.1, color='red')

# %% test the model
# predict on test set
X_test_torch = torch.FloatTensor(X_test)
with torch.no_grad():
    y_test_pred = model(X_test_torch).round()

#%% Naive classifier accuracy
# convert y_test tensor [1, 1, 0] to list of strings '[1. 1. 0.]'
y_test_str = [str(i) for i in y_test.detach().numpy()]
y_test_str

# get most common class count
most_common_cnt = Counter(y_test_str).most_common()[0][1]

# print naive classifier accuracy
print(f"Naive classifier: {most_common_cnt/len(y_test_str) * 100}%")

# %% Test accuracy
# get test set accuracy
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy: {test_acc * 100}%")
# %%
