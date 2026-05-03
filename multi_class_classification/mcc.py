#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
# %% data import
iris = load_iris()
x = iris.data
y = iris.target

# %% train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %% convert to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# %% dataset
class IrisData(Dataset):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.x = torch.from_numpy(x_train)
        self.y = torch.from_numpy(y_train)
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# %% dataloader
iris_data = IrisData(x_train=x_train, y_train=y_train)
train_loader = DataLoader(dataset=iris_data, batch_size=32, shuffle=True)

# %% check dims
print(f"X shape: {iris_data.x.shape}, Y shape: {iris_data.y.shape}")

# %% define class
class MultiClass(nn.Module):
    def __init__(self, num_feat, num_class, hidden):
        super().__init__()
        self.lin1 = nn.Linear(num_feat, hidden)
        self.lin2 = nn.Linear(hidden, num_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = self.softmax(x)
        return x


# %% hyper parameters
NUM_FEATURES = iris_data.x.shape[1]
HIDDEN = 6
NUM_CLASSES = len(iris_data.y.unique())

# %% create model instance
model = MultiClass(num_class=NUM_CLASSES, num_feat=NUM_FEATURES, hidden=HIDDEN)

# %% loss function
criterion = nn.CrossEntropyLoss()

# %% optimizer
LR = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# %% training
epochs = 100
losses = []
for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(x)

        loss = criterion(y_pred, y)
        loss.backward()

        optimizer.step()
    losses.append(float(loss.data.detach().numpy()))

# %% show losses over epochs
sns.lineplot(x=range(epochs), y=losses)

# %% test the model
x_test_torch = torch.from_numpy(x_test)
with torch.no_grad():
    y_test_log = model(x_test_torch)
    y_test_pred = torch.max(y_test_log.data, 1)

# %% Accuracy
accuracy_score(y_test, y_test_pred.indices)

# %% save model state dict
torch.save(model.state_dict(), 'model_iris.pt')
# %%
