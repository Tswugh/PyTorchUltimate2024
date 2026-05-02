#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

# %%
x_list = cars.wt.values
x_np = np.array(x_list, dtype=np.float32).reshape(-1, 1)
y_list = cars.mpg.values.tolist()
x = torch.from_numpy(x_np)
y = torch.tensor(y_list)

# %%
w = torch.rand(1, requires_grad=True, dtype=torch.float32)
b = torch.rand(1, requires_grad=True, dtype=torch.float32)

epochs = 1000
LR = 0.001

for epoch in range(epochs):
    for i in range(len(x)):
        y_pred = x[i]*w + b
        loss = torch.pow(y_pred - y[i], 2)

        loss.backward()

        loss_val = loss.data[0]

        with torch.no_grad():
            w -= w.grad * LR
            b -= b.grad * LR
            w.grad.zero_()
            b.grad.zero_()
    print(loss_val)

# %%
print(f"Weight: {w.item()}, Bias: {b.item()}")

# %%
y_pred = ((x*w)+b).detach().numpy()

# %%
sns.scatterplot(x=x_list, y=y_list)
sns.lineplot(x=x_list, y=y_pred.reshape(-1))

# %%
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_np, y_list)
print(f"Slope: {reg.coef_}, Intercept: {reg.intercept_}")
