import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter

df = pd.read_csv('heart.csv')
df.head()

X = np.array(df.loc[ :, df.columns != 'output'])
y = np.array(df['output'])

print(f"X: {X.shape}, y: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

class NeuralNetworkFromScratch:
    def __init__(self, LR, X_train, y_train, X_test, y_test):
        self.w = np.random.randn(X_train_scale.shape[1])
        self.bias = np.random.randn()
        self.LR = LR
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.L_train = []
        self.L_test = []

    def activation(self, x):
        return 1/ (1+np.exp(-x))

    def dactivation(self, x):
        return self.activation(x) * (1-self.activation(x))

    def forward(self, x):
        hidden_1 = np.dot(x, self.w) + self.bias
        activate_1 = self.activation(hidden_1)
        return activate_1

    def backward(self, x, y_true):
        hidden_1 = np.dot(x, self.w) + self.bias
        y_pred = self.forward(x)
        dL_dpred = 2 * (y_pred-y_true)
        dpred_dhidden1 = self.dactivation(hidden_1)
        dhidden1_db = 1
        dhidden1_dw = x

        dL_db = dL_dpred * dpred_dhidden1 *dhidden1_db
        dL_dw = dL_dpred * dpred_dhidden1 * dhidden1_dw

        return dL_db, dL_dw

    def optimizer(self, dL_db, dL_dw):
        self.bias = self.bias - dL_db * self.LR
        self.w = self.w - dL_dw * self.LR

    def train(self, iterations):
        for i in range(iterations):
            random_pos= np.random.randint(len(self.X_train))

            y_train_true = self.X_train[random_pos]
            y_train_pred = self.forward(self.X_train[random_pos])

            loss = np.sum(np.square(y_train_pred - y_train_true))
            self.L_train.append(loss)

            dL_db, dL_dw = self.backward(X_train[random_pos], y_train[random_pos])
            self.optimizer(dL_db, dL_dw)

            loss_sum = 0
            for j in range(len(self.X_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.X_test[j])

                loss_sum += np.square(y_pred - y_true)

            self.L_test.append(loss_sum)

        return "training success"


LR = 0.1
ITERATIONS = 1000
network = NeuralNetworkFromScratch(LR=LR, X_train=X_train_scale, y_train=y_train,
                                   X_test=X_test_scale, y_test=y_test)

print(network.train(ITERATIONS))

total = X_test_scale.shape[0]
correct = 0
y_preds = []
for i in range(total):
    y_true = y_test[i]
    y_pred = np.round(network.forward(X_test_scale[i]))
    y_preds.append(y_pred)
    correct += 1 if y_true == y_pred else 0

print(correct/total)
print(Counter(y_test))
print(confusion_matrix(y_true=y_test, y_pred=y_preds))
