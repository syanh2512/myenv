# #Gradient Descent Example

# import pandas as pd
# import numpy as np
# import torch

# X=torch.tensor([1,2,3,4], dtype=torch.float32)
# Y=torch.tensor([2,4,6,8], dtype=torch.float32)

# w=torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# def forward(x):
#     return w*x

# def loss(y, y_predicted):
#     return ((y_predicted-y)**2).mean()

# lr=0.01
# n_iters=80

# print(f'Prediction before training: f(5)= {forward(5):.3f}')

# for epoch in range(n_iters):
#     y_pred=forward(X)

#     l=loss(Y,y_pred)

#     #gradients
#     l.backward()

#     #update weights
#     with torch.no_grad():
#         w-=lr*w.grad

#     #zero gradients
#     w.grad.zero_()

#     if epoch%10 == 0:
#         print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")

# print(f'Prediction after training: f(5)= {forward(5):.3f}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# #Training Pipeline

# import torch
# import torch.nn as nn

# X=torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
# Y=torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

# X_test = torch.tensor([5], dtype=torch.float32)
# n_samples, n_features = X.shape

# input_size = n_features
# output_size = n_features

# model = nn.Linear(input_size, output_size)

# learning_rate = 0.02
# n_iters = 300

# loss = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# print(f'Prediction before training: f(5)= {model(X_test).item():.3f}')

# for epoch in range(n_iters):
#     y_pred=model(X)

#     l=loss(Y,y_pred)

#     #gradients
#     l.backward()

#     #update weights
#     optimizer.step()

#     #zero gradients
#     optimizer.zero_grad()

#     if epoch%20 == 0:
#         [w, b] = model.parameters()
#         #print([w, b])
#         print(f"epoch {epoch+1}: w = {w[0][0]:.3f}, loss = {l:.8f}")

# print(f'Prediction after training: f(5)= {model(X_test).item():.3f}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# #Linear Regression

# import torch
# import torch.nn as nn
# import numpy as np
# from sklearn import datasets
# import matplotlib.pyplot as plt

# # prepare data
# X_numpy, Y_numpy = datasets.make_regression(n_samples=50, n_features=1, noise=20, random_state=1)
# #print(X_numpy, Y_numpy)
# X = torch.from_numpy(X_numpy.astype(np.float32))
# Y = torch.from_numpy(Y_numpy.astype(np.float32))
# # print(X,Y)
# Y= Y.view(Y.shape[0], 1)
# # print(Y)
# # print(X.shape)
# n_samples, n_features = X.shape

# #model
# input_size = n_features
# output_size = 1

# model = nn.Linear(input_size,output_size)

# #loss and optimizer
# learning_rate = 0.01
# loss = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# #training loop
# num_epochs = 100
# for epoch in range(num_epochs):
#     #forward pass and loss
#     y_pred = model(X)
#     l = loss(y_pred, Y)

#     #backward pass
#     l.backward()

#     #update weights
#     optimizer.step()

#     #zero gradients
#     optimizer.zero_grad()

#     if (epoch+1) %10 == 0:
#         print(f'epoch: {epoch+1}, loss = {l.item():.4f}')

# #plot
# predicted = model(X).detach().numpy()
# plt.plot(X_numpy, Y_numpy, 'ro')
# plt.plot(X_numpy, predicted, 'b')
# plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Logistic Regression

from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#prepare data
bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target

n_samples, n_features = X.shape
# print(n_samples, n_features)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

#scale
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train.view(Y_train.shape[0],1)
Y_test = Y_test.view(Y_test.shape[0],1)


#model, f= wx +b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features) -> None:
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

#loss and optimizer
learning_rate = 0.1
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
num_epochs = 200
for epoch in range(num_epochs):
    #forward pass
    y_pred = model(X_train)
    l = loss(y_pred, Y_train)

    #backward pass, gradients
    l.backward()

    #update gradients
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    if (epoch+1)%20 ==0:
        print(f'epoch : {epoch+1}, loss = {l.item():.5f}')

with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(Y_test).sum() / float(Y_test.shape[0]) * 100
    #print(acc)
    print(f'accuracy: {acc:.3f}%')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






