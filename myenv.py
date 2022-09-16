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

# #Logistic Regression

# from turtle import forward
# import torch
# import torch.nn as nn
# import numpy as np
# from sklearn import datasets
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split


# #prepare data
# bc = datasets.load_breast_cancer()
# X, Y = bc.data, bc.target

# n_samples, n_features = X.shape
# # print(n_samples, n_features)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# #scale
# sc = StandardScaler()

# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)

# X_train = torch.from_numpy(X_train.astype(np.float32))
# X_test = torch.from_numpy(X_test.astype(np.float32))
# Y_train = torch.from_numpy(Y_train.astype(np.float32))
# Y_test = torch.from_numpy(Y_test.astype(np.float32))

# Y_train = Y_train.view(Y_train.shape[0],1)
# Y_test = Y_test.view(Y_test.shape[0],1)


# #model, f= wx +b, sigmoid at the end
# class LogisticRegression(nn.Module):
#     def __init__(self, n_input_features) -> None:
#         super(LogisticRegression, self).__init__()
#         self.linear = nn.Linear(n_input_features, 1)

#     def forward(self, x):
#         y_predicted = torch.sigmoid(self.linear(x))
#         return y_predicted

# model = LogisticRegression(n_features)

# #loss and optimizer
# learning_rate = 0.1
# loss = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# #training loop
# num_epochs = 200
# for epoch in range(num_epochs):
#     #forward pass
#     y_pred = model(X_train)
#     l = loss(y_pred, Y_train)

#     #backward pass, gradients
#     l.backward()

#     #update gradients
#     optimizer.step()

#     #zero gradients
#     optimizer.zero_grad()

#     if (epoch+1)%20 ==0:
#         print(f'epoch : {epoch+1}, loss = {l.item():.5f}')

# with torch.no_grad():
#     y_pred = model(X_test)
#     y_pred_cls = y_pred.round() #làm tròn
#     acc = y_pred_cls.eq(Y_test).sum() / float(Y_test.shape[0]) * 100
#     #print(acc)
#     print(f'accuracy: {acc:.3f}%')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# #Dataset and DataLoader

# import torch
# import torchvision
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import math


# class WineDataset(Dataset):
#     def __init__(self) -> None:
#         #data loading
#         xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
#         self.x = torch.from_numpy(xy[:, 1:])
#         self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
#         self.n_samples = xy.shape[0]

#     def __getitem__(self, index):
#         return self.x[index], self.y[index]


#     def __len__(self):
#         return self.n_samples

# dataset = WineDataset()
# # first_data = dataset[177]
# # features, labels = first_data
# # print(features, labels)
# dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

# num_epochs = 2
# total_samples = len(dataset)
# n_iterations = math.ceil(total_samples/4)
# #print(total_samples, n_iterations)

# for epoch in range(num_epochs):
#     for i, (inputs, labels) in enumerate(dataloader):
#         #forward backward, update
#         if (i+1) % 5 == 0:
#             print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

# torchvision.datasets.MNIST()




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# #Data Transforms

# from genericpath import samefile
# from random import sample
# import torch
# import torchvision
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import math


# class WineDataset(Dataset):
#     def __init__(self, transform=None):
#         #data loading
#         xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
#         self.x = xy[:, 1:]
#         self.y = xy[:, [0]] # n_samples, 1
#         self.n_samples = xy.shape[0]
#         self.transform = transform

#     def __getitem__(self, index):
#         sample = self.x[index], self.y[index]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample


#     def __len__(self):
#         return self.n_samples

# class ToTensor:
#     def __call__(self, sample):
#         inputs, targets = sample
#         return torch.from_numpy(inputs), torch.from_numpy(targets)

# class MulTransform:
#     def __init__(self, factor):
#         self.factor = factor
    
#     def __call__(self, sample):
#         inputs, targets = sample
#         inputs *=self.factor
#         return inputs, targets

# dataset = WineDataset(transform=None)
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)
# print(type(features), type(labels))

# composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
# dataset = WineDataset(transform=composed)
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)
# print(type(features), type(labels))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# #Softmax and Cross-entropy Loss

# import torch
# import torch.nn as nn
# import numpy as np

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x), axis = 0)

# def cross_entropy(real, predicted):
#     loss = -np.sum(real * np.log(predicted))
#     return loss

# # x=np.array([2.0, 1.0, 0.1])
# # outputs = softmax(x)
# # print('softmax numpy:', outputs)

# # x = torch.tensor([2.0, 1.0, 0.1])
# # outputs = torch.softmax(x, dim=0)
# # print(outputs)

# loss = nn.CrossEntropyLoss()

# # 3 samples
# y=torch.tensor([2, 0, 1])

# y_pred_good = torch.tensor([[0.1,1.0,2.1],[2.0,1.0,0.1],[0.1,3.0,0.1]])
# y_pred_bad = torch.tensor([[2.1,1.0,0.1],[0.1,1.0,2.1],[0.1,3.0,0.1]])

# l1 = loss(y_pred_good,y)
# l2 = loss(y_pred_bad, y)

# print(l1.item())
# print(l2.item())


# #_, predictions means only needs predictions
# _,predictions1 = torch.max(y_pred_good, 1)
# _,predictions2 = torch.max(y_pred_bad, 1)

# print(predictions1)
# print(predictions2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# #Activation functions

# #Step, 
# # sigmoid (finally layer of binary classification problems), 
# # tanH (for hidden layers), 
# # ReLU (for hidden layers),
# # Leaky ReLU (improved ReLU, solved vanishing gradient problem),
# # Softmax (good in last layer in multi class classification problems)

# from turtle import forward
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# #option 1: create nn modules
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size) -> None:
#         super(NeuralNet, self).__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         # nn.Sigmoid
#         # nn.Softmax
#         # nn.Tanh
#         # nn.LeakyReLU
#         self.linear2 = nn.Linear(hidden_size,1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         out=self.linear1(x)
#         out = self.relu(out)
#         out = self.linear2(out)
#         out = self.sigmoid(out)
#         return out


# #option 2: use activation function in forward pass
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size) -> None:
#         super(NeuralNet, self).__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size,1)

#     def forwar(self, x):
#         out = torch.relu(self.linear1)
#         out = torch.sigmoid(self.linear2)
#         # F.relu
#         # F.leaky_relu
#         # torch.softmax
#         # torch.tanh
#         return out



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Feed-Forward Neural Net


from tkinter import HIDDEN
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#device config
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
input_size = 784 #28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01

#MNIST dataset
train_dataset=torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
test_dataset=torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
#print(samples.shape, labels.shape)


for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(samples[i][0], cmap='gray') # samples[i][0] 0 mean 1 channel
#plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2= nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #image = ([100, 1, 28, 28])
        #100, 784
        images = images.reshape(-1, 28*28)
        labels = labels

        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)


        #backward
        optimizer.zero_grad()
        loss.backward()
        #update paratemters
        optimizer.step()


        if (i+1) % 100 ==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss:.4f}')

#test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        labels = labels
        outputs = model(images)

        #value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct/n_samples
    print(f'accuracy = {acc}%')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~