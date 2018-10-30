###########################
# Created by Linyang He(15307130240@fudan.edu.cn)
# Implement the Neural Network with the usage of PyTorch(0.4.1)
# NN would approximate the function: y = 2*x^2 + 3*x 
###########################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math

torch.manual_seed(1)

class NN(nn.Module):
    def __init__(self, n_feature, n_hidden_1, n_hidden_2, n_output):
        super(NN, self).__init__()
        #There are two hidden layers and one output layer
        self.hidden_1 = nn.Linear(n_feature, n_hidden_1)
        self.hidden_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.output = nn.Linear(n_hidden_2, n_output)
        #We use the SGD as the optimizer here, and the learning rate is 0.0001
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.0001)
        #The loss function for our NN is MSE
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x):
        #The activation function we use is ReLU.
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.output(x)
        return x

    def train(self, x, y, epochs):
        #We compute the norm cube of y in advance to compute the relative error
        y_norm_cube = y.norm()**2

        for i in range(epochs):
            #Implement forward propagation
            prediction = self.forward(x)
            #Compute the loss
            loss = self.loss_function(prediction, y)
            #Clears the gradients of all optimized torch.Tensor s.
            self.optimizer.zero_grad()
            #Implement backward propagation
            loss.backward()
            #Performs a single optimization step (parameter update).
            self.optimizer.step()
            #If the relative errror is less than 1e-6, then finish the training
            if loss / y_norm_cube < 1e-6:
                break
            #Track the training information
            if i % 1000 == 0:
                print(i, loss)
        #Print the iterations
        print('Iterations: ',i)

#Define different functions
def func1(x):
    y = 2*x.pow(2) + 3*x + 0.2*torch.rand(x.size())
    return y
def func2(x):
    y = np.sin(x) + 0.01*torch.rand(x.size())
    return y
def func3(x):
    y = np.log(x) + 0.01*torch.rand(x.size())
    return y
def func4(x):
    y = np.exp(x) + 0.01*torch.rand(x.size())
    return y

#Create training data and test data.
def data_create():
    #the unsqueeze function turn the vector into a matrix. 
    x = torch.unsqueeze(torch.linspace(-10, 10, 500), 1)
    # x = torch.unsqueeze(torch.linspace(0.0001, 1, 500), 1)
    y = func1(x)

    x_test = torch.from_numpy(np.random.uniform(-10, 10, 100))
    # x_test = torch.from_numpy(np.random.uniform(0.0001, 1, 100))    
    x_test = torch.unsqueeze(x_test, 1).float()
    y_test = func1(x_test)

    return x, y, x_test, y_test

'''
Training stage
'''

#Define a neural network
mynet = NN(1, 5, 10, 1)
# mynet = NN(1, 15, 10, 1)

#Print the network structure information
print('Network structure:\n',mynet)

#Get the data
x, y, x_test, y_test = data_create()

#Train the neural network
mynet.train(x, y, 500000)
#Compute the prediction
prediction = mynet(x)
#Compute the loss based on the function we used when we train the model
loss = mynet.loss_function(prediction, y)
#Compute and print the relative errorr. 
rel_loss = loss / y.norm() ** 2
print('Relative error for training set is', rel_loss.data.numpy())


'''
Test stage
'''
prediction_test = mynet(x_test)
loss_test = mynet.loss_function(prediction_test, y_test)
rel_loss_test = loss_test / y_test.norm() ** 2
print('Relative error for test set is', rel_loss_test.data.numpy())


'''
Plot stage
'''
plt.figure(1, figsize=(10, 3))
plt.subplot(121)
plt.title('y = 2x^2 + 3x(Training Set)')
p1 = plt.scatter(x.data.numpy(), y.data.numpy(),s=1, c = 'blue', marker = '*')
p2, = plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2, alpha = 0.8)
plt.legend([p1, p2],['Real Function','NN Learned'])
# plt.scatter(x.data.numpy(), prediction.data.numpy(),  s = 1, c = 'r')


plt.subplot(122)
plt.title('y = 2x^2 + 3x(Test Set)')
p3 = plt.scatter(x_test.data.numpy(), y_test.data.numpy(), s=1, c = 'blue', marker = '*')
p4 = plt.scatter(x_test.data.numpy(), prediction_test.data.numpy(), s = 2, c = 'r')
plt.legend([p3,p4],['Real Function','NN Learned'])
plt.show()