import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)

class NN(nn.Module):
    def __init__(self, n_feature, n_hidden_1, n_hidden_2, n_output):
        super(NN, self).__init__()
        self.hidden_1 = nn.Linear(n_feature, n_hidden_1)
        self.hidden_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.output = nn.Linear(n_hidden_2, n_output)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.5)
        self.loss_function = torch.nn.MSELoss()


    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.output(x)
        return x

    def train(self, x, epochs):
        for i in range(epochs):
            prediction = self.forward(x)
            loss = self.loss_function(prediction, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return prediction
    

def data_create():
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), 1)
    #the unsqueeze function turn the vector into a matrix. 
    y = 2*x.pow(2) + x + 0.2*torch.rand(x.size())
    x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)
    return x, y


mynet = NN(1, 10, 5, 1)

print(mynet)

x, y = data_create()
prediction = mynet.train(x, 1000)