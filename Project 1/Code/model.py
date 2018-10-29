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

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.output(x)
        return x

mynet = NN(1, 5, 5, 1)


print(mynet)

def data_create():
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2*torch.rand(x.size())
    x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)
    return x, y
