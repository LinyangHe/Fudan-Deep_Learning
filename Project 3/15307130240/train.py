import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Data
from model import PointerNetwork

LR = 0.001
EPOCH = 5
batch_size = 250
total_size = 10000
input_size = 1
weight_size = 256
hidden_size = 512

def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

def getdata(shiyan=1, batch_size=batch_size):
    if shiyan == 1:
        high = 100
        senlen = 5
        x = np.array([np.random.choice(range(high), senlen, replace=False)
                      for _ in range(batch_size)])
        y = np.argsort(x)
    elif shiyan == 2:
        high = 100
        senlen = 10
        x = np.array([np.random.choice(range(high), senlen, replace=False)
                      for _ in range(batch_size)])
        y = np.argsort(x)
    elif shiyan == 3:
        senlen = 5
        x = np.array([np.random.random(senlen) for _ in range(batch_size)])
        y = np.argsort(x)
    elif shiyan == 4:
        senlen = 10
        x = np.array([np.random.random(senlen) for _ in range(batch_size)])
        y = np.argsort(x)
    return x, y

def evaluate(model, X, Y):
    probs = model(X) 
    prob, indices = torch.max(probs, 2) 
    equal_cnt = sum([1 if torch.equal(index.detach(), y.detach()) else 0 for index, y in zip(indices, Y)])
    accuracy = equal_cnt/len(X)
    print('Acc: {:.2f}%'.format(accuracy*100))

#Get Dataset
x, y = getdata(shiyan=2, batch_size = total_size)
x = to_cuda(torch.FloatTensor(x).unsqueeze(2))     
y = to_cuda(torch.LongTensor(y)) 
#Split Dataset
train_size = (int)(total_size * 0.9)
train_X = x[:train_size]
train_Y = y[:train_size]
test_X = x[train_size:]
test_Y = y[train_size:]
#Build DataLoader
train_data = Data.TensorDataset(train_X, train_Y)
data_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = batch_size,
    shuffle = True,
)


#Define the Model
model = PointerNetwork(input_size, weight_size, hidden_size, is_GRU=False)
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fun = torch.nn.CrossEntropyLoss()


#Training...
print('Training... ')
for epoch in range(EPOCH):
    for (batch_x, batch_y) in data_loader:
        probs = model(batch_x)         
        outputs = probs.view(-1, batch_x.shape[1])
        batch_y = batch_y.view(-1) 
        loss = loss_fun(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 2 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
        evaluate(model, train_X, train_Y)
#Test...    
print('Test...')
evaluate(model, test_X, test_Y)