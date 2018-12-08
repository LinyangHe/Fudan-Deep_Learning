import numpy as np

##################
# ???
from model import Pointer_network
# ???
####################
import torch
MAX_EPOCH = 100000


#############################
#?????   construct your model


model = Pointer_network(     )

#?????
#############################


batch_size = 256
# optimizer = torch.optim.Adam(params= model.parameters() ,lr= 0.001)
# loss_fun = torch.nn.CrossEntropyLoss()

def getdata(shiyan = 1 , batch_size = batch_size ):
    if shiyan == 1:
        high = 100
        senlen = 5
        x = np.array([np.random.choice(range(high), senlen, replace=False) for _ in range(batch_size)])
        y = np.argsort(x)
    elif shiyan == 2:
        high = 100
        senlen = 10
        x = np.array([np.random.choice(range(high), senlen, replace=False) for _ in range(batch_size)])
        y = np.argsort(x)
    elif shiyan == 3 :
        senlen = 5
        x = np.array([np.random.random(senlen) for _ in range(batch_size)])
        y = np.argsort(x)
    return x,y

def evaluate():
    accuracy_sum = 0.0
    for i in range(300):

        test_x ,test_y = getdata(shiyan = 1)

        ###############################
        # ????


        # compute prediction ,and then get the accuracy
        #############################
        accuracy_sum += accuracy
    print('accuracy is ',accuracy_sum/(batch_size*300.0))

for epoch in range(MAX_EPOCH):

    train_x ,train_y = getdata(shiyan=1)

    ############################
    # compute the  prediction



    ###########################
    # loss = loss_fun(prediction ,train_y)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    if epoch % 300 ==0 and epoch != 0:
        # print(epoch ,' \t loss is \t',loss.item())
        # print loss

    if epoch % 2000==0  and epoch != 0: #
        evaluate()




