# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import torchvision  # for downloading the MNIST dataset
from sklearn.metrics import mean_squared_error

losses = []
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden, n_observe=784):
        """Initialize model."""
        # 请补全此处代码
        self.n_hidden = n_hidden
        self.n_observe = n_observe
        self.W = np.zeros((self.n_observe, self.n_hidden))
        self.a = np.random.normal(0, 1, (self.n_observe, 1))
        self.b = np.random.normal(0, 1, (self.n_hidden, 1))
        self.LR = 0.01  # learning rate
        self.EPOCH = 3
        self.GIBBS_N = 1

    def train(self, data):
        """Train model using data."""
        data = np.reshape(data, (60000, 784))
        print(data.shape)
        N = data.shape[0]
        for epoch in range(self.EPOCH):
            print("Epoch: ", epoch)
            for i in range(N):
                v = data[i, :].reshape((784, 1))
                h = sigmoid(self.b + np.dot(self.W.T, v))
                forward_gradient = np.dot(v, h.T)

                v_ = sigmoid(self.a + np.dot(self.W, h))
                h_ = sigmoid(self.b + np.dot(self.W.T, v_))
                backward_gradient = np.dot(v_, h_.T)

                self.W += self.LR * (forward_gradient - backward_gradient)
                self.a += self.LR * (v - v_)
                self.b += self.LR * (h - h_)
                if i % 2500 == 0:
                    loss = mean_squared_error(v, v_)
                    losses.append(loss)
                    print(i, "Loss:", loss)
        print("Training Finished!")

    def sample(self, num_data):
        """Sample from trained model."""
        # num_data is a number sample
        # 请补全此处代码
        v = num_data.reshape((784, 1))
        h = sigmoid(self.b + np.dot(self.W.T, v))
        for i in range(self.GIBBS_N):
            v = sigmoid(self.a + np.dot(self.W, h))
            h = sigmoid(self.b + np.dot(self.W.T, v))
        return v.reshape((28, 28))

def plot():
    labels = raw_data.train_labels  # just for plot!
    num_index = []  # index for number 0-9
    for i in range(10):
        j = 0
        num = -1
        while(num != i):
            j += 1
            num = labels[j]
        num_index.append(j)

    j = 1
    for i in num_index:
        v = mnist[i]
        v_ = rbm.sample(v)
        plt.subplot(5, 4, j)
        plt.imshow(v, cmap='gray')
        j += 1

        plt.subplot(5, 4, j)
        plt.imshow(v_, cmap='gray')
        j += 1
    plt.show()

# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':
    # load mnist dataset, no label with torchvision
    raw_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,                                     # this is training data
        download=False,                 # set it to True to download it if you don't have it
    )
    mnist = raw_data.train_data.numpy() / 255.0  # (60000, 28, 28)
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols
    print(mnist.shape)

    # construct rbm model
    rbm = RBM(100, img_size)
    # train rbm model using mnist
    rbm.train(mnist)
    # sample from rbm model and plot
    plot()
