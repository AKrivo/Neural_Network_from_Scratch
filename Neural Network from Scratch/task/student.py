import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt
from tqdm import tqdm


# scroll to the bottom to start coding your solution


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    # write your code here


# Stage 1/7

def scale(X_train, X_test):
    X_tr = X_train / X_train.max(axis=1, keepdims=True)
    X_te = X_test / X_test.max(axis=1, keepdims=True)
    return X_tr, X_te


def xavier(n_in, n_out):
    part = n_in + n_out
    bound = (np.sqrt(6)) / (np.sqrt(part))
    w = np.random.uniform(-bound, bound, (n_in, n_out))
    return w


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mse(pred, real):
    return np.mean((pred - real) ** 2)


def dmse(pred, real):
    return 2 * (pred - real)


def dsigm(x):
    return sigmoid(x) * (1 - sigmoid(x))


def accuracy(model, X, y):
    ypred = model.forward(X)
    total = 0
    true = 0
    for i in range(0, len(y)):
        if y[i].argmax() == ypred[i].argmax():
            true += 1
        total += 1
    return true / total


def batch_split(X, y, size):
    # Splitting the set's to batches with given size
    train_batch_X = np.array_split(X, len(X) // size)
    train_batch_y = np.array_split(y, len(y) // size)
    return train_batch_X, train_batch_y


def epoch_train(estimator, X, y, alpha, size):
    train_batch_X, train_batch_y = batch_split(X, y, size)
    for i in range(0, len(train_batch_X)):
        estimator.forward(train_batch_X[i])
        estimator.backprop(train_batch_X[i], train_batch_y[i], alpha)
    loss = mse(estimator.forward(X), y)
    acc = accuracy(estimator, X, y)
    return loss, acc


def multy_epoch(n, model, X, Y, alpha, batch_size):
    lossl = []
    accl = []
    num = 0
    for i in tqdm(range(0, n)):
        a, b = epoch_train(model, X, Y, alpha, batch_size)
        num += i
        lossl.append(a)
        accl.append(b)
    return lossl, accl


# Stage 2/7

class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        # Initiate weights and biases using Xavier
        self.w = xavier(n_features, n_classes)
        self.b = xavier(1, n_classes)
        self.forw = None

    def forward(self, X):
        # Forward step
        self.forw = sigmoid(np.dot(X, self.w) + self.b)
        return self.forw

    def backprop(self, X, y, alpha):
        activation = self.forw
        delta = dmse(activation, y) * dsigm((np.dot(X, self.w) + self.b))
        cw = np.dot(X.transpose(), delta) / X.shape[0]
        cb = np.mean(delta, axis=0)
        # Updating weights and biases.
        self.w = self.w - alpha * cw
        self.b = self.b - alpha * cb
        # Calculate and return the loss for monitoring
        # loss = mse(self.forward(X), y)
        return self.w, self.b


class TwoLayerNeural():
    def __init__(self, n_features, n_neurons, n_classes):
        self.w = [xavier(n_features, n_neurons), xavier(n_neurons, n_classes)]
        self.b = [xavier(1, n_neurons), xavier(1, n_classes)]
        self.forw = []

    def forward(self, X):
        # Calculating feedforward
        self.forw = [X]
        for b, w in zip(self.b, self.w):
            X = sigmoid(np.dot(X, w) + b)
            self.forw.append(X)
        return self.forw[-1]

    def backprop(self, X, y, alpha):
        # Calculating gradients for each of
        # your weights and biases.
        g3 = np.dot(self.forw[1], self.w[1]) + self.b[1]
        g2 = np.dot(self.forw[0], self.w[0]) + self.b[0]
        delta3 = dmse(self.forw[2], y) * dsigm(g3)
        delta2 = np.dot(delta3, self.w[1].transpose()) * dsigm(g2)
        cw3 = np.dot(self.forw[1].transpose(), delta3) / self.forw[1].shape[0]
        cw2 = np.dot(X.transpose(), delta2) / X.shape[0]
        b3 = np.mean(delta3, axis=0)
        b2 = np.mean(delta2, axis=0)
        # Updating your weights and biases.
        self.w[1] = self.w[1] - alpha * cw3
        self.b[1] = self.b[1] - alpha * b3
        self.w[0] = self.w[0] - alpha * cw2
        self.b[0] = self.b[0] - alpha * b2
        return self.w, self.b





##################################################
X_trainS, X_testS = scale(X_train, X_test)
rows, n_featuresX = X_trainS.shape
n_classesX = 10
hidden_layer = 64
test = OneLayerNeural(n_featuresX, n_classesX)
test_multy = TwoLayerNeural(n_featuresX, hidden_layer, n_classesX)
# a = test_multy.forward(X_trainS[:2])
# b = test_multy.backprop(X_trainS[:2], y_train[:2],  alpha=0.1)
# c =  test_multy.forward(X_trainS[:2])
# q = [mse(c, y_train[:2])]
l, ac = multy_epoch(20, test_multy, X_trainS, y_train, 0.5, 100)
g = plot(l, ac, filename='plot_2')
print(ac)