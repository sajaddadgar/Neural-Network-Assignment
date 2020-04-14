import loader as mnist
import numpy as np
import matplotlib.pylab as plt


class ANN:
    # constructor
    def __init__(self, layers_size):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []

    # sigmoid function
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    # random initializing weight and bayes
    def initialize_parameters(self):
        np.random.seed(1)
        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))

    # forward propagation
    def forward_propagation(self, X):
        store = {}
        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            A = self.sigmoid(Z)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z
        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        A = self.sigmoid(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z
        return A, store

    # derivative of sigmoid function
    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    # back propagation
    def backward_propagation(self, X, Y, store):
        derivatives = {}
        store["A0"] = X.T
        A = store["A" + str(self.L)]
        dA = -np.divide(Y, A) + np.divide(1 - Y, 1 - A)
        dZ = dA * self.sigmoid_derivative(store["Z" + str(self.L)])
        dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)
        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db
        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])
            dW = 1. / self.n * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
        return derivatives

    # this method first initialize weight and bayes. Then we will have the training running in n_iterations times
    # inside the loop use forward propagation and calculate cost and use back propagation, then update weight and bayes.
    def fit(self, X, Y, learning_rate=0.01, n_iterations=2500):
        np.random.seed(1)
        self.n = X.shape[0]
        self.layers_size.insert(0, X.shape[1])
        self.initialize_parameters()
        for loop in range(n_iterations):
            A, store = self.forward_propagation(X)
            cost = np.squeeze(-(Y.dot(np.log(A.T)) + (1 - Y).dot(np.log(1 - A.T))) / self.n)
            derivatives = self.backward_propagation(X, Y, store)
            for l in range(1, self.L + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                    "db" + str(l)]
            if loop % 100 == 0:
                print(cost)
                self.costs.append(cost)

    #  use current weight and bayes to calculate prediction with forward propagation.
    def predict(self, X, Y):
        A, cache = self.forward_propagation(X)
        n = X.shape[0]
        prediction = np.zeros((1, n))
        for i in range(0, A.shape[1]):
            if A[0, i] > 0.5:
                prediction[0, i] = 1
            else:
                prediction[0, i] = 0
        print("Accuracy: " + str(np.sum((prediction == Y) / n)))

    # visualize cost
    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.show()


def get_dataset():
    # get training data and test data from MNIST dataset
    train_x_from_dataset, train_y_from_dataset, test_x_from_dataset, test_y_from_dataset = mnist.get_data()
    # preprocess for training data
    # get data with label 6 and 9
    index_6 = np.where(train_y_from_dataset == 6)
    index_9 = np.where(train_y_from_dataset == 9)
    # shuffle data
    index = np.concatenate([index_6[0], index_9[0]])
    np.random.seed(1)
    np.random.shuffle(index)
    # get data that we want (data with label 6 and 9)
    train_y = train_y_from_dataset[index]
    train_x = train_x_from_dataset[index]
    # if data's label = 6 set 0 and if data's label = 9 set 1
    train_y[np.where(train_y == 6)] = 0
    train_y[np.where(train_y == 9)] = 1
    # preprocess for test data
    index_6 = np.where(test_y_from_dataset == 6)
    index_9 = np.where(test_y_from_dataset == 9)
    index = np.concatenate([index_6[0], index_9[0]])
    np.random.shuffle(index)
    test_y = test_y_from_dataset[index]
    test_x = test_x_from_dataset[index]
    test_y[np.where(test_y == 6)] = 0
    test_y[np.where(test_y == 9)] = 1
    return train_x, train_y, test_x, test_y


def preprocess_data(train_x, test_x):
    # Normalize data
    train_x = train_x / 255.
    test_x = test_x / 255.
    return train_x, test_x


if __name__ == '__main__':
    # get training data and test data
    train_x, train_y, test_x, test_y = get_dataset()
    # normalization of data
    train_x, test_x = preprocess_data(train_x, test_x)
    # dimension of train and test data
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
    # layers dimension
    layers_dims = [196, 1]
    # create neural network with layers dimension
    ann = ANN(layers_dims)
    ann.fit(train_x, train_y, learning_rate=0.1, n_iterations=1000)
    # predict training data
    ann.predict(train_x, train_y)
    # predict training data
    ann.predict(test_x, test_y)
    # visualization
    ann.plot_cost()