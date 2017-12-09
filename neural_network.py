from __future__ import division

import numpy as np


def softmax(y):
    max_of_rows = np.max(y, 1)
    m = np.array([max_of_rows, ] * y.shape[1]).T
    y = y - m
    y = np.exp(y)
    return y / (np.array([np.sum(y, 1), ] * y.shape[1])).T


def load_data():
    """
    Loads the MNIST dataset. Reads the training files and creates matrices.
    :return: train_data:the matrix with the training data
    test_data: the matrix with the data that will be used for testing
    train_truth: the matrix consisting of one
                        hot vectors on each row(ground truth for training)
    test_truth: the matrix consisting of one
                        hot vectors on each row(ground truth for testing)
    """
    train_files = ['data/train%d.txt' % (i,) for i in range(10)]
    test_files = ['data/test%d.txt' % (i,) for i in range(10)]
    tmp = []
    for i in train_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    # load train data in N*D array (60000x784 for MNIST)
    #                              divided by 255 to achieve normalization
    train_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int') / 255
    print ("Train data array size: ", train_data.shape)
    tmp = []
    for i in test_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    # load test data in N*D array (10000x784 for MNIST)
    #                             divided by 255 to achieve normalization
    test_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int') / 255
    print ("Test data array size: ", test_data.shape)
    tmp = []
    for i, _file in enumerate(train_files):
        with open(_file, 'r') as fp:
            for line in fp:
                tmp.append([1 if j == i else 0 for j in range(0, 10)])
    train_truth = np.array(tmp, dtype='int')
    del tmp[:]
    for i, _file in enumerate(test_files):
        with open(_file, 'r') as fp:
            for _ in fp:
                tmp.append([1 if j == i else 0 for j in range(0, 10)])
    test_truth = np.array(tmp, dtype='int')
    print("Train truth array size: ", train_truth.shape)
    print("Test truth array size: ", test_truth.shape)
    return train_data, test_data, train_truth, test_truth


X_train, X_test, y_train, y_test = load_data()


def activation_function(act_fun):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def logarithmic(a):

        # return np.log( 1 + np.exp(a))
        m = np.maximum(0, a)
        return m + np.log(np.exp(-m) + np.exp(a - m))

    def tahn(a):

        return (np.exp(2 * a) - 1) / (np.exp(2 * a) + 1)

    def cosine(a):

        return np.cos(a)

    """
    Because later we will need to compute the derivative of the activation function,
    we compute it here

    """

    def derivative_of_logarithmic(a):
        # if we take the derivative of logaritmic we end up with sigmoid
        return sigmoid(a)

    def derivative_of_tahn(a):

        return 1 - np.power(tahn(a), 2)

    def derivative_of_cosine(a):
        return -np.sin(a)

    if act_fun == 1:
        return logarithmic, derivative_of_logarithmic
    if act_fun == 2:
        return tahn, derivative_of_tahn
    if act_fun == 3:
        return cosine, derivative_of_cosine


class ML_NeuralNetwork:

    def __init__(self, x_input_train, hidden_neurons, hidden_layer_act_func, lamda, number_of_iteration, t, eta,
                 tolerance):
        self.X_train = np.concatenate((np.ones((x_input_train.shape[0], 1)), x_input_train), axis=1)
        self.hidden_neurons = hidden_neurons
        self.activation_function, self.derActivationFunc = activation_function(hidden_layer_act_func)
        self.lamda = lamda
        self.number_of_iteration = number_of_iteration
        self.t = t
        self.eta = eta
        self.tolerance = tolerance
        # T is Nb x K, T = outputs -> # of possible classes
        self.number_of_outputs = t.shape[1]
        # initialize random weights
        # W1 is M x (D+1), M = hidden units
        self.weights1 = np.random.rand( self.hidden_neurons, self.X_train.shape[1]) * 0.2 - 0.1
        # W2 is K x D +1, M = hidden units, K = k categories
        self.weights2 = np.random.rand(self.number_of_outputs, self.hidden_neurons + 1)

    def feedForward(self, x, t, weights1, weights2):

        # We calculate first the dot product between weights1 and x and then we pass it as an arg in the chosen act_fun and its gradient func
        firstLayerResult = self.activation_function(np.dot(x, weights1.T))

        # Add bias
        firstLayerResult_with_bias = np.concatenate((np.ones((firstLayerResult.shape[0], 1)), firstLayerResult), axis=1)

        # Y is the output
        y = np.dot(firstLayerResult_with_bias, weights2.T)

        max_error = np.max(y, axis=1)

        # Loss function
        Ew = np.sum(t * y) - np.sum(max_error) - \
             np.sum(np.log(np.sum(np.exp(y - np.array([max_error, ] * self.number_of_outputs).T), 1))) - \
             (0.5 * self.lamda) * np.sum(weights2 * weights2)

        # softmaxResult is the probability
        softmaxResult = softmax(y)

        gradw2 = np.dot((t - softmaxResult).T, firstLayerResult_with_bias) - self.lamda * weights2

        # Gradient ascent calculation for W1 (we get rid of the bias from w2)
        gradw1 = (weights2[:, 1:].T.dot((t - softmaxResult).T) * self.derActivationFunc(x.dot(weights1.T)).T).dot(
            x)

        return Ew, gradw1, gradw2

    def neural_network_train(self):
        Ew_old = -np.inf
        for i in range(self.number_of_iteration):
            error, gradWeight1, gradWeight2 = self.feedForward(self.X_train, self.t, self.weights1, self.weights2)
            print("iteration #", i, "and error=", error)

            if np.absolute(error - Ew_old) < self.tolerance:
                break
            # grapse ton tipo
            self.weights1 += self.eta * gradWeight1

            self.weights2 += self.eta * gradWeight2
            Ew_old = error

    def neural_network_test(self, test_data, test_truth_data):
        # First we add in the test data the bias
        test_data_with_bias = np.concatenate((np.ones((test_data.shape[0], 1)), test_data), axis=1)

        # Feed forward

        resultsOfActF = self.activation_function(np.dot(test_data_with_bias, self.weights1.T))

        # We now the bias
        resultsOfActF_with_bias = np.concatenate((np.ones((resultsOfActF.shape[0], 1)), resultsOfActF), axis=1)

        y = np.dot(resultsOfActF_with_bias, self.weights2.T)

        probabilitiesResult = softmax(y)

        decision = np.argmax(probabilitiesResult, axis=1)

        error = 0

        for i in range(len(test_truth_data)):
            if np.argmax(test_truth_data[i]) != decision[i]:
                error += 1

        print("The Error is", error / test_truth_data.shape[0] * 100, "%")

    def grad_check(self):
        epsilon = 1e-6
        _list = np.random.randint(self.X_train.shape[0], size=5)
        x_sample = np.array(self.X_train[_list, :])
        t_sample = np.array(self.t[_list, :])

        Ew, gradWeight1, gradWeight2 = self.feedForward(x_sample, t_sample, self.weights1, self.weights2)

        print("gradWeight1 shape: ", gradWeight1.shape)
        print("gradWeight2 shape: ", gradWeight2.shape)
        numericalGrad1 = np.zeros(gradWeight1.shape)
        numericalGrad2 = np.zeros(gradWeight2.shape)

        # W1 gradcheck
        for k in range(0, numericalGrad1.shape[0]):
            for d in range(0, numericalGrad1.shape[1]):
                w_temp = np.copy(self.weights1)
                w_temp[k, d] += epsilon
                e_plus, _, _ = self.feedForward(x_sample, t_sample, w_temp, self.weights2)

                w_tmp = np.copy(self.weights1)
                w_tmp[k, d] -= epsilon
                e_minus, _, _ = self.feedForward(x_sample, t_sample, w_tmp, self.weights2)
                numericalGrad1[k, d] = (e_plus - e_minus) / (2 * epsilon)

        # Absolute norm

        print("The difference estimate for gradient of w1 is : ",
              np.amax(np.abs(gradWeight1 - numericalGrad1)))

        # W2 gradcheck
        for k in range(0, numericalGrad2.shape[0]):
            for d in range(0, numericalGrad2.shape[1]):
                w_temp = np.copy(self.weights2)
                w_temp[k, d] += epsilon
                e_plus, _, _ = self.feedForward(x_sample, t_sample, self.weights1, w_temp)

                w_tmp = np.copy(self.weights2)
                w_tmp[k, d] -= epsilon

                e_minus, _, _ = self.feedForward(x_sample, t_sample, self.weights1, w_temp)
                numericalGrad1[k, d] = (e_plus - e_minus) / (2 * epsilon)

        # Absolute norm

        print("The difference estimate for gradient of w2 is : ",
              np.amax(np.abs(gradWeight2 - numericalGrad2)))


if __name__ == '__main__':
    print("Neural Network multi classification, for Mnist dataset")

    act_func = int(input(
        "Please choose an activation function( insert a number between 1-3): 1 for logarithmic, 2 for tanh and 3 for cosine "))
    hidden_units = int(input("Please choose the number of hidden neurons:"))

    act_func = int(act_func)
    hidden_units = int(hidden_units)
    x_data, test_data, train_truth_data, test_truth_data = load_data()
    lamda = 0.1
    eta = 0.5 / x_data.shape[0]
    # Maximum number of iteration of gradient ascend
    number_of_iterations = 800
    tolerance = 1e-6
    mlnn = ML_NeuralNetwork(x_data, hidden_units, act_func, lamda, number_of_iterations, train_truth_data, eta,
                            tolerance)

    mlnn.grad_check()
    mlnn.neural_network_train()
    mlnn.neural_network_test(test_data, test_truth_data)

    # Documentation

    # np.ones Return a new array of given shape and type, filled with ones.
    # x_input_train.shape[0] its the rows ---> n  https://stackoverflow.com/questions/10200268/what-does-shape-do-in-for-i-in-rangey-shape0
    # X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    # input layer(  # features), the size of the hidden layer (variable parameter to be tuned), and the number of the output layer (# of possible classes)


