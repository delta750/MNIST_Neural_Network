from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image


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
    print "Train data array size: ", train_data.shape
    tmp = []
    for i in test_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    # load test data in N*D array (10000x784 for MNIST)
    #                             divided by 255 to achieve normalization
    test_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int') / 255
    print "Test data array size: ", test_data.shape
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

        return np.log( 1 + np.exp(a))

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

        return 1- np.power(tahn(a),2)

    def  derivative_of_cosine(a):
        return -np.sin(a)


    if act_fun == 1:
        return logarithmic, derivative_of_logarithmic
    if act_fun == 2:
        return  tahn, derivative_of_tahn
    if act_fun == 3:
        return cosine, derivative_of_cosine


class ML_NeuralNetwork:

    def __init__(self, x_input_train, hidden_neurons, chosen_activation_function, lamda, number_of_iteration, t, eta, tolerance ):
        self.X_train = np.hstack((np.ones((x_input_train.shape[0], 1)), x_input_train))
        self.hidden_neurons = hidden_neurons
        self.activation_function , self.derActivationFunc = activation_function(chosen_activation_function)
        self.lamda = lamda
        self.number_of_iteration = number_of_iteration
        self.t = t
        self.eta = eta
        self.tolerance = tolerance
        # T is Nb x K, T = outputs -> # of possible classes
        self.number_of_outputs = t.shape[1]
        # initialize random weights
        # W1 is M x (D+1), M = hidden units
        self.weights1 = np.random.randn(self.hidden_neurons,x_input_train.shape[1])
        # W2 is K x D +1, M = hidden units, K = k categories
        self.weights2 = np.random.rand(self.number_of_outputs, self.hidden_neurons + 1)


        def feedForward(self, x, t, weights1, weights2):

            # We calculate first the dot product between weights1 and x and then we pass it as an arg in the chosen act_fun and its gradient func
            firstLayerResult = self.activation_function(np.dot(x,weights1.T))

            # Add bias
            firstLayerResult = np.hstack((np.ones((firstLayerResult.shape[0], 1)), firstLayerResult))

            # Y is the output
            y = np.dot(firstLayerResult, weights2.T)
            # softmaxResult is the probability
            softmaxResult = softmax(y)
            max_error = np.max(softmaxResult, axis=1)

            # Loss function
            Ew =  np.sum(t * y) - np.sum(max_error) - \
                np.sum(np.log(np.sum(np.exp(y - np.array([max_error, ] * y.shape[1]).T), 1))) - \
                (0.5 * lamda) * np.sum(np.square(weights2))

            # Gradient ascent calculation for W2, (T-Y).T*Z -λW2

            grand2 = np.dot((t-y).T , firstLayerResult) -eta* weights2

            # Gradient ascent calculation for W1
            # TODO

            return Ew, grand2


        # Documentation

       # np.ones Return a new array of given shape and type, filled with ones.
       #x_input_train.shape[0] its the rows ---> n  https://stackoverflow.com/questions/10200268/what-does-shape-do-in-for-i-in-rangey-shape0
       # X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
       #input layer(  # features), the size of the hidden layer (variable parameter to be tuned), and the number of the output layer (# of possible classes)







































