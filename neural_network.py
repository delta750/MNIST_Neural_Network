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
    print "Train truth array size: ", train_truth.shape
    print "Test truth array size: ", test_truth.shape
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













































