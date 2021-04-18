#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Linear Regression Regressor

   Brown CS142, Spring 2020
'''
import random
import numpy as np


def squared_error(predictions, Y):
    '''
    Computes sum squared loss (the L2 loss) between true values, Y, and predictions.

    @params:
        Y: A 1D Numpy array with real values (float64)
        predictions: A 1D Numpy array of the same size of Y
    @return:
        sum squared loss (the L2 loss) using predictions for Y.
    '''
    vals = len(Y)
    total = 0
    for pos in range(vals):
        holder = (Y[pos] - predictions[pos])**2
        # if(holder > 200):
        #     print('here')
        #     print(Y[pos])
        #     print(predictions[pos])
        #
        # print(holder)
        total = total + holder

    return total

class LinearRegression:
    '''
    LinearRegression model that minimizes squared error using matrix inversion.
    '''
    def __init__(self, n_features):
        '''
        @attrs:
            n_features: the number of features in the regression problem
            weights: The weights of the linear regression model.
        '''
        self.n_features = n_features + 1  # An extra feature added for the bias value
        self.weights = np.zeros(n_features + 1)

    def train(self, X, Y):
        '''
        Trains the LinearRegression model weights using matrix inversion.

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            None
        '''
        self.train_solver(X, Y)

    def train_solver(self, X, Y):
        '''
        Trains the LinearRegression model by finding the optimal set of weights
        using matrix inversion.

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            None
        '''
        x_trans = np.transpose(X)
        pre_inv = np.matmul(x_trans, X)
        post_inv = np.linalg.pinv(pre_inv)
        second = np.matmul(post_inv, x_trans)
        final = np.matmul(second, Y)
        print(final)
        self.weights = final

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        print(self.weights)
        num_rows = len(X)
        predictions = np.zeros(num_rows)
        for pos in range(num_rows):
            predictions[pos] = np.inner(X[pos], self.weights)

        return predictions

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            A float number which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        print(squared_error([row[2] for row in X], (-1*Y))/X.shape[0])
        return squared_error(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            A float number which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]
