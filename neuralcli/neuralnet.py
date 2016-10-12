from __future__ import absolute_import, division, print_function
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder  
import numpy as np
from scipy.io import loadmat 
import csv
import sys
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# supress numpy error warnings so as not to hinder the cli output
np.seterr(all='ignore')

class NeuralNet():
    """The main neural network class for training"""

    def __init__(self, X, Y, writer, output="./params", lam=1, maxiter=250, norm=False):
        """
        Arguments:
            X {np.ndarray} -- The training set
            Y {np.ndarray} -- The expected output of the training set
            writer {class} -- A writer interface which implements the write() method
        
        Keyword Arguments:
            output {str} -- Where to save the trained params to (default: {"./params"})
            lam {number} -- Lambda term for regularization (default: {1})
            maxiter {number} -- Max iterations for minimization (default: {250})
        """

        X = np.matrix(X)
        Y = np.matrix(Y)
        if norm:
            X = normalize(X)

        self.X = X
        self.Y = Y

        self.num_labels = np.shape(Y)[1]
        self.input_size = np.shape(X)[1]
        self.hidden_size = np.shape(X)[1]  

        self.lam = lam
        self.output = output
        self.params = self.generate_params()
        self.maxiter = maxiter
        self.writer = writer
        self.minval = 0.000000001

    def set_input_size(self, input_size):
        """
        set the input size of the network

        Arguments:
            input_size {int}
        """

        self.input_size = input_size

    def set_num_labels(self, num_labels):
        """
        set the num of labels of output
        
        Arguments:
            num_labels {int}
        """

        self.num_labels = num_labels

    def set_hidden_size(self, hidden_size):
        """
        set the hidden layer size
        
        Arguments:
            hidden_size {int}
        """

        self.hidden_size = hidden_size
    def set_params(self, params):
        """
        set the params
        
        Arguments:
            params {np.ndarray} -- params
        """

        self.params = params

    def set_X(self, X, norm=True):
        """
        Set the training set
        
        Arguments:
            X {np.ndarray} -- The input layer
        
        Keyword Arguments:
            norm {bool} -- Should the input be normalized (default: {True})
        """

        if norm:
            X = normalize(X)

        self.X = X

    def set_Y(self, Y):
        """
        Set the expected output
        
        Arguments:
            Y {np.ndarray} -- The expected output
        """

        self.Y = Y

    def train(self, verbose=False, save=True):
        """
        minimize a cost function defined under backpropogation
        
        Keyword Arguments:
            verbose {bool} -- should the backpropgation print progress (default: {False})
            save {bool} -- should output of parameters be saved to a file (default: {True})
        
        Returns:
            np.ndarray
        """
        
        fmin = minimize(fun=self.fit, x0=self.params, args=(self.X, self.Y, verbose),  
                        method='TNC', jac=True, options={'maxiter': self.maxiter})

        if save:
            writer = csv.writer(open(self.output, 'w'))
            writer.writerow(fmin.x)

        self.params = fmin.x

    def generate_params(self):
        """
        generate a random sequence of weights for the parameters of the neural network

        Returns:
            np.ndarray
        """

        return (np.random.random(size=self.hidden_size * (self.input_size + 1) + self.num_labels * (self.hidden_size + 1)) - 0.5) * 0.25

    def load_params(self, name):
        """
        load parameters from a csv file
        
        Arguments:
            name {string} -- the location of the file
        
        Returns:
            np.ndarray -- the loaded params
        """

        return np.loadtxt(open(name,"rb"), delimiter=",",skiprows=0, dtype="float")

    def sigmoid(self, z):
        """
        compute the sigmoid activation function
        
        Arguments:
            z {mixed} 
        
        Returns:
            number 
        """

        return 1 / (1 + np.exp(-z))

    def sigmoid_gradient(self, z):  
        """
        gradient of the sigmoid func

        Arguments:
            z {mixed}
        
        Returns:
            np.ndarray|float
        """

        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    def reshape_theta(self, params):
        """
        reshape the 1 * n parameter vector into the correct shape for the first and second layers
        
        Arguments:
            params {np.ndarray} -- a vector of weights
        
        Returns:
            theta1 {np.ndarray}
            theta2 {np.ndarray}
        """

        theta1 = np.matrix(np.reshape(params[:self.hidden_size * (self.input_size + 1)], (self.hidden_size, (self.input_size + 1))))
        theta2 = np.matrix(np.reshape(params[self.hidden_size * (self.input_size + 1):], (self.num_labels, (self.hidden_size + 1))))

        return theta1, theta2

    def feed_forward(self, X, theta1, theta2):  
        """
        run forward propgation using a value of X
        
        Arguments:
            X {np.ndarray} -- Input set
            theta1 {np.ndarray} -- The first layer weights
            theta2 {np.ndarray} -- The second layer weights
        
        Returns:
            a1 {np.ndarray}
            z2 {np.ndarray}
            a2 {np.ndarray}
            z3 {np.ndarray}
            h  {np.ndarray}
        """

        m = X.shape[0]

        a1 = np.insert(X, 0, values=np.ones(m), axis=1)

        z2 = a1 * theta1.T
        a2 = np.insert(self.sigmoid(z2), 0, values=np.ones(m), axis=1)
        z3 = a2 * theta2.T
        h = self.sigmoid(z3)

        return a1, z2, a2, z3, h

    def fit(self, params, X, y, output=True):  
        """
        main function to run a single pass on the nn. First run forward propgation to get the error of output given some
        parameters and then perfom backpropgation to work out the gradient of the function using the given weights.
        
        Arguments:
            params {np.ndarray} -- weight layer parameters
            X {np.ndarray} -- Input matrix
            y {np.ndarray} -- Expected output matrix
        
        Keyword Arguments:
            output {bool} -- print to the writer (default: {True})
        
        Returns:
            J {float64} -- the margin of error with the given weights
            grad {np.ndarray} -- the matrix of gradients for the given weights
        """

        m = X.shape[0]
        X = np.matrix(X)
        y = np.matrix(y)

        theta1, theta2 = self.reshape_theta(params)
        a1, z2, a2, z3, h = self.feed_forward(X, theta1, theta2)

        # initializations
        J = 0
        delta1 = np.zeros(theta1.shape)
        delta2 = np.zeros(theta2.shape)

        J = self.get_cost(y, h) / m

        J += (float(self.lam) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
        if output:
            self.writer.write(J)

        for t in range(m):
            a1t = a1[t,:]
            z2t = z2[t,:]
            a2t = a2[t,:]
            ht = h[t,:]
            yt = y[t,:]

            d3t = ht - yt

            z2t = np.insert(z2t, 0, values=np.ones(1)) 
            d2t = np.multiply((theta2.T * d3t.T).T, self.sigmoid_gradient(z2t))

            delta1 = delta1 + (d2t[:,1:]).T * a1t
            delta2 = delta2 + d3t.T * a2t

        delta1 = delta1 / m
        delta2 = delta2 / m

        delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * self.lam) / m
        delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * self.lam) / m

        grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

        return J, grad

    def get_cost(self, y, h):
        """
        get the cost of prediction, the error margin
        
        Arguments:
            y {np.ndarray} -- The expected output
            h {np.ndarray} -- The prediction array
        
        Returns:
            cos {float64} -- the margin of error with the given weights
        """
        first_term = np.multiply(-y, np.log(h.clip(self.minval)))
        second_term = np.multiply((1 - y), np.log(1 - h.clip(self.minval)))

        return np.sum(first_term - second_term)

    def accuracy(self, type='train'):
        """
        get the accuracy of the learned parameters on a specific set
        """

        examples = len(self.Y)
        theta1, theta2 = self.reshape_theta(self.params)

        a1, z2, a2, z3, h = self.feed_forward(self.X, theta1, theta2)  
        y_pred = np.array(np.argmax(h, axis=1))

        correct = 0
        for x in range(examples):
            if self.Y[x, y_pred[x]] == 1:
                correct +=1

        accuracy = (correct / float(examples))
        self.writer.write(type +' accuracy = {0}%'.format(accuracy * 100))

    def predict(self, x):
        """
        predict given a row example
        
        Arguments:
            x {np.array} -- the feature row used to predict and output

        Returns:
            np.array -- the prediction
        """

        theta1, theta2 = self.reshape_theta(self.params)
        _,_,_,_,h = self.feed_forward(x, theta1, theta2)
        return np.array(np.argmax(h, axis=1))[0,0]

    def test(self, step=10): 
        """
        run a diagnostic check on the given data set and expected output. This method plots the the margin of prediction
        error against the increase in size of training examples. This can be useful to determine what is going wrong 
        with your hypothesis, i.e. whether it is underfitting or overfitting the training set.
        
        Arguments:
            X {[type]} -- The input set
            Y {[type]} -- The expected output
        
        Keyword Arguments:
            step {number} -- The size of step taken in to increase the dataset (default: {10})
        """
        # split into 6/2/2 ratio train/cv/test
        x_train, x_cross_validation, x_test = split(self.X)
        y_train, y_cross_validation, y_test = split(self.Y)

        error_train = []
        error_val = []
        amount = 0
        i = 1
        while i < len(x_train):
            self.writer.write("running at index %s of %s" % (i, len(x_train)))
            params = self.generate_params()
            current_input = x_train[0:i, :] 
            current_output = y_train[0:i, :] 

            fmin = minimize(fun=self.fit, x0=params, args=(current_input, current_output, False),  
                            method='TNC', jac=True, options={'maxiter': self.maxiter})
            train_cost, _= self.fit(fmin.x, current_input, current_output, False)
            val_cost, _ = self.fit(fmin.x, x_cross_validation, y_cross_validation, False)

            error_train.append(train_cost)
            error_val.append(val_cost)

            amount += 1
            i = amount * step

        plt.plot(error_train)
        plt.plot(error_val)

        plt.legend(['train', 'validation'], loc='upper left')
        plt.ylabel("error")
        
        plt.xlabel("Iteration")
        plt.show()

def split(input):
    """[summary]
    
    [description]
    
    Arguments:
        input {np.ndarray} -- The input set
    
    Returns:
        train_set {np.ndarray}
        cross_set {np.ndarray}
        test_set {np.ndarray}
    """

    length = len(input)
    unit = length/10

    train = int(round(unit*6, 0))
    cross_test = int(round(unit*2, 0))

    train_set = input[0:train, :]
    cross_set = input[train:train+cross_test, :]
    test_set = input[train+cross_test: length, :]

    return train_set, cross_set, test_set