from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder  
import numpy as np
from scipy.io import loadmat 
import csv
import sys
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

class NeuralNet():
    """The main neural network class for training"""

    def __init__(self, X, Y, output="./params", lam=1, maxiter=250):
        X = np.matrix(X)
        Y = np.matrix(Y)
        X = normalize(X)

        self.X = X
        self.Y = Y

        self.num_labels = np.shape(Y)[1]
        self.input_size = np.shape(X)[1]
        self.hidden_size = np.shape(X)[1]  

        self.lam = lam
        self.output = output
        self.params = self.generate_params()

    """
    [train]
    minimize a cost function defined under backpropogation

    @param bool {verbose} -- should the backpropgation print progress
    @param bool {save} -- should output of parameters be saved to a file
    """
    def train(self, verbose=False, save=True):
        fmin = minimize(fun=self.fit, x0=self.params, args=(self.X, self.Y, verbose),  
                        method='TNC', jac=True, options={'maxiter': maxiter})

        if save:
            writer = csv.writer(open(self.output, 'w'))
            writer.writerow(fmin.x)

        return fmin.x

    """
    [generate_params]
    generate a random sequence of weights for the parameters of the neural network

    @return np.ndarray
    """
    def generate_params(self):
        return (np.random.random(size=self.hidden_size * (self.input_size + 1) + self.num_labels * (self.hidden_size + 1)) - 0.5) * 0.25

    """
    [load_params]
    load parameters from a csv file
    
    @param string {name} -- the location of the file
    @return np.ndarray
    """
    def load_params(self, name):
        return np.loadtxt(open(name,"rb"), delimiter=",",skiprows=0, dtype="float")

    """
    [sigmoid]
    comute the sigmoid activation function

    @return mixed
    """
    def sigmoid(self, z):  
        return 1 / (1 + np.exp(-z))

    """
    [sigmoid]
    derivative of the sigmoid func
    @return np.ndarray
    """
    def sigmoid_gradient(self, z):  
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    """
    [reshape_theta]
    reshape the 1 * n parameter vector into the correct shape for the first and second layers

    @param np.ndarray {params} -- a vector of weights
    @return array<np.ndarray>
    """
    def reshape_theta(self, params):
        theta1 = np.matrix(np.reshape(params[:self.hidden_size * (self.input_size + 1)], (self.hidden_size, (self.input_size + 1))))
        theta2 = np.matrix(np.reshape(params[self.hidden_size * (self.input_size + 1):], (self.num_labels, (self.hidden_size + 1))))

        return theta1, theta2

    """
    [feed_forward]
    run forward propgation using a value of X

    @param np.matrix {X} -- Input set
    @param np.ndarray {theta1} -- The first layer weights
    @param np.matrix {theta2} -- The second layer weights

    @return np.ndarray {a1} -- input
    @return np.ndarray {z2} -- sigmoid of first layer
    @return np.ndarray {a2} -- activation of second layer
    @return np.ndarray {z3} -- sigmoid of 
    @return np.ndarray {h}
    """
    def feed_forward(self, X, theta1, theta2):  
        m = X.shape[0]

        a1 = np.insert(X, 0, values=np.ones(m), axis=1)

        z2 = a1 * theta1.T
        a2 = np.insert(self.sigmoid(z2), 0, values=np.ones(m), axis=1)
        z3 = a2 * theta2.T
        h = self.sigmoid(z3)

        return a1, z2, a2, z3, h

    """
    [fit]
    main function to run a single pass on the nn. First run forward propgation to get the error of output given some
    parameters and then perfom backpropgation to work out the gradient of the function using the given weights.

    @param np.ndarray {params} -- weight layer parameters
    @param np.matix {X} -- Input matrix
    @param np.matrix {y} -- Expected output matrix

    @return float64 {J} -- the margin of error with the given weights
    @return np.ndarray {grad} -- the matrix of gradients for the given weights
    """
    def fit(self, params, X, y, output=True):  
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
            print J

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

    """
    [get_cost]
    get the cost of prediction

    @param np.ndarray {y} -- The expected output
    @param np.matix {h} -- The perdiction array
    @param float {minval} -- The minimum value that h projection can be [so no log by zero errors]

    @return float64 {cost} -- the margin of error with the given weights
    """
    def get_cost(self, y, h, minval=0.0000000001):
        first_term = np.multiply(-y, np.log(h.clip(minval)))
        second_term = np.multiply((1 - y), np.log(1 - h.clip(minval)))

        return np.sum(first_term - second_term)

    """
    [training_acc]
    get the accuracy of the learned parameters on the training set

    @param string|None {from_file} -- Whether the parameters should be loaded from file

    @return void
    """
    def training_acc(self, from_file=None):
        params = self.params
        examples = len(self.Y)
        if from_file:
            params = self.load_params(from_file)

        theta1, theta2 = self.reshape_theta(params)

        a1, z2, a2, z3, h = self.feed_forward(self.X, theta1, theta2)  
        y_pred = np.array(np.argmax(h, axis=1))
        correct = 0
        for x in xrange(examples):
            if self.Y[x, y_pred[x]] == 1:
                correct +=1

        accuracy = (correct / examples)  
        print 'train accuracy = {0}%'.format(accuracy * 100)

    """
    [test_acc]
    get the accuracy of the learned parameters on the test set

    @param np.ndarray {X} -- The test set
    @param np.ndarray {Y} -- The test set expected output
    @param string|None {from_file} -- Whether the parameters should be loaded from file
    
    @return void
    """
    def test_acc(self, X, Y, from_file):
        X = np.matrix(X)
        Y = np.matrix(Y)
        X = normalize(X)

        params = self.params
        if from_file:
            params = self.load_params(from_file)

        theta1, theta2 = self.reshape_theta(params)

        a1, z2, a2, z3, h = self.feed_forward(X, theta1, theta2)  
        y_pred = np.array(np.argmax(h, axis=1))
        correct = 0
        for x in xrange(0, len(Y)):
            if Y[x, y_pred[x]] == 1:
                correct +=1

        accuracy = (correct / len(Y))  
        print 'test accuracy = {0}%'.format(accuracy * 100)

    """
    [split]
    split a given set of examples, break this down into a train|validation|test set.

    @param np.ndarray {input} -- The input set
    
    @return np.ndarray {train_set}
    @return np.ndarray {cross_set}
    @return np.ndarray {test_set}
    """
    def split(self, input):
        length = len(input)
        unit = length/10

        train = int(round(unit*6, 0))
        cross_test = int(round(unit*2, 0))

        train_set = input[0:train, :]
        cross_set = input[train:train+cross_test, :]
        test_set = input[train+cross_test: length, :]

        return train_set, cross_set, test_set

    """
    [test]
    run a diagnostic check on the given data set and expected output. This method plots the the margin of prediction
    error against the increase in size of training examples. This can be useful to determine what is going wrong 
    with your hypothesis, i.e. whether it is underfitting or overfitting the training set.

    @param np.ndarray {X} -- The input set
    @param np.ndarray {Y} -- The expected output
    @param np.ndarray {step} -- The size of step taken in to increase the dataset
    
    @return void
    """
    def test(self, X, Y, step=10): 
        X = normalize(X)
        X = np.matrix(X)
        Y = np.matrix(Y)

        # split into 6/2/2 ratio train/cv/test
        x_train, x_cross_validation, x_test = split(X)
        y_train, y_cross_validation, y_test = split(Y)

        error_train = []
        error_val = []
        amount = 0
        i = 1
        while i < len(x_train):
            print "running at index %s of %s" % (i, len(x_train)) 
            params = self.generate_params()
            current_input = x_train[0:i, :] 
            current_output = y_train[0:i, :] 

            fmin = minimize(fun=self.fit, x0=params, args=(X, y, False),  
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
        