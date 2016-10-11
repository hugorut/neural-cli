from __future__ import absolute_import, division, print_function
from .neuralnet import NeuralNet
import numpy as np
import click
from .writer import Writer

writer = Writer()

@click.group()
def main():
    pass

@click.command(options_metavar='<options>')
@click.option("--lam", type=click.FLOAT, default=1, help="The regularization amount [default 1]")
@click.option("--maxiter", default=250, type=click.INT, help="The maximum iterations for chosen to minimise the cost function [default 250]")
@click.option("--output", type=click.STRING, help="A file path to save the minimised parameters to")
@click.option("--normalize", default=True, type=click.BOOL, help="Perform normalization on the training set [default true]")
@click.option("--verbose", default=True, type=click.BOOL, help="Output the training progress [default true]")
@click.argument("X", type=click.File('rb'))
@click.argument("Y", type=click.File('rb'))
def train(x, y, output, lam, maxiter, normalize, verbose):
    """Train a neural network with the given X and Y parameters.
    
    Arguments:\n
        [X] must be a file path to a CSV which holds your training data\n
        [Y] must be a file path to a CSV which holds your expected outputs for the training examples

    (neural cli will ommit the first row for column headers for both CSVs)
    """
    X = np.loadtxt(x, delimiter=",",skiprows=1, dtype="float")
    Y = np.loadtxt(y, delimiter=",",skiprows=1, dtype="float")

    nn = NeuralNet(X=X, Y=Y, writer=writer, output=output, lam=lam, maxiter=maxiter, norm=normalize)
    nn.train(verbose=verbose, save=output)
    nn.accuracy()

@click.command(options_metavar='<options>')
@click.argument("x", type=click.File('rb'))
@click.argument("labels", type=click.INT)
@click.argument("params", type=click.File('rb'))
@click.option("--sizeh", type=click.INT, help="the size of the hidden layer that the parameters were trained on")
@click.option("--normalize", type=click.BOOL, help="Perform normalization on the training set [default false]")
def predict(x, labels, params, sizeh, normalize):
    """
    predict an output with a given row. Prints the index of the prediction of the output row.
    
    Arguments:\n
        [x] the file that holds the 1 * n row example that should be predicted  \n
        [labels] the size of the output layer that the parameters were trained on \n
        [params] the file that holds a 1 * n rolled parameter vector \n
    """

    x = np.loadtxt(x, delimiter=",",skiprows=0, dtype="float")
    x = x[np.newaxis];
    nn = NeuralNet(X=None, Y=None, writer=writer, norm=normalize)

    input_size = np.shape(x)[1]
    hidden_size = input_size
    if sizeh:
        hidden_size = sizeh

    nn.set_hidden_size(input_size)
    nn.set_hidden_size(hidden_size)
    nn.set_num_labels(labels)
    nn.set_params(np.loadtxt(params, delimiter=",",skiprows=0, dtype="float"))
    writer.write(nn.predict(x))

@click.command(options_metavar='<options>')
@click.argument("X", type=click.File('rb'))
@click.argument("Y", type=click.File('rb'))
@click.option("--lam", type=click.FLOAT, default=1, help="The regularization amount [default 1]")
@click.option("--maxiter", default=250, type=click.INT, help="The maximum iterations for chosen to minimise the cost function [default 250]")
@click.option("--normalize", default=True, type=click.BOOL, help="Perform normalization on the training set [default true]")
@click.option("--step", default=10, type=click.INT, help="The increments that the training will increase the set by [default 10]")
def test(x, y, lam, maxiter, normalize, step):
    """Test the given network on the train and validation sets
    
    Arguments:\n
        [X] must be a file path to a CSV which holds your training data\n
        [Y] must be a file path to a CSV which holds your expected outputs for the training examples

    (neural cli will ommit the first row for column headers for both CSVs)
    """

    X = np.loadtxt(x, delimiter=",",skiprows=1, dtype="float")
    Y = np.loadtxt(y, delimiter=",",skiprows=1, dtype="float")

    nn = NeuralNet(X=X, Y=Y, writer=writer, lam=lam, maxiter=maxiter, norm=normalize)
    nn.test(step)


main.add_command(test)
main.add_command(predict)
main.add_command(train)