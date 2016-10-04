import neuralnet as nn
import numpy as np
import click

@click.group()
def cli():
    pass

@click.command(options_metavar='<options>')
@click.option("--lambda", type=click.FLOAT, help="The regularization amount [default 1]")
@click.option("--maxiter", type=click.INT, help="The maximum iterations for chosen to minimise the cost function [default 250]")
@click.option("--output", type=click.File('rb'), help="A file path to save the minimised parameters to")
@click.option("--normalize", type=click.BOOL, help="Perform normalization on the training set [default true]")
@click.option("--verbose", type=click.BOOL, help="Output the training progress [default true]")
@click.argument("X", type=click.File('rb'))
@click.argument("Y", type=click.File('rb'))
def train(X, Y, output, lam=1, maxiter=250, normalize=True, verbose=True):
    """Train a neural network with the given X and Y parameters.
    
    Arguments:\n
        [X] must be a file path to a CSV which holds your training data\n
        [Y] must be a file path to a CSV which holds your expected outputs for the training examples

    (neural cli will ommit the first row for column headers for both CSVs)
    """
    X = np.loadtxt(open(X,"rb"), delimiter=",",skiprows=1, dtype="float")
    Y = np.loadtxt(open(Y,"rb"), delimiter=",",skiprows=1, dtype="float")

    nn = nn.NeuralNet(X=X, Y=Y, output=output, lam=lam, maxiter=maxiter)
    nn.train(verbose=False, save=output)

cli.add_command(train)

if __name__ == '__main__':
    cli()