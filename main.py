import neuralnet
import numpy as np
import click
import writer

writer = writer.Writer()

@click.group()
def cli():
    pass

@click.command(options_metavar='<options>')
@click.option("--lam", type=click.FLOAT, default=1, help="The regularization amount [default 1]")
@click.option("--maxiter", default=250, type=click.INT, help="The maximum iterations for chosen to minimise the cost function [default 250]")
@click.option("--output", type=click.File('rb'), help="A file path to save the minimised parameters to")
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

    nn = neuralnet.NeuralNet(X=X, Y=Y, writer=writer, output=output, lam=lam, maxiter=maxiter)
    nn.train(verbose=verbose, save=output)

@click.command(options_metavar='<options>')
@click.argument("x", type=click.File('rb'))
@click.argument("params", type=click.File('rb'))
@click.option("--normalize", type=click.BOOL, help="Perform normalization on the training set [default true]")
def predict(x, params, nomalize=True):
    """Predict an output for a given row of data"""

    x = np.loadtxt(open(x,"rb"), delimiter=",",skiprows=1, dtype="float")
    nn = nn.NeuralNet(X=X, Y=None, writer=writer, output=output, lam=lam, maxiter=maxiter)

    nn.set_params(np.loadtxt(open(params,"rb"), delimiter=",",skiprows=1, dtype="float"))
    print nn.predict(x)

@click.command(options_metavar='<options>')
def test():
    """Test the given network on the train and validation sets"""
    return "test"


cli.add_command(test)
cli.add_command(predict)
cli.add_command(train)

if __name__ == '__main__':
    cli()