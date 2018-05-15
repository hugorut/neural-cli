# Command Line Neural Network

Neuralcli provides a simple command line interface to a python implementation of a simple classification neural network. Neuralcli allows a quick way and easy to get instant feedback on a hypothesis or to play around with one of the most popular concepts in machine learning today.

## Installation

Installation of neuralcli is provided through pip, just run:
```
pip install neuralcli
```

If you don't have some of the libraries used, such as `numpy` or `skitlearn` the install make take some time as pip installs all the dependencies. After pip finishes the install run the following `command -v neuralcli` to check that the executable has been successfully added. 

**Troubleshooting**

When you run `neuralcli` for the first time you may get an output similar to below
```
/usr/local/lib/python2.7/site-packages/matplotlib/font_manager.py:273: UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment.
warnings.warn('Matplotlib is building the font cache using fc-list. This may take a moment.')
```
This is just a warning from matplotlib, and will be removed the next time you run the command.

Additionally matlibplot may throw another error that will produce an output similar to:
```
**RuntimeError**: Python is not installed as a framework
```

To fix this issue follow the steps outlined [here](http://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python)

## Use

Neuralcli comes bundled with three main commands.

### Train

The train command takes a set of input features along with their expected outputs and performs backpropogation to learn the weights for a neural network. These weights can be saved to an output file to use for classification prediction later. The command takes the following.

**parameters:**

| name | type | description                                                                      | example        |
|------|------|----------------------------------------------------------------------------------|----------------|
| X    | file | a file path to a CSV which holds your training data                              | ./train.csv    |
| Y    | file | a file path to a CSV which holds your expected outputs for the training examples | ./expected.csv |

**flags:**

| name        | type   | description                                                     | default | example      |
|-------------|--------|-----------------------------------------------------------------|---------|--------------|
| --lam       | float  | The regularization amount                                       | 1       | 0.07         |
| --maxiter   | int    | The maximum iterations for chosen to minimise the cost function | 250     | 30           |
| --output    | string | A file path to save the minimised parameters to                 | nil     | ./output.csv |
| --normalize | bool   | Perform normalization on the training set                       | true    | false        |
| --verbose   | bool   | Output the training progress                                    | true    | false        |

**example:**

```
$ neuralcli train ./X.csv ./Y.csv --output=./weights.csv --normalize=true
```

Once you run the train command the neural network will intialize and begin to learn the weights, you should see an output similar to bellow if the `--verbose` flag is set to true.

![](http://i.imgur.com/EqPJD2s.gif)

### Predict

The prediction command takes a set of learned weights and a given input to predict a an ouput. The learned weights are loaded into the neural network by providing an file which holds them in a rolled 1 * n vector shape. In order for the predict command to work correctly these parameters need to be unrolled and therefore you need to provide the sizes of the input layer, hidden layer, and output labels that you wish to unroll the 

**parameters:**

| name   | type | description                                                                        | example     |
|--------|------|------------------------------------------------------------------------------------|-------------|
| x      | file | the file that holds the 1 * n row example that should be predicted                 | ./input.csv |
| params | file | The file that holds a 1 * n rolled parameter vector (saved from the train command) | ./ouput.csv |
| labels | int  | The size of the output layer that the parameters were trained on                   | 3           |

**flags:**

| name        | type   | description                                                     | default | example      |
|-------------|--------|-----------------------------------------------------------------|---------|--------------|
| --normalize | bool   | Perform normalization on the training set                       | true    | false        |
| --sizeh     | int    | The size of the hidden layer if it differs from the input layer | nil     | 8            |

**example:**

```
$ neuralcli predict ./x.csv 3 ./params.csv 
```

Neuralcli will now print a prediction in INT form, corresponding to the index of you output labels.
e.g. `0` will correspond to you first classification label. 

### Test

The test command gives some primitive feedback about the correctness of your hypothesis by running a diagnostic check on the given data set and expected output. This method plots the the margin of prediction error against the increase in size of training examples. This can be useful to determine what is going wrong with your hypothesis, i.e. whether it is underfitting or overfitting the training set.

**parameters:**

| name | type | description                                                                      | example        |
|------|------|----------------------------------------------------------------------------------|----------------|
| X    | file | a file path to a CSV which holds your training data                              | ./train.csv    |
| Y    | file | a file path to a CSV which holds your expected outputs for the training examples | ./expected.csv |

**flags:**

| name        | type   | description                                                     | default | example      |
|-------------|--------|-----------------------------------------------------------------|---------|--------------|
| --lam       | float  | The regularization amount                                       | 1       | 0.07         |
| --maxiter   | int    | The maximum iterations for chosen to minimise the cost function | 250     | 30           |
| --normalize | bool   | Perform normalization on the training set                       | true    | false        |
| --verbose   | bool   | Output the training progress                                    | true    | false        |
| --step      | int    | The increments that the training will increase the set by       | 10      | 100          |

**example:**

```
$ neuralcli train ./X.csv ./Y.csv --step=50 --normalize=true
```

Neural cli will then run the test sequence printing its progress as it increases the size of the training set.

![](http://i.imgur.com/TFlhHJN.gif)

After this runs it will then print a plot of the hypothesis error against the size of training set the weights where learned on. Below is an example graph plotted from the iris dataset.

![](http://i.imgur.com/o3ZTQxY.png)
