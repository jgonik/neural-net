# Neural Network Coding Challenge

## Description

Implement and train a neural network from scratch in Python without using Tensorflow
or PyTorch. The network is designed for the MNIST dataset. It optimizes the weights
and offsets of the network using SGD (Stochastic Gradient Descent). Depending on the
number of iterations and other network parameters, it can achieve 97-98% accuracy on
the test dataset.

## Implementation

An object called 'NN' represent the neural network model and its parameters, including
the weights and offsets of the first and second layers, the input size, the hidden
layer size, the output size, and the activation function. The weights and offsets are
initially random, and then stochastic gradient descent is used to train the network
and optimize its parameters. The step size is set to 0.01.

## Instructions for Running

To train a neural network specified by the provided configuration file, execute the
following command in terminal:

**python3 ./Neural_Network.py <"configfilename.cfg"> <num_iterations>**

## Weights and Offsets

The first layer's weights and offsets are saved in "first_layer.txt".

The second layer's weights and offsets are saved in "second_layer.txt".

## Package

The package is uploaded on Test PyPi. The url is https://test.pypi.org/project/neural-net-challenge/.
