

from random import seed
from random import random


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):

	"""
	Creates a neural network.
	3 layers.
	Each layer is an array of neurons (dictionaries) 
	All weights initialised as small random numbers in range (0, 1)
	"""
	network = list()

	# Input layer is a row from training data set.

	# Hidden layer has n_hidden neurons.
	# Each neuron has n_inputs + 1 (bias), weights.
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)

	# Output layer that connects to the hidden layer has n_outputs neurons
	# Each neuron has n_hidden + 1 weights
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)

	return network

seed(1)

network = initialize_network(2, 1, 2)

for layer in network:
	print(layer)


# Forward propagate 
