

from random import seed
from random import random
from math import exp



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




def activate(weights, inputs):

	"""
	Neuron activation
	Calculate neuron activation for an input
	"""

	# bias 
	activation = weights[-1]

	# weighted sum of inputs 
	for i in range(len(weights)-1):

		activation += weights[i] * inputs[i]

	return activation




# Forward propagate 
def transfer(activation):

	"""
	Transfer neuron activation
	Maps an activation funtion to sigmoid / logistic function (S-shaped curve between 0 and 1)
	output = 1 / (1 + e^(-activation))
	"""

	return 1.0 / (1.0 + exp(-activation))



# Forward propagate input to a network output
def forward_propagate(network, row):

	inputs = row

	for layer in network:

		new_inputs = []

		# for each neuron in the layer 
		for neuron in layer:

			# find weighted sum of inputs 
			activation = activate(neuron['weights'], inputs) 
			print(f'activation = {activation}')

			# apply scaling / transfer function to weighted sum
			neuron['output'] = transfer(activation)           
			print(neuron)

			# add to inputs to be fed to each neuron in next layer 
			new_inputs.append(neuron['output'])               

		inputs = new_inputs

	return inputs



# Test forward propagation
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		   [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]

row = [1, 0, None] # input pattern 1, 0

output = forward_propagate(network, row)
print(output)


# Back Propagate Error
# (i.e. make the weights more useful)


