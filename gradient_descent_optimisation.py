
#https://machinelearningmastery.com/gradient-descent-optimization-from-scratch/
#https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/

import numpy as np 
import matplotlib.pyplot as plt
from numpy.random import rand
from numpy.random import seed


# gradient descent algorithm
def gradient_descent(objective, derivative, bounds, n_iter, step_size):
	"""
	Inputs:
	- objective function
	- gradient functions
	- bounds on the inputs to the objective function
	- number of iterations
	- step size

	Returns
	- the solution and its evaluation at the end of the search.

	"""
	# track all solutions
	solutions, scores = list(), list()

	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	print(solution)

	for i in range(n_iter):

		# calculate gradient
		gradient = derivative(solution)

		# take a step
		solution = solution - step_size * gradient

		# evaluate candidate point
		solution_eval = objective(solution)

		# store solution
		solutions.append(solution)
		scores.append(solution_eval)

		# report progress
		print(f'>{i} gradient={gradient}, f({solution}) = {solution_eval}')

	return [solutions, scores]

def gradient_descent_with_momentum(objective, derivative, bounds, n_iter, step_size, momentum):
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	
	# keep track of the change
	change = 0.0

	# track all solutions
	solutions, scores = list(), list()
	
	# run the gradient descent
	for i in range(n_iter):
		
		# calculate gradient
		gradient = derivative(solution)
		
		# calculate update
		new_change = step_size * gradient + momentum * change
		
		# take a step
		solution = solution - new_change
		
		# save the change
		change = new_change
		
		# evaluate candidate point
		solution_eval = objective(solution)

		# store solution
		solutions.append(solution)
		scores.append(solution_eval)
		
		# report progress
		print(f'>{i} gradient={gradient}, f({solution}) = {solution_eval}')
	
	return [solutions, scores]



# objective function
def objective(x):
	return x**2.0


# derivative of objective function
def derivative(x):
	return x * 2.0

seed(4)

# define range for input
r_min, r_max = -1.0, 1.0

# sample input range uniformly at 0.1 increments
inputs = np.arange(r_min, r_max+0.1, 0.1)

# compute targets
results = objective(inputs)

# define range for input
bounds = np.asarray([[r_min, r_max ]])

# define the total iterations
n_iter = 30

# define the step size
step_size = 0.1

# # perform the gradient descent search
# solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, step_size)

# define momentum
momentum = 0.3

# perform the gradient descent search
solutions, scores = gradient_descent_with_momentum(objective, derivative, bounds, n_iter, step_size, momentum)

# create a line plot of input vs result
plt.plot(inputs, results)

# plot the solutions found
plt.plot(solutions, scores, '.-', color='red')

# show the plot
plt.show()