

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
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

	for i in range(n_iter):
		# calculate gradient
		gradient = derivative(solution)
		# take a step
		solution = solution - step_size * gradient
		# evaluate candidate point
		solution_eval = objective(solution)
		# report progress
		print('>%d f(%s) = %.5f' % (i, solution, solution_eval))

	return [solution, solution_eval]

	