import matplotlib.pyplot as plt


def f(x, theta):
	return sum(map(lambda x, theta: x * theta, x, theta))


def loss_func(h, y):
	return (h - y) ** 2


def sgd(X, y, theta=None, alpha=0.001, num_iters=1000):
	if theta is None:
		theta = [0 for _ in range(len(X[0]))]
	# we don't record every loss we've calculated
	# but the average loss of all the sample
	loss_history = []
	iter_count = 0
	while True:
		batch_loss = 0
		for i, x in enumerate(X):
			iter_count += 1
			if iter_count == num_iters:
				return theta, loss_history
			# forward
			h = f(x, theta)
			loss = loss_func(h, y[i])
			batch_loss += loss

			# this is for debug, to check if the analytic gradient descent is right
			delta = 0.001
			theta_with_delta = theta[:-1] + [theta[-1] + delta]
			h_delta = f(x, theta_with_delta)
			loss_delta = (h_delta - y[i]) ** 2
			gradient_numeric = (loss_delta - loss) / delta
			gradient_analytic = 2 * (h - y[i]) * x[-1]
			print gradient_analytic, gradient_numeric, \
				gradient_numeric / gradient_analytic, theta[1]

			# backward
			theta = map(lambda theta, x: theta - alpha * (h - y[i]) * x, theta, x)
		loss_history.append(batch_loss / len(X))


if __name__ == "__main__":
	X = [
			[1, 1, -1],
			[1, 2, -1.8],
			[1, 3.2, -2.8],
			[1, 4, -4.5],
			[1, 5.4, -5],
			[1, 5.6, -6],
			[1, 6.2, -7],
			[1, 7.7, -8.2],
			[1, 8.8, -9.2]
		]
	y = [1, 1.5, 2.7, 3.3, 5.3, 6.6, 7.1, 8.2, 9.0, 10.0]
	theta, loss_history = sgd(X, y, theta=None, alpha=0.001, num_iters=5000)
	print theta
	plt.plot(loss_history[10:])
	plt.xlabel("iteration count")
	plt.ylabel("loss")
	plt.show()
