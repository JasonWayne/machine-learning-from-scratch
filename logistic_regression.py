import math
import matplotlib.pyplot as plt


def sigmoid(z):
	return 1 / (1 + math.exp(-z))


def f(x, theta):
	return sigmoid(sum(map(lambda x, theta: x * theta, x, theta)))


def logistic_regression(X, y, alpha=0.001, iter_nums=3000):
	theta = [0 for _ in range(len(X[0]))]
	iter_count = 0
	loss_history = []
	while True:
		batch_loss = 0
		for i in range(len(X)):
			iter_count += 1
			if iter_count == iter_nums:
				return theta, loss_history
			# forward
			h = f(X[i], theta)
			loss = - y[i] * math.log(h) - (1 - y[i]) * math.log(1 - h)
			batch_loss += loss
			# backward
			delta = 0.00000001
			theta_with_delta = [theta[0] + delta] + theta[1:]
			h2 = f(X[i], theta_with_delta)
			loss2 = - y[i] * math.log(h2) - (1 - y[i]) * math.log(1 - h2)
			gradient_theta0 = (loss2 - loss) / delta
			# gradient = map(lambda theta, x: (h - y[i]) * x, theta, X[i])

			gradient = map(lambda theta, x: (h - y[i]) * x, theta, X[i])
			theta = map(lambda theta, grad: theta - alpha * grad, theta, gradient)
			print gradient, gradient_theta0, gradient[0] / gradient_theta0

		batch_loss /= len(X)
		loss_history.append(batch_loss)


def test():
	# average score, teacher's comment, classmates' comment
	X = [
			[1, 8, 5, 4],
			[1, 10, 5, 3],
			[1, 6, 3, 3],
			[1, 7, 3, 3],
			[1, 9, 3, 4],
			[1, 10, 5, 5],
			[1, 8, 4, 4],
			[1, 8.5, 3, 4],
			[1, 5.4, 2, 4],
			[1, 6.5, 3, 3]
		]
	# if the student are good student
	y = [1, 1, 0, 0, 1, 1, 1, 1, 0, 0]
	theta, losses = logistic_regression(X, y)
	print theta
	plt.plot(losses)
	plt.xlabel("num_iterations")
	plt.ylabel("loss")
	plt.show()


if __name__ == "__main__":
	test()
