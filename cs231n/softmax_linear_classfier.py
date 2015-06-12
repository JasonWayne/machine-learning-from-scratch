'''
http://cs231n.github.io/neural-networks-case-study/
'''

import numpy as np
import matplotlib.pyplot as plt


N = 100		# Number of points per class
D = 2		# Dimensionality
K = 3		# number of classes

X = np.zeros((N * K, D))
# this work fine but I belive y should be np.zeros((1, N * K)) specifically
y = np.zeros(N * K, dtype='uint8')

for j in xrange(K):
	ix = range(N * j, N * (j + 1))
	r = np.linspace(0.0, 1, N)		# radius
	t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2		# theta
	# np.c_?
	X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
	y[ix] = j
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

# initialize parameters randomly
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))

num_examples = X.shape[0]
loss_history = []


step_size = 1e-2
reg = 1e-3
for i in xrange(10000):
	scores = np.dot(X, W) + b
	exp_score = np.exp(scores)
	probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)

	# probs[range(num_examples), y]?
	correct_logprobs = - np.log(probs[range(num_examples), y])

	data_loss = np.sum(correct_logprobs) / num_examples
	reg_loss = 0.5 * reg * np.sum(W * W)
	loss = data_loss + reg_loss
	loss_history.append(loss)

	# dl / df, and we can use dldf or df to represent
	df = probs
	df[range(num_examples), y] -= 1
	df /= num_examples

	# dw mean dl / dw
	# dw = dldf * dfdw and dldf has been computed as df
	dW = np.dot(X.T, df)
	db = np.sum(df, axis=0, keepdims=True)

	W += - step_size * dW
	b += - step_size * db

plt.plot(loss_history)
plt.xlabel('num_iterations')
plt.ylabel('loss')
plt.show()


predicted_class = np.argmax(scores, axis=1)
# get an accuracy of 0% using probs, why?
# predicted_class = np.argmax(probs, axis=1)
print 'training accuracy: %.2f' % np.mean(predicted_class == y)

