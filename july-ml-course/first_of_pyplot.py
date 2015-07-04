import matplotlib.pyplot as plt
import math


if __name__ == "__main__":
	# prepare data 
	x = [float(i) / 100 for i in range(1, 300)]
	y = [math.log(i) for i in x]
	# plot
	plt.plot(x, y, 'r-', linewidth=3, label='log curve')

	# prepare data
	p1 = 20
	p2 = 175
	a = [x[p1], x[p2]]
	b = [y[p1], y[p2]]
	# plot
	plt.plot(a, b, 'g-', linewidth=2)
	plt.plot(a, b, 'b*', markersize=15, alpha=0.5)

	# config 
	plt.legend(loc='upper left')
	plt.grid(True)
	plt.xlabel('x')
	plt.ylabel('log(x)')

	# show
	plt.show()
