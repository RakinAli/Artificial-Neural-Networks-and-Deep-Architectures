from RBF import RBF_network
import numpy as np
import matplotlib.pyplot as plt

weights_rbf = np.array([[0], [np.pi / 4], [np.pi / 2], [3*np.pi/4], [np.pi], [5 * np.pi / 4], [3 * np.pi / 2], [7 * np.pi / 4]])
sigmas = np.array([2, 2, 2, 2, 2, 2, 2, 2])

model = RBF_network(weights_rbf, sigmas)

X = np.arange(0, 2*np.pi, 0.1).reshape((1, -1)) # Training domain
sin_2x = np.sin(2*X)
square_x = np.array([1 if np.sin(2*x) >= 0 else -1 for x in X[0]]).reshape((1, -1))

model.fit_data(X, square_x)

predictions = model.predict(X)

print (predictions.shape)

plt.plot(X[0], square_x[0], color='red')
plt.plot(X[0], predictions[0], color='blue')
plt.show()
