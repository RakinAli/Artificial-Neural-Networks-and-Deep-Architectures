from RBF import RBF_network
import numpy as np
from SOM import SOM

#import matplotlib.pyplot as plt
#
#weights_rbf = np.array([[0], [np.pi / 4], [np.pi / 2], [3*np.pi/4], [np.pi], [5 * np.pi / 4], [3 * np.pi / 2], [7 * np.pi / 4]])
#sigmas = np.array([2, 2, 2, 2, 2, 2, 2, 2])
#
#model = RBF_network(weights_rbf, sigmas)
#
#X = np.arange(0, 2*np.pi, 0.1).reshape((1, -1)) # Training domain
#sin_2x = np.sin(2*X)
#square_x = np.array([1 if np.sin(2*x) >= 0 else -1 for x in X[0]]).reshape((1, -1))
#
#model.fit_data(X, square_x)
#
#predictions = model.predict(X)
#
#print (predictions.shape)
#
#plt.plot(X[0], square_x[0], color='red')
#plt.plot(X[0], predictions[0], color='blue')
#plt.show()
#

a = np.array([[1,2,3],[4,5,6]])
b = np.array([1,2, 3])
print (a - b)
print(np.linalg.norm(a-b, axis=1))

dataset = np.array([[1,2,3,3,3,4,4,2,2,2,3,4,6], [4,5,6,4,3,43,4,3,34,2,2,21,1]])
som = SOM(4, 2)
som.fit_data(dataset)