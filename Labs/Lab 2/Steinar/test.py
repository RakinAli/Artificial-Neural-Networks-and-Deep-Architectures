from RBF import RBF_network
import numpy as np
from SOM import SOM, SOM_grid

import matplotlib.pyplot as plt
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
#print (a - b)
#print(np.linalg.norm(a-b, axis=1))
#
dataset = np.random.uniform(size=(2, 1000))
#som = SOM(4, 2)
#plt.scatter(dataset[0], dataset[1])
#plt.scatter(som.weights[0], som.weights[1], color='red')
#for i, w in enumerate(som.weights.T):
#    plt.annotate(i, (w[0], w[1]))
#plt.show()
#som.fit_data_circular(dataset, epochs=1000)
#
#
#
#plt.scatter(dataset[0], dataset[1])
#plt.scatter(som.weights[0], som.weights[1], color='red')
#for i, w in enumerate(som.weights.T):
#    plt.annotate(i, (w[0], w[1]))
#plt.show()

model = SOM_grid(grid_size=(4, 4), dim=2)

plt.scatter(dataset[0], dataset[1])
plt.scatter(model.neurons[:,:,0], model.neurons[:,:, 1], color='red')

for i in range(model.neurons.shape[0]):
    for j in range(model.neurons.shape[1]):
        plt.annotate((i, j), (model.neurons[i, j, 0], model.neurons[i, j, 1]))

plt.show()


model.fit_data(dataset, 400)

plt.scatter(dataset[0], dataset[1])
plt.scatter(model.neurons[:,:,0], model.neurons[:,:, 1], color='red')

for i in range(model.neurons.shape[0]):
    for j in range(model.neurons.shape[1]):
        plt.annotate((i, j), (model.neurons[i, j, 0], model.neurons[i, j, 1]))

plt.show()

