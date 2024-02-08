from RBF import RBF_network
import numpy as np
import matplotlib.pyplot as plt
from double_layer_regression_nn import DoubleLayerNN

weights_rbf = np.array([[0], [np.pi / 2], [np.pi / 4], [3 * np.pi / 4], [np.pi * 5 / 4], [7 * np.pi / 4], [2*np.pi]])
weights_rbf = np.array([[np.random.uniform(0, np.pi*2)] for _ in range(7)])

#weights_rbf = weights_rbf + np.ones(shape=weights_rbf.shape)

#weights_rbf = np.array([[x] for x in np.arange(0, 2*np.pi, 0.1)])

sigmas = np.array([np.pi / 4 for _ in range(len(weights_rbf))])

model = RBF_network(weights_rbf, sigmas)

X = np.arange(0, 2*np.pi, 0.1).reshape((1, -1)) # Training domain
test_x = np.arange(0.05, 2*np.pi, 0.1).reshape((1, -1))
sin_2x = np.sin(2*X)
sin_2x = sin_2x + np.random.normal(scale=0.1,size=sin_2x.shape) # Add noise
square_x = np.array([1 if np.sin(2*x) >= 0 else -1 for x in X[0]]).reshape((1, -1))
sin_2x_test = np.sin(2*test_x)

model.fit_data(X, sin_2x)

predictions = model.predict(test_x)

model2 = DoubleLayerNN(input_nodes=1, nr_of_hidden_nodes=5000, learning_rate=0.00005)
errors = model2.fit_data(X, sin_2x, iterations=1000)
predictions2 = model2.predict(test_x)

print ('absolute residual error for model 1: ', str(np.sum(np.abs(predictions - sin_2x_test)) / predictions.shape[1]))
print ('absolute residual error for model 1: ', str(np.sum(np.abs(predictions2 - sin_2x_test)) / predictions.shape[1]))


plt.scatter(X[0], sin_2x[0], color='red')
plt.plot(test_x[0], predictions2[0], color='blue', label='Trained with batch learning')
plt.plot(test_x[0], predictions[0], color='green', label='Trained with sequential learning')
plt.legend()
plt.title('The two learning approaches with sigma=pi/4 with random weights')
plt.show()


plt.plot(errors, color='blue')
plt.xlabel('Iteration')
plt.ylabel('Mean square error')
plt.title('Convergance for the delta approach, lr=.1 random weights')
plt.show()
