from RBF import RBF_network
import numpy as np
import matplotlib.pyplot as plt

weights_rbf = np.array([[0], [np.pi / 2], [np.pi / 4], [3 * np.pi / 4], [np.pi * 5 / 4], [7 * np.pi / 4], [2*np.pi]])
#weights_rbf = np.array([[3 * np.pi / 4], [np.pi * 5 / 4], [7 * np.pi / 4]])

#weights_rbf = np.array([[x] for x in np.arange(0, 2*np.pi, 0.1)])

sigmas = np.array([np.pi / 4 for _ in range(len(weights_rbf))])

model = RBF_network(weights_rbf, sigmas)

X = np.arange(0, 2*np.pi, 0.1).reshape((1, -1)) # Training domain
test_x = np.arange(0.05, 2*np.pi, 0.1).reshape((1, -1))
sin_2x = np.sin(2*X)
sin_2x = sin_2x + np.random.normal(scale=0.1,size=sin_2x.shape)# Add noise
square_x = np.array([1 if np.sin(2*x) >= 0 else -1 for x in X[0]]).reshape((1, -1))
sin_2x_test = np.sin(2*test_x)

errors = model.fit_data_sequential(X, sin_2x, epochs=800, lr=.8)

predictions = model.predict(test_x)

print ('absolute residual error: ', str(np.sum(np.abs(predictions - sin_2x_test)) / predictions.shape[1]))

model.fit_data(X, sin_2x)
predictions = model.predict(test_x)

print ('absolute residual error: ', str(np.sum(np.abs(predictions - sin_2x_test)) / predictions.shape[1]))

plt.scatter(X[0], sin_2x[0], color='red')
plt.plot(test_x[0], predictions[0], color='blue')
plt.show()
plt.plot(errors)
plt.show()