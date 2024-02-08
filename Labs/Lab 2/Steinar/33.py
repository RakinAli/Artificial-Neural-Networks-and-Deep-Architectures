import numpy as np
from RBF import RBF_network
import matplotlib.pyplot as plt

X = np.arange(0, 2*np.pi, 0.1).reshape((1, -1)) # Training domain
test_x = np.arange(0.05, 2*np.pi, 0.1).reshape((1, -1))
sin_2x = np.sin(2*X)# + np.random.normal(scale=0.1, size = X.shape)
square_x = np.array([1 if np.sin(2*x) >= 0 else -1 for x in X[0]]).reshape((1, -1))
sin_2x_test = np.sin(2*test_x)

np.random.seed(2)

# competitive learning:
weights = np.random.uniform(size=6)
eta = 0.1  

for e in range(1000):
    for d in range(len(X[0])):
        point = X[0,int(np.random.uniform()*X.shape[1])]

        dist = np.abs(weights - point)

        winner = np.argmin(dist)

        weights[winner] = weights[winner] + eta * (point - weights[winner])


plt.scatter(weights, [0 for _ in range(len(weights))])
plt.title('Positions of the weights found with competitive learning')
plt.show()        


weights = np.array([[w] for w in weights])

model = RBF_network(weights, [2*np.pi/6 for _ in range(weights.shape[0])])
model.fit_data(X, sin_2x)

predictions = model.predict(test_x)

print ('Absolute relative error: ', str(np.sum(np.abs(predictions - sin_2x_test)) / predictions.shape[1]))

plt.scatter(weights, [0 for _ in range(len(weights))], color='orange', label='weights')
plt.plot(X[0], sin_2x[0], color='red', label='True function')
#plt.plot(X[0], sin_2x_test[0], color='green', label='True function without noise')
plt.plot(test_x[0], predictions[0], color='blue', label='Approximated function')
plt.title('Function approximated with RBF with weights initialized using CL')
plt.legend(loc='lower right')

plt.show()


