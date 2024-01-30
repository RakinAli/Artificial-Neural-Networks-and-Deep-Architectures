from double_layer_nn import DoubleLayerNN
import numpy as np
import matplotlib.pyplot as plt

# Generate data
def simultaneous_shuffle(a, b):
    assert (len(a) == len(b))
    permutation = np.random.permutation(len(a))
    return a[permutation], b[permutation]


# generate linearly seperable data.
np.random.seed(2)
n = 100
mA = [1.0, 0.3]
sigmaA = 0.2
mB = [0.0, -0.1]
sigmaB = 0.3


classA = np.zeros(shape=(2, n))
classB = np.zeros(shape=(2, n))
targetA = np.ones((n, 1))
targetB = np.zeros((n, 1)) - np.ones((n, 1))

classA[0, 0:int(n/2)] = np.random.normal(loc=-mA[0], scale=sigmaA, size=int(n/2))
classA[0, int(n/2):] = np.random.normal(loc=mA[0], scale=sigmaA, size=int(n/2))
classA[1] = np.random.normal(loc=mA[1], scale=sigmaA, size=n)

classB[0] = np.random.normal(loc=mB[0], scale=sigmaB, size=n)
classB[1] = np.random.normal(loc=mB[1], scale=sigmaB, size=n)

X = np.concatenate((classA.T, classB.T))
T = np.concatenate((targetA, targetB))

X = X.T
T = T.T

X = np.vstack([X, np.ones(X.shape[1])])

###############################

model = DoubleLayerNN(nr_of_hidden_nodes=10)
mse = model.fit_data(X, T)

test_data_x = np.random.uniform(-1.5, 1.5, 1000)
test_data_y = np.random.uniform(-1.5, 1.5, 1000)

dataset = np.array([np.copy(test_data_x), np.copy(test_data_y), np.ones(len(test_data_x))])

predictions = model.predict(dataset)

predictions_dataset = model.predict(X)
print(predictions_dataset == T)

red = dataset[:, predictions[0, :] >= 0]
blue = dataset[:, predictions[0, :] < 0]

print (dataset.shape)
print (red.shape)
print (blue.shape)

plt.scatter(red[0], red[1], color='red')
plt.scatter(blue[0], blue[1], color='blue')
plt.scatter(classA[0], classA[1], color='green')
plt.scatter(classB[0], classB[1], color='orange')
plt.show()