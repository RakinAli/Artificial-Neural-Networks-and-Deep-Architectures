import sys
import os
sys.path.append(os.path.relpath('..'))
import numpy as np
from data_generator import data_generator
import matplotlib.pyplot as plt
from backprop.double_layer_nn import DoubleLayerNN

# Number of hidden nodes
hidden_nodes = 1


dg = data_generator(mA=[1, 0.3], mB=[0, -0.1], sigmaA=0.2, sigmaB=0.3)

dataset, targets = dg.generate_data_unseparable(100, 100)

dg.plot_data(dataset, targets)

dataset = dataset.T
targets = targets.T

# Add bias term to dataset
dataset = np.vstack((dataset, np.ones(dataset.shape[1])))

model = DoubleLayerNN(nr_of_hidden_nodes=hidden_nodes)
mse = model.fit_data(dataset, targets)

print ('accuracy: ' + str(model.accuracy(dataset, targets)))

test_data_x = np.random.uniform(-1.5, 1.5, 1000)
test_data_y = np.random.uniform(-1.5, 1.5, 1000)

dataset1 = np.array([np.copy(test_data_x), np.copy(test_data_y), np.ones(len(test_data_x))])

predictions = model.predict(dataset1)

red = dataset1[:, predictions[0, :] >= 0]
blue = dataset1[:, predictions[0, :] < 0]

plt.scatter(red[0], red[1], color='red')
plt.scatter(blue[0], blue[1], color='blue')
plt.scatter(dataset[0][targets[0] == 1], dataset[1][targets[0] == 1], color='green')
plt.scatter(dataset[0][targets[0] == 0], dataset[1][targets[0] == 0], color='orange')

plt.show()

plt.plot(mse)
plt.show()
