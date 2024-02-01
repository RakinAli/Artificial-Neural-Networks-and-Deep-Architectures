import sys
import os
sys.path.append(os.path.relpath('..'))
import numpy as np
from data_generator import data_generator
import matplotlib.pyplot as plt
from backprop.double_layer_nn import DoubleLayerNN

# Assignment 3.1.1.1
# Number of hidden nodes
hidden_nodes = 4
dg = data_generator(mA=[1, 0.3], mB=[0, -0.1], sigmaA=0.2, sigmaB=0.3)

dataset, targets = dg.generate_data_unseparable(100, 100)
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
plt.scatter(dataset[0][targets[0] == -1], dataset[1][targets[0] == -1], color='orange')

plt.show()

plt.plot(mse)
plt.show()


# Split data
train_data, train_targets, validation_data, validation_targets = dg.split_data(dataset, targets, lambda x: x==1, lambda x: x==-1, 0.75, 0.254)
model = DoubleLayerNN()

train_mse, val_mse = model.fit_data(train_data, train_targets, val_X=validation_data, val_Y=validation_targets)

x1_min, x1_max = dataset[0,:].min() - 1, dataset[0, :].max() + 1
x2_min, x2_max = dataset[1,:].min() - 1, dataset[1, :].max() + 1

x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))
grid_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel(), np.ones(len(x2_mesh.ravel()))].T

Z = model.predict(grid_data)
print(Z.shape)
print (Z.reshape(x1_mesh.shape))

plt.contourf(x1_mesh, x2_mesh, Z.reshape(x1_mesh.shape), cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(train_data[0], train_data[1])
plt.show()

plt.plot(train_mse, color='blue')
plt.plot(val_mse, color='red')
plt.show()