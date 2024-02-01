import sys
import os
sys.path.append(os.path.relpath('..'))
import numpy as np
from data_generator import data_generator
import matplotlib.pyplot as plt
from backprop.double_layer_regression_nn import DoubleLayerNN
from sklearn.neural_network import MLPRegressor


X = np.arange(-0.5, 0.5, 0.01)
Y = np.arange(-0.5, 0.5, 0.01)

x_mesh, y_mesh = np.meshgrid(X, Y)
data_points = np.c_[x_mesh.ravel(), y_mesh.ravel()]
Z = np.exp(-(data_points[:, 0]**2  + data_points[:, 1]**2)/10) - 0.5

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x_mesh, y_mesh, Z.reshape(x_mesh.shape))
plt.show()

dataset = np.c_[x_mesh.ravel(), y_mesh.ravel(), np.ones(len(x_mesh.ravel()))].T
targets = np.array([Z])


for h in range(1, 26, 3):
    model = DoubleLayerNN(nr_of_hidden_nodes=h, learning_rate=0.0001)
    
    train_data_indices = np.random.choice(dataset.shape[1], int(0.8 * dataset.shape[1]), replace=False)
    train_data = dataset[:, train_data_indices]
    train_targets = targets[:, train_data_indices]
    mse = model.fit_data(train_data, train_targets, iterations=10000)
    predictions = model.predict(dataset)

    # Plot the data

    # plot mse:
    plt.plot(mse)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.title('Training error for MLP with ' + str(h) + ' hidden neurons')
    plt.savefig('./plots/train-error-' + str(h) + '-hidden-neurons.png')
    plt.clf()
    plt.cla()

 

#sklearn_model = MLPRegressor(solver='sgd', hidden_layer_sizes=(10), activation='logistic', momentum=0, batch_size=len(dataset.T))
#sklearn_model.fit(dataset.T, targets.T)
#
#print(sklearn_model.loss_)
#
#plt.plot(mse)
#plt.show()
#
#predictions = model.predict(dataset)
#print(np.all(predictions < targets + 0.1) and np.all(predictions > targets - 0.1))
#print(np.max(predictions))
#print(np.min(predictions))
#print(np.max(targets))
#print(np.max(targets))
#
#print((1/predictions.shape[1]) * np.sum((predictions - targets)**2))
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#ax.plot_surface(x_mesh, y_mesh, predictions[0].reshape(x_mesh.shape), color='blue')
#ax.plot_surface(x_mesh, y_mesh, targets.reshape(x_mesh.shape), color='green')
#ax.plot_surface(x_mesh, y_mesh, Z.reshape(x_mesh.shape), color='red')
#plt.show()
#
#print(x_mesh.ravel().shape)
#print(y_mesh.ravel().shape)
#print(predictions[0].shape)