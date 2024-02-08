from SOM import SOM
import numpy as np
import matplotlib.pyplot as plt

dataset = np.array([[0.4000, 0.4439], [0.2439, 0.1463], [0.1707, 0.2293], [0.2293, 0.7610], [0.5171, 0.9414], [0.8732, 0.6536], [0.6878, 0.5219], [0.8488, 0.3609], [0.6683, 0.2536], [0.6195, 0.2634]])
dataset = dataset.T

model = SOM(10, 2)
model.fit_data(dataset)

plt.scatter(dataset[0], dataset[1])
plt.scatter(model.weights[0], model.weights[1], color='red')
for i, w in enumerate(model.weights.T):
    plt.annotate(i, (w[0], w[1]))
plt.show()