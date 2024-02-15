from Hopfield_Network import Hopfield_Network
import numpy as np

model = Hopfield_Network(8, asynchronous=True)

data = np.array([[-1, -1, 1, -1, 1, -1, -1, 1], [-1, -1, -1, -1, -1, 1, -1, -1], [-1, 1, 1, -1, -1, 1, -1, 1]])

distorted_data = np.array([[1, -1, 1, -1, 1, -1, -1, 1], [1, 1, -1, -1, -1, 1, -1, -1], [1, 1, 1, -1, 1, 1, -1, 1]])


model.fit_data(data)
print(model.recall(distorted_data[2]))
