from Hopfield_Network import Hopfield_Network
import numpy as np

model = Hopfield_Network(8, asynchronous=False)

data = np.array([[-1, -1, 1, -1, 1, -1, -1, 1], [-1, -1, -1, -1, -1, 1, -1, -1], [-1, 1, 1, -1, -1, 1, -1, 1]])

model.fit_data(data)

print(model.recall(data[2]))
