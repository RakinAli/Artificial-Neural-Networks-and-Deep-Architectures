import numpy as np
from Hopfield_Network import Hopfield_Network
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

with open('../pict.dat') as f:
    line = str.split(f.readline(), ',')
    dataset = np.array(line, dtype=float)
    dataset = dataset.reshape((int(dataset.shape[0] / 1024), 1024))


first_three = dataset[0:3]

model = Hopfield_Network(1024, True)
model.fit_data(first_three)

converged, converged_data = model.recall(dataset[10], 200)
intermediate_results = model.get_intermediate_results()

fig, axarr = plt.subplots(2, 1)

def update(val):
    axarr[0].imshow(intermediate_results[int(val)].reshape((32, 32)))


axarr[0].imshow(first_three[0].reshape((32, 32)))

slider = Slider(axarr[1], 'recall', 0.0, len(intermediate_results) - 1, 0, '%i')
slider.on_changed(update)

plt.show()
