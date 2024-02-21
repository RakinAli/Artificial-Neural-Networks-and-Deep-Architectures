import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Hopfield_Network import Hopfield_Network

with open('../pict.dat') as f:
    line = str.split(f.readline(), ',')
    dataset = np.array(line, dtype=float)
    dataset = dataset.reshape((int(dataset.shape[0] / 1024), 1024))


# Fraction of flipped bits
fraction = 0.8

first_three = dataset[0:3]
original_pattern = dataset[0]
distorted_pattern = original_pattern.copy()
choice = np.random.choice(len(distorted_pattern), int(fraction * len(distorted_pattern)), replace=False)

distorted_pattern[choice] = -distorted_pattern[choice]

print (np.all(distorted_pattern == original_pattern))


model = Hopfield_Network(1024, False)
model.fit_data(first_three)
converged, converged_data = model.recall(distorted_pattern, 200)
intermediate_results = model.get_intermediate_results()

fig = plt.figure()

slider_axis = fig.add_axes((0.1, 0.05, 0.85, 0.05))
original_axis = fig.add_axes((0.05, 0.15, 0.4, 0.85))
reconstructed_axis = fig.add_axes((0.55, 0.15, 0.4, 0.85))

original_axis.imshow(original_pattern.reshape((32, 32)))

prev_val = -1
def update(val):
    global prev_val
    if int(val) == prev_val:
        return
    prev_val = int(val) 
    reconstructed_axis.imshow(intermediate_results[int(val)].reshape((32, 32)))


original_axis.imshow(original_pattern.reshape((32, 32)))
reconstructed_axis.imshow(converged_data.reshape((32, 32)))

slider = Slider(slider_axis, 'recall', 0.0, len(intermediate_results) - 1, len(intermediate_results) - 1, '%i')
slider.on_changed(update)

plt.show()

fig, axarr = plt.subplots(1, 2)
fig.suptitle('Comparison of the original pattern and the recovered pattern')
axarr[0].imshow(original_pattern.reshape(32,32))
axarr[1].imshow(converged_data.reshape(32,32))
axarr[0].set_title('The original pattern')
axarr[1].set_title('The recovered pattern')

plt.show()


reconstructions = []

for noise in np.arange(0, 1, 0.1):
    original_pattern = dataset[0]
    distorted_pattern = original_pattern.copy()
    choice = np.random.choice(len(distorted_pattern), int(noise * len(distorted_pattern)), replace=False)

    distorted_pattern[choice] = -distorted_pattern[choice]

    converged, reconstructed_pattern = model.recall(distorted_pattern)

    if np.all(reconstructed_pattern == original_pattern):
        reconstructions.append(True)
    else:
        reconstructions.append(False)
        print(noise)


plt.scatter(np.arange(0, 1, 0.1), reconstructions)
plt.ylabel('Pattern recovered, 1=True, 0=False')
plt.xlabel('Noise')
plt.title('Did the network manage to reconstruct the pattern?')
plt.show()



