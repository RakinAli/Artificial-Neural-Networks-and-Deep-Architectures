import numpy as np
from Hopfield_Network import Hopfield_Network
import matplotlib.pyplot as plt

with open('../pict.dat') as f:
    line = str.split(f.readline(), ',')
    dataset = np.array(line, dtype=float)
    dataset = dataset.reshape((int(dataset.shape[0] / 1024), 1024))


# Add the remaining patterns 
# and see how it affects the recall
    
for i in range(2, 9):
    training_data = dataset[0:i]
    model = Hopfield_Network(1024, True)
    model.fit_data(training_data)

    fig, axarr = plt.subplots(i, 2)

    # Try to recall all patterns when they are moderately distorted:
    for p in range(0, i):
        original_pattern = dataset[p]
        distorted_pattern = original_pattern.copy()
        choice = np.random.choice(len(distorted_pattern), int(0.1 * len(distorted_pattern)), replace=False)

        converged, reconstruction = model.recall(original_pattern)
        print(converged)
        axarr[p, 0].imshow(training_data[p].reshape((32, 32)))
        axarr[p, 1].imshow(reconstruction.reshape((32, 32)))

    plt.show()

        