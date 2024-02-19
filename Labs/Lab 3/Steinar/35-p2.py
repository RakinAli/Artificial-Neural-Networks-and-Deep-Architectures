import numpy as np
from Hopfield_Network import Hopfield_Network
import matplotlib.pyplot as plt


retrievable_patterns_arr = []

dataset = np.sign(0.5 + np.random.normal(size=(300, 100)))

# Add the remaining patterns 
# and see how it affects the recall
for i in range(1, 100):
    print(i)
    training_data = dataset[0:i]
    model = Hopfield_Network(100, True)
    model.fit_data(training_data)

    # Try to recall all the patterns and see how many are recalled:
    retrievable_patterns = 0
    for p in range(0, i):
        original_pattern = dataset[p]
        
        converged, reconstruction = model.recall(original_pattern)
        if np.all(reconstruction == original_pattern):
            retrievable_patterns += 1

    retrievable_patterns_arr.append(retrievable_patterns / len(training_data))


print(retrievable_patterns_arr)
plt.plot(retrievable_patterns_arr)
plt.show()

        