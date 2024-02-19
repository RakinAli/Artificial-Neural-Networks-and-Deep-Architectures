import numpy as np
from Hopfield_Network import Hopfield_Network
import matplotlib.pyplot as plt

dataset = np.random.binomial(1, 0.1, size=(10, 100))

memorized_patterns = dataset[0:4]

rho = np.sum(memorized_patterns) / (len(memorized_patterns) * 100)

weights_matrix = (memorized_patterns.T - rho) @ (memorized_patterns - rho)

# recall step
pattern = memorized_patterns[0] # Pattern to recall

prev_pattern = np.zeros(100)
curr_pattern = pattern.copy()

epoch = 0

theta = 0.9

while not np.all(prev_pattern == curr_pattern) and epoch < 200:
    perm = np.random.permutation(100)
    prev_pattern = curr_pattern.copy()
    i = 0

    for p in perm:
        w = weights_matrix[p]

        new_x = np.sign(np.dot(w, curr_pattern) - theta)
        curr_pattern = curr_pattern.copy()
        curr_pattern[p] = 0.5 + 0.5 * (new_x + (new_x == 0))
        i += 1

    epoch += 1

print(curr_pattern)
print (np.all(curr_pattern == pattern))

