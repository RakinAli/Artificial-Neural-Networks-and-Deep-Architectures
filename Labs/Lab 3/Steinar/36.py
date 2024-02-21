import numpy as np
from Hopfield_Network import Hopfield_Network
import matplotlib.pyplot as plt

def recall(pattern, theta):
    prev_pattern = np.zeros(100)
    curr_pattern = pattern.copy()

    epoch = 0
    while not np.all(prev_pattern == curr_pattern) and epoch < 40:
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

    return curr_pattern


def check_if_all_stored(memorized_patterns):
    for datapoint in memorized_patterns:
        reconstruction = recall(datapoint, theta)

        if not np.all(reconstruction == datapoint):
            return False
        
    return True

stored_patterns_sum = np.zeros(130)

for i in range(100):
    print(i)

    dataset = np.random.binomial(1, 0.05, size=(100, 100))

    stored_patterns = []

    for theta in np.arange(-3, 10, 0.1):

        for nr_of_patterns in range(1, 100):
            memorized_patterns = dataset[0:nr_of_patterns]

            rho = np.sum(memorized_patterns) / (len(memorized_patterns) * 100)

            weights_matrix = (memorized_patterns.T - rho) @ (memorized_patterns - rho)

            if not check_if_all_stored(memorized_patterns=memorized_patterns):
                stored_patterns.append(nr_of_patterns - 1)
                break
            
    stored_patterns_sum = stored_patterns_sum + np.array(stored_patterns)


print(stored_patterns_sum / 100)


plt.plot(np.arange(-3, 10, 0.1), stored_patterns_sum / 100)
plt.xlabel('Value of theta')
plt.ylabel('Avg number of stored patterns')
plt.title('Average numbers of stored patterns with 10\% activation')
plt.show()