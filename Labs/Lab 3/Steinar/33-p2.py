from Hopfield_Random_Weights import Hopfield_Network
import numpy as np
import matplotlib.pyplot as plt

model = Hopfield_Network(128, True, True)

random_pattern = np.random.binomial(n=1, p=0.5, size=128)
random_pattern[random_pattern == 0] = -1
converged, pattern = model.recall(random_pattern)

energy_results = model.get_energy_results()

plt.plot(energy_results)
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.title('Hopfield random with symmetric random weights')
plt.show()