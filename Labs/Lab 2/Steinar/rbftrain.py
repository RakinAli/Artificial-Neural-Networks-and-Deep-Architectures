import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3, 10, 0.05)
y = 2.5 * np.exp(-(x)**2/9) + 3.2 * np.exp(-(x)**2/4) + np.random.normal(0, 1, len(x))

nParam = 2
A = np.zeros((len(x), nParam), dtype=float)
A[:,0] = np.exp(-(x)**2/9)
A[:, 1] = np.exp(-(x-0.5)**2/4)
(p, residuals, rank, s) = np.linalg.lstsq(A, y)
plt.plot(x, y, '.')
plt.plot(x, p[0]*A[:,0] + p[1]*A[:, 1], 'x')
plt.show()