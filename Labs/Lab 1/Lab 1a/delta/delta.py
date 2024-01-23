import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def simultaneous_shuffle(a, b):
    assert (len(a) == len(b))
    permutation = np.random.permutation(len(a))
    return a[permutation], b[permutation]


# generate linearly seperable data.
np.random.seed(2)
n = 100
mA = [1.5, 0.5]
sigmaA = 0.5
mB = [-1.5, 0.5]
sigmaB = 0.5

classA = np.zeros(shape=(2, n))
classB = np.zeros(shape=(2, n))
targetA = np.ones((n, 1))
targetB = np.zeros((n, 1)) - np.ones((n, 1))

classA[0] = np.random.normal(loc=mA[0], scale=sigmaA, size=n)
classA[1] = np.random.normal(loc=mA[1], scale=sigmaA, size=n)

classB[0] = np.random.normal(loc=mB[0], scale=sigmaB, size=n)
classB[1] = np.random.normal(loc=mB[1], scale=sigmaB, size=n)

dataset = np.concatenate((classA.T, classB.T))
targets = np.concatenate((targetA, targetB))

dataset, targets = simultaneous_shuffle(dataset, targets)

dataset = dataset.T
targets = targets.T

print ('dataset.shape: ' + str(dataset.shape))
print ('targets.shape: ' + str(targets.shape))

# intiialize the weights: 
W = np.random.normal(size=(dataset.shape[0], 1))

# Training
iterations = 100
lr = 0.001
iterations_values = []
for i in range(iterations):

    batch = dataset#[:, 0:10]
    batch_targets = targets#[:, 0:10]

    error =  W.T @ batch - batch_targets

    delta_w = -error @ batch.T

    W = W + lr * delta_w.T
    iterations_values.append(W)


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

x = np.linspace(-100, 100, 100)
y = W[0]/W[1] * x

l = plt.plot(x, y)

def update_slider(val):
    iteration_to_show = int(val) - 1
    current_w = iterations_values[iteration_to_show]

    x = np.linspace(-100, 100, 100)
    y = current_w[0]/current_w[1] * x

    l[0].set_ydata(y)

ax.scatter(classA[0], classA[1], color='red')
ax.scatter(classB[0], classB[1], color='green')
ax.set_ylim(np.min(dataset[1,:]) - 0.5, np.max(dataset[1, :])+ 0.5)
ax.set_xlim(np.min(dataset[0,:]) - 0.5, np.max(dataset[0, :])+ 0.5)

ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_slider, label='iteration', valmin=1, valmax=iterations, valinit=iterations, valfmt='%0.0f')
slider.on_changed(update_slider)

plt.show()

