import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys

def simultaneous_shuffle(a, b):
    assert (len(a) == len(b))
    permutation = np.random.permutation(len(a))
    return a[permutation], b[permutation]


# generate linearly seperable data.
np.random.seed(2)
n = 100
mA = [1.0, 0.3]
sigmaA = 0.2
mB = [0.0, -0.1]
sigmaB = 0.3


classA = np.zeros(shape=(2, n))
classB = np.zeros(shape=(2, n))
targetA = np.ones((n, 1))
targetB = np.zeros((n, 1)) - np.ones((n, 1))

classA[0, 0:int(n/2)] = np.random.normal(loc=-mA[0], scale=sigmaA, size=int(n/2))
classA[0, int(n/2):] = np.random.normal(loc=mA[0], scale=sigmaA, size=int(n/2))
classA[1] = np.random.normal(loc=mA[1], scale=sigmaA, size=n)

classB[0] = np.random.normal(loc=mB[0], scale=sigmaB, size=n)
classB[1] = np.random.normal(loc=mB[1], scale=sigmaB, size=n)

dataset = np.concatenate((classA.T, classB.T))
test_data = np.concatenate((testA.T, testB.T))
targets = np.concatenate((targetA, targetB))
test_targets = np.concatenate((test_targetsA, test_targetsB))

dataset, targets = simultaneous_shuffle(dataset, targets)

dataset = dataset.T
targets = targets.T

dataset = np.vstack([dataset, np.ones(dataset.shape[1])])

print ('dataset.shape: ' + str(dataset.shape))
print ('targets.shape: ' + str(targets.shape))

# intiialize the weights: 
W = np.random.normal(size=(dataset.shape[0], 1))

# Training
iterations = 100
lr = 0.0005
iterations_values = []
error_values = []
for i in range(iterations):
    batch = dataset#[:, 0:10]
    batch_targets = targets#[:, 0:10]

    e =  W.T @ batch - batch_targets

    delta_w = -e @ batch.T

    W = W + lr * delta_w.T

    errors = (W.T @ batch - batch_targets) ** 2 

    iterations_values.append(W)
    error_values.append(np.sum(errors))


fig, ax = plt.subplots()
#plt.subplots_adjust(bottom=0.2)

x = np.linspace(-100, 100, 100)
y = W[0]/W[1] * x + W[2]/W[1]

l = plt.plot(x, y)

def update_slider(val):
    iteration_to_show = int(val) - 1
    current_w = iterations_values[iteration_to_show]

    x = np.linspace(-100, 100, 100)
    y = current_w[0]/current_w[1] * x + current_w[2]/current_w[1]

    l[0].set_ydata(y)

ax.scatter(classA[0], classA[1], color='red')
ax.scatter(classB[0], classB[1], color='green')
ax.set_ylim(np.min(dataset[1,:]) - 0.5, np.max(dataset[1, :])+ 0.5)
ax.set_xlim(np.min(dataset[0,:]) - 0.5, np.max(dataset[0, :])+ 0.5)

#ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
#slider = Slider(ax_slider, label='iteration', valmin=1, valmax=iterations, valinit=iterations, valfmt='%0.0f')
#slider.on_changed(update_slider)

plt.show()

plt.scatter(list(range(iterations)), error_values)
plt.show()
