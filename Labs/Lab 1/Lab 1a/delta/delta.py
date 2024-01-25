import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys

batch_learning = False
separable = False
bias = True


if len(sys.argv) > 1:
    if sys.argv[1] == 'batch':
        batch_learning = True
    if sys.argv[1] == 'no-bias':
        bias = False
        separable = True
        batch_learning = True

if len(sys.argv) > 2:
    if sys.argv[2] == 'separable':
        separable = True

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

# If dataset should not be linearly separable
if not separable:
    mA = [1, 0.5]
    mB = [-1, 0.5]

# no bias dataset
if not bias:
    mA = [3, 0.5]
    mB = [0, 0.5]

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

if bias:
    dataset = np.vstack([dataset, np.ones(dataset.shape[1])])

print ('dataset.shape: ' + str(dataset.shape))
print ('targets.shape: ' + str(targets.shape))

# intiialize the weights: 
W = np.random.normal(size=(dataset.shape[0], 1))

# Training
iterations = 100 if batch_learning else 100 * dataset.shape[1]
lr = 0.0005
iterations_values = []
error_values = []
for i in range(iterations):
    if batch_learning:
        batch = dataset#[:, 0:10]
        batch_targets = targets#[:, 0:10]
    else:
        batch = np.array([dataset[:, i % dataset.shape[1]]]).T
        batch_targets = np.array([targets[:, i % dataset.shape[1]]]).T

    e =  W.T @ batch - batch_targets

    delta_w = -e @ batch.T

    W = W + lr * delta_w.T

    errors = (W.T @ batch - batch_targets) ** 2 

    iterations_values.append(W)
    error_values.append(np.sum(errors))


fig, ax = plt.subplots()
#plt.subplots_adjust(bottom=0.2)

x = np.linspace(-100, 100, 100)
y = W[0]/W[1] * x
if bias:
    y = W[0]/W[1] * x + W[2]/W[1]

l = plt.plot(x, y)

def update_slider(val):
    iteration_to_show = int(val) - 1
    current_w = iterations_values[iteration_to_show]

    x = np.linspace(-100, 100, 100)
    y = current_w[0]/current_w[1] * x
    if bias:
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
