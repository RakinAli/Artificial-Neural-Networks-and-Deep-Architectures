import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys

batch_learning = False
separable = True
bias = True

if len(sys.argv) > 1:
    if sys.argv[1] == "batch":
        batch_learning = True
    if sys.argv[1] == "no-bias":
        bias = False
        separable = True
        batch_learning = True

if len(sys.argv) > 2:
    if sys.argv[2] == "separable":
        separable = True


def simultaneous_shuffle(a, b):
    assert len(a) == len(b)
    permutation = np.random.permutation(len(a))
    return a[permutation], b[permutation]


# Function to calculate Mean Square Error (MSE)
def calculate_mse(W, dataset, targets):
    predictions = W.T @ dataset
    error = predictions - targets
    mse = np.mean(np.square(error))
    return mse


# generate linearly separable data.
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

print("dataset.shape: " + str(dataset.shape))
print("targets.shape: " + str(targets.shape))

# initialize the weights:
W = np.random.normal(size=(dataset.shape[0], 1))

# Training
iterations = 100 if batch_learning else 100 * dataset.shape[1]
lr = 0.001
iterations_values = []
mse_values = []  # List to store MSE values during training

for i in range(iterations):
    if batch_learning:
        batch = dataset
        batch_targets = targets
    else:
        batch = np.array([dataset[:, i % dataset.shape[1]]]).T
        batch_targets = np.array([targets[:, i % dataset.shape[1]]]).T

    error = W.T @ batch - batch_targets

    delta_w = -error @ batch.T

    W = W + lr * delta_w.T
    iterations_values.append(W)

    # Calculate and store the MSE for this iteration
    mse = calculate_mse(W, dataset, targets)
    mse_values.append(mse)

# Create the first graph with the slider
fig, ax1 = plt.subplots()
plt.subplots_adjust(bottom=0.2)

x = np.linspace(-100, 100, 100)
y = W[0] / W[1] * x
if bias:
    y = W[0] / W[1] * x + W[2] / W[1]

l = plt.plot(x, y)


def update_slider(val):
    iteration_to_show = int(val) - 1
    current_w = iterations_values[iteration_to_show]

    x = np.linspace(-100, 100, 100)
    y = current_w[0] / current_w[1] * x
    if bias:
        y = current_w[0] / current_w[1] * x + current_w[2] / current_w[1]

    l[0].set_ydata(y)


ax1.scatter(classA[0], classA[1], color="red")
ax1.scatter(classB[0], classB[1], color="green")
ax1.set_ylim(np.min(dataset[1, :]) - 0.5, np.max(dataset[1, :]) + 0.5)
ax1.set_xlim(np.min(dataset[0, :]) - 0.5, np.max(dataset[0, :]) + 0.5)

ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(
    ax_slider,
    label="iteration",
    valmin=1,
    valmax=iterations,
    valinit=iterations,
    valfmt="%0.0f",
)
slider.on_changed(update_slider)

# Create the second graph for displaying the MSE values
fig, ax2 = plt.subplots()
ax2.plot(range(1, iterations + 1), mse_values, label="MSE")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Mean Square Error")
ax2.legend()

plt.show()
