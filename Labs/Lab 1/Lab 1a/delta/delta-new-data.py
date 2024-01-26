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

classA = classA.T
classB = classB.T

testA_indexes = np.concatenate((np.random.choice(list(range(50)), 10, replace=False), np.random.choice(list(range(50)), 40, replace=False)))
testB_indexes = np.random.choice(list(range(100)), 50, replace=False)

trainA_indexes = np.array([x for x in range(100) if x not in testA_indexes])
trainB_indexes = np.array([x for x in range(100) if x not in testB_indexes])

trainA = classA[trainA_indexes].T
trainB = classB[trainB_indexes].T
train_targetsA = targetA[trainA_indexes]
train_targetsB = targetB[trainB_indexes]

testA = classA[testA_indexes].T
testB = classB[testB_indexes].T
test_targetsA = targetA[testA_indexes]
test_targetsB = targetB[testB_indexes]

classA = classA.T
classB = classB.T

train_dataset = np.concatenate((trainA.T, trainB.T))
dataset = np.concatenate((classA.T, classB.T))
targets = np.concatenate((targetA, targetB))
train_targets = np.concatenate((train_targetsA, train_targetsB))
test_dataset = np.concatenate((testA.T, testB.T))
test_targets = np.concatenate((test_targetsA, test_targetsB))

dataset, targets = simultaneous_shuffle(dataset, targets)
train_dataset, train_targets = simultaneous_shuffle(train_dataset, train_targets)
test_dataset, test_targets = simultaneous_shuffle(test_dataset, test_targets)

dataset = dataset.T
targets = targets.T
train_dataset = train_dataset.T
train_targets = train_targets.T
test_dataset = test_dataset.T
test_targets = test_targets.T

print (dataset.shape)
print (targets.shape)
print (train_dataset.shape)
print (train_targets.shape)
print (test_dataset.shape)
print (test_targets.shape)

dataset = np.vstack([dataset, np.ones(dataset.shape[1])])
train_dataset = np.vstack([train_dataset, np.ones(train_dataset.shape[1])])
test_dataset = np.vstack([test_dataset, np.ones(test_dataset.shape[1])])

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
    batch = train_dataset#[:, 0:10]
    batch_targets = train_targets#[:, 0:10]

    e =  W.T @ batch - batch_targets

    delta_w = -e @ batch.T

    W = W + lr * delta_w.T

    errors = (W.T @ batch - batch_targets) ** 2 

    iterations_values.append(W)
    error_values.append(np.sum(errors))


fig, ax = plt.subplots()
#plt.subplots_adjust(bottom=0.2)

x = np.linspace(-100, 100, 1000)
y = -W[0]/W[1] * x - W[2]/W[1]

ax.set_ylim(np.min(dataset[1,:]) - 0.5, np.max(dataset[1, :])+ 0.5)
ax.set_xlim(np.min(dataset[0,:]) - 0.5, np.max(dataset[0, :])+ 0.5)

l = ax.plot(x, y)

#def update_slider(val):
#    iteration_to_show = int(val) - 1
#    current_w = iterations_values[iteration_to_show]
#
#    x = np.linspace(-100, 100, 100)
#    y = current_w[0]/current_w[1] * x + current_w[2]/current_w[1]
#
#    l[0].set_ydata(y)
#
ax.scatter(classA[0], classA[1], color='red')
ax.scatter(classB[0], classB[1], color='green')
#ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
#slider = Slider(ax_slider, label='iteration', valmin=1, valmax=iterations, valinit=iterations, valfmt='%0.0f')
#slider.on_changed(update_slider)

plt.show()

plt.scatter(list(range(iterations)), error_values)
plt.show()



def calculate_accuracy_per_group(true_values, prediction_values):
    true_group_A = 0
    true_group_B = 0
    false_groupA = 0
    false_groupB = 0 
    
    for t, p in zip(true_values, prediction_values):
        if t == -1:
            if t == p:
                true_group_B = true_group_B + 1
            else:
                false_groupB = false_groupB + 1
        else:
            if t == p:
                true_group_A = true_group_A + 1
            else:
                false_groupA = false_groupA + 1

    return true_group_A / (true_group_A + false_groupA) if (true_group_A + false_groupA > 0) else -1, true_group_B / (true_group_B + false_groupB) if true_group_B + false_groupB > 0 else -1


test_values = W.T @ test_dataset
test_predictions = [1 if x >= 0 else -1 for x in test_values[0]]

train_values = W.T @ train_dataset
train_predictions = [1 if x >= 0 else -1 for x in train_values[0]]

dataset_values = W.T @ dataset
dataset_predictions = [1 if x >= 0 else -1 for x in dataset_values[0]]

test_accuracyA, test_accuracyB = calculate_accuracy_per_group(test_targets[0], test_predictions)
train_accuracyA, train_accuracyB = calculate_accuracy_per_group(train_targets[0], train_predictions)
dataset_accuracyA, dataset_accuracyB = calculate_accuracy_per_group(targets[0], dataset_predictions)

print (test_accuracyA)
print (test_accuracyB)
print (train_accuracyA)
print (train_accuracyB)
print (dataset_accuracyA)
print (dataset_accuracyB)
