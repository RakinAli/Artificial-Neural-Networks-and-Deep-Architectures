import numpy as np
import matplotlib.pyplot as plt

def simultaneous_shuffle(a, b):
    assert (len(a) == len(b))
    permutation = np.random.permutation(len(a))
    return a[permutation], b[permutation]

np.random.seed(2)
n = 100
mA = [3, 0.5]
sigmaA = 0.5
mB = [1.5, 0.5]
sigmaB = 0.5

classA = np.zeros(shape=(2,n))
classB = np.zeros(shape=(2,n))
targetA = np.ones((n, 1))
targetB = np.zeros((n, 1))

classA[0] = np.random.normal(loc=mA[0], scale=sigmaA, size=n)
classA[1] = np.random.normal(loc=mA[1], scale=sigmaA, size=n)

classB[0] = np.random.normal(loc=mB[0], scale=sigmaB, size=n)
classB[1] = np.random.normal(loc=mB[1], scale=sigmaB, size=n)

dataset = np.concatenate((classA.T, classB.T))
targets = np.concatenate((targetA, targetB))

print (dataset.shape)
print (targets.shape)

dataset, targets = simultaneous_shuffle(dataset, targets)
dataset = dataset.T

# Add the bias to the dataset
dataset = np.vstack((dataset, np.ones(dataset.shape[1])))


########
# initialize the weights
W = np.random.normal(size=(dataset.shape[0]))


# Show the data in a plot
x = np.linspace(-1, 1, 100)
y = W[0]/W[1] * x + W[2] / W[1]

plt.plot(x, y)
plt.scatter(classA[0], classA[1], color='red')
plt.scatter(classB[0], classB[1], color='green')
plt.show()


iterations = 100
errors = []
for i in range(iterations):
    index = i % n
    datapoint = dataset[:, index]
    results = np.dot(W, datapoint)

    errors.append(np.sum(np.power(np.asmatrix(W.T @ dataset).T - targets, 2)))

    classification = 1 if results >= 0 else 0
    true_class = targets[index, 0]

    if not classification == true_class:
        if true_class == 1:
            W = W + datapoint 
        else:
            W = W - datapoint


all_results = np.dot(W, dataset) >= 0
print (errors)

print (np.sum(all_results == targets.reshape(-1)))

x = np.linspace(-1, 1, 100)
y = W[0]/W[1] * x + W[2] / W[1]

plt.plot(x, y)
plt.scatter(classA[0], classA[1], color='red')
plt.scatter(classB[0], classB[1], color='green')
plt.show()

plt.plot(list(range(iterations)), errors)
plt.ylabel('Square error')
plt.xlabel('Iterations')
plt.show()

