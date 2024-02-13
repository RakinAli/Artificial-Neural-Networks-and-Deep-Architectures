import numpy as np
from RBF_2_dim import RBF_network
import matplotlib.pyplot as plt

dataset = None
targets = None

test_dataset = None
test_targets = None

with open('../dataset/ballist.dat') as f:
    d = f.readlines()
    
    for i in range(len(d)):
        d[i] = d[i].replace('\t', " ")
        d[i] = str.strip(d[i], '\n')
        d[i] = d[i].split(' ')

    d = np.array(d, dtype=float)
    dataset = d[:, 0:2]
    targets = d[:, 2:]


with open('../dataset/balltest.dat') as f:
    d = f.readlines()
    
    for i in range(len(d)):
        d[i] = d[i].replace('\t', " ")
        d[i] = str.strip(d[i], '\n')
        d[i] = d[i].split(' ')

    d = np.array(d, dtype=float)
    test_dataset = d[:, 0:2]
    test_targets = d[:, 2:]

np.random.seed(2)

# competitive learning:
weights = np.random.uniform(size=(10, 2))
eta = 0.1

prev_weights = weights.copy()

for e in range(10):
    for d in range(dataset.shape[0]):
        point = dataset[int(np.random.uniform()*dataset.shape[0])]

        dist = np.linalg.norm(weights - point, axis=1)

        args = np.argsort(dist)

        weights[args[0]] = weights[args[0]] + eta * (point - weights[args[0]])
        weights[args[1]] = weights[args[1]] + 0.5*eta * (point - weights[args[1]])

        prev_weights = weights.copy()



plt.scatter(dataset[:, 0], dataset[:, 1], color='blue', label='input data')
plt.scatter(weights[:, 0], weights[:, 1], color='green', label='weights', marker='x')
plt.title('The weights in the input space')
plt.legend()
plt.show()

model = RBF_network(weights, [3 for _ in range(weights.shape[0])], output_nodes=2)
errors = model.fit_data_sequential(dataset.T, targets.T, epochs=6000, lr=0.0005)

plt.plot(errors)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('Convergance for the 2 dimensional RBF network')
plt.show()

predictions = model.predict(test_dataset.T)
train_predictions = model.predict(dataset.T)

print(predictions.shape[1])
print ('mse test: ', str(np.sum((1/predictions.shape[1]) * (predictions - test_targets.T)**2)))
print ('mse train: ', str(np.sum((1/train_predictions.shape[1]) * (train_predictions - targets.T)**2)))




    

