import numpy as np
import matplotlib.pyplot as plt

# Generate data
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

X = np.concatenate((classA.T, classB.T))
T = np.concatenate((targetA, targetB))

X = X.T
T = T.T

X = np.vstack([X, np.ones(X.shape[1])])

print (classA.shape)
print (classB.shape)
print (T.shape)
print (X.shape)


###############################

# activation function (sigmoid with range ]-1:1[ )
def activation(Y):
    return 2/(1 + np.exp(-Y)) - 1

def sigmoid_derivative(sigmoid_values):
    return ((1 + sigmoid_values) * (1 - sigmoid_values)) / 2

# Some hyperparameters:
dim_i = X.shape[0] # input layer dimension
hidden_nodes = 9 # hidden layer dimension
output_nodes = 1 # output layer dimension

# Initialize the weights:
W_hidden = np.random.normal(0, 1, (hidden_nodes, dim_i))
W_output = np.random.normal(0, 1, (output_nodes, hidden_nodes))


# Forward pass
H_1 = W_hidden @ X
A_1 = activation(H_1)
Y_1 = W_output @ A_1
O = activation(Y_1)

# backwards pass
dW_output = (T - O) * sigmoid_derivative(O) * A_1
dW_hidden = dW_output * sigmoid_derivative(A_1) * X

print (dW_output.shape)