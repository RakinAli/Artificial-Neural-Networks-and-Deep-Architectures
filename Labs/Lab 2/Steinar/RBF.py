import numpy as np

class RBF_network:
    def __init__(self, weights_rbf, sigmas):
        self.weights_rbf = weights_rbf
        self.weights_output = np.zeros((1, weights_rbf.shape[0]))
        self.sigmas = sigmas

    def calculate_activations(self, X):
        n_dims, n_samples = X.shape
        n_rbf_neurons, _ = self.weights_rbf.shape
        activations = np.zeros((n_rbf_neurons, n_samples))

        for n in range(n_samples):
            for r in range(n_rbf_neurons):
                input = X[:, n]
                current_neuron = self.weights_rbf[r]
                current_sigma = self.sigmas[r]
                activations[r, n] = np.exp(-(np.linalg.norm(current_neuron - input)**2)/2*current_sigma**2)

        return activations
    
    def calculate_output(self, activations):
        return self.weights_output @ activations

    def fit_data(self, X, Y):
        activations = self.calculate_activations(X)
        self.weights_output = (np.linalg.inv(activations @ activations.T) @ activations @ Y.T).T


    def predict(self, X):
        activations = self.calculate_activations(X)
        return self.calculate_output(activations)
    
    def predict_with_sign(self, X):
        activations = self.calculate_activations(X)
        return np.sign(self.calculate_output(activations))

        