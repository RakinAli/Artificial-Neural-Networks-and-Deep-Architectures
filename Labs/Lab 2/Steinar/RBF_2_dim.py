import numpy as np

class RBF_network:
    def __init__(self, weights_rbf, sigmas, output_nodes=1):
        self.weights_rbf = weights_rbf
        self.weights_output = np.zeros((output_nodes, weights_rbf.shape[0]))
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

    
    def fit_data_sequential(self, input, targets, epochs=200, lr=0.005):
        errors = []
        X = input.copy()
        Y = targets.copy()


        for i in range(epochs * X.shape[1]):
            if i % X.shape[1] == 0:
                perm = np.random.permutation(X.shape[1])
                X = X[:, perm]
                Y = Y[:, perm]

            batch = np.array([X[:, i % X.shape[1]]])
            batch_targets = np.array([Y[:, i % X.shape[1]]])
            
            batch = batch.T
            batch_targets = batch_targets.T

            activations = self.calculate_activations(batch)

            #if i % X.shape[1] == 0:
            error = (self.weights_output @ activations - batch_targets) ** 2
            errors.append(np.sum(error) / X.shape[1])

            e =  self.weights_output @ activations  - batch_targets
            
            delta_w = - activations @ e.T

            self.weights_output = self.weights_output + lr * delta_w.T

        return errors


    def predict(self, X):
        activations = self.calculate_activations(X)
        return self.calculate_output(activations)
    
    def predict_with_sign(self, X):
        activations = self.calculate_activations(X)
        return np.sign(self.calculate_output(activations))

        