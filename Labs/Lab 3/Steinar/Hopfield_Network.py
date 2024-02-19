import numpy as np

class Hopfield_Network:
    
    def __init__(self, n_features, asynchronous=True):
        self.n_features = n_features
        self.asynchronous = asynchronous
        self.weights = np.zeros((n_features, n_features))
        self.intermediate_results = []
        self.energy_results = []

    
    def fit_data(self, X):
        """Fits the data X to the network, X is N x D matrix"""
        assert X.shape[1] == self.n_features, 'Incorrect number of dimensions'
        self.weights = 1/X.shape[0] * (X.T @ X)
        np.fill_diagonal(self.weights, 0)

    
    def recall(self, pattern, limit=10000, store_intermediate_energies=False):
        """returns converged(bool), pattern"""
        if self.asynchronous:
            return self.__recall_asynchronous(pattern, limit, store_intermediate_energies)
        
        return self.__recall_synchronous(pattern, limit)
        

    def energy(self, pattern):
        return -self.weights @ pattern @ pattern.T 
    

    def get_intermediate_results(self):
        return self.intermediate_results
    

    def get_energy_results(self):
        return self.energy_results

    
    def __recall_synchronous(self, pattern, limit):
        prev_pattern = np.zeros(self.n_features)
        curr_pattern = pattern
        i = 0
        prev_energy = np.inf

        self.intermediate_results = []

        while not np.all(prev_pattern == curr_pattern) and i < limit:
            self.intermediate_results.append(curr_pattern)
            prev_pattern = curr_pattern
            curr_pattern = np.sign(self.weights @ curr_pattern)
            curr_pattern = curr_pattern + (curr_pattern == 0)
            i += 1
            energy = self.energy(curr_pattern)
            self.energy_results.append(energy)
            
            if energy == prev_energy:
                break
            
            prev_energy = energy

        return np.all(prev_pattern == curr_pattern), curr_pattern 

    
    def __recall_asynchronous(self, pattern, limit, store_intermediate_energies):
        prev_pattern = np.zeros(self.n_features)
        curr_pattern = pattern.copy()
        epoch = 0
        self.intermediate_results.append(curr_pattern)
        if store_intermediate_energies:
            self.energy_results = []

        while not np.all(prev_pattern == curr_pattern) and epoch < limit:
            perm = np.random.permutation(self.n_features)
            prev_pattern = curr_pattern.copy()
            i = 0

            for p in perm:
                if store_intermediate_energies:
                    self.energy_results.append(self.energy(curr_pattern))
                if i % 100 == 0:
                    self.intermediate_results.append(curr_pattern)

                w = self.weights[p]

                new_x = np.sign(np.dot(w, curr_pattern))
                curr_pattern = curr_pattern.copy()
                curr_pattern[p] = new_x + (new_x == 0) # We do not want zeros
                i += 1

            epoch += 1

        return np.all(prev_pattern == curr_pattern), curr_pattern