import numpy as np

class Hopfield_Network:
    
    def __init__(self, n_features, asynchronous=True):
        self.n_features = n_features
        self.asynchronous = asynchronous
        self.weights = np.zeros((n_features, n_features))

    
    def fit_data(self, X):
        """Fits the data X to the network, X is N x D matrix"""
        assert X.shape[1] == self.n_features, 'Incorrect number of dimensions'
        self.weights = 1/X.shape[0] * (X.T @ X)
        np.fill_diagonal(self.weights, 0)

    
    def recall(self, pattern, limit=100):
        """returns converged(bool), pattern"""
        if self.asynchronous:
            return self.__recall_asynchronous(pattern, limit)
        
        return self.__recall_synchronous(pattern, limit)
        

    
    def __recall_synchronous(self, pattern, limit):
        prev_pattern = np.zeros(self.n_features)
        curr_pattern = pattern
        i = 0

        while not np.all(prev_pattern == curr_pattern) and i < limit:
            prev_pattern = curr_pattern
            curr_pattern = np.sign(self.weights @ curr_pattern)
            i += 1

        return np.all(prev_pattern == curr_pattern), curr_pattern 

    
    def __recall_asynchronous(self, pattern, limit):
        pass