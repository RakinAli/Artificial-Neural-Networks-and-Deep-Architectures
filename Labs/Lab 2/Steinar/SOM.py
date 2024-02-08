import numpy as np

class SOM:

    def __init__(self, nr_of_nodes = 100, n_features = 100, learning_rate=0.2):
        self.weights =  np.random.normal(size=(n_features, nr_of_nodes))
        self.learning_rate = learning_rate

    def fit_data(self, data, epochs=10):
        
        for e in range(epochs):
            for d in range(data.shape[1]):
                data_point = data[:, 1]
                differences = self.weights.T - data_point
                norms = np.linalg.norm(differences, axis=1)
                closest_index = np.argmin(norms)

                self.update_weights(closest_index, data_point)


    def update_weights(self, winning_index, point):
        index_from = winning_index - 1
        if winning_index == 0:
            index_from = 0
        
        index_to = winning_index + 1
        if winning_index == self.weights.shape[1] - 1:
            index_to = winning_index 

        weights_to_update = self.weights[:, index_from:index_to+1]

        weights_to_update = weights_to_update.T + self.learning_rate*(weights_to_update.T - point)

        print(self.weights.shape)
        print(weights_to_update.shape)

        self.weights[:, index_from:index_to] = weights_to_update.T



