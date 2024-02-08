import numpy as np

class SOM:

    def __init__(self, nr_of_nodes = 100, n_features = 100, learning_rate=0.2):
        self.weights =  np.random.uniform(size=(n_features, nr_of_nodes))
        self.learning_rate = learning_rate

    def fit_data(self, data, epochs=20):
        
        for e in range(epochs):
            for d in range(data.shape[1]):
                data_point = data[:, d]
                differences = self.weights.T - data_point
                norms = np.linalg.norm(differences, axis=1)
                closest_index = np.argmin(norms)

                self.update_weights(closest_index, data_point)


    def fit_data_circular(self, data, epochs=20):
        for e in range(epochs):
            for d in range(data.shape[1]):
                data_point = data[:, d]
                differences = self.weights.T - data_point
                norms = np.linalg.norm(differences, axis=1)
                closest_index = np.argmin(norms)

                self.update_weights_circular(closest_index, data_point)



    def update_weights(self, winning_index, point):
        if not winning_index == 0:
            self.update_neighbour(winning_index-1, point, 1)

        if not winning_index == self.weights.shape[1] - 1:
            self.update_neighbour(winning_index+1, point, 1)

        weights_to_update = self.weights[:, winning_index]

        weights_to_update = weights_to_update.T - self.learning_rate*(weights_to_update.T - point)

        self.weights[:, winning_index] = weights_to_update.T


    def update_weights_circular(self, winning_index, point):
            neighbour_left = winning_index-1
            neighbour_right = winning_index+1

            if neighbour_left < 0:
                neighbour_left = self.weights.shape[1]-1

            if neighbour_right >= self.weights.shape[1]:
                neighbour_right = 0


            self.update_neighbour(neighbour_left, point, 1)
            self.update_neighbour(neighbour_right, point, 1)

            weights_to_update = self.weights[:, winning_index]

            weights_to_update = weights_to_update.T - self.learning_rate*(weights_to_update.T - point)

            self.weights[:, winning_index] = weights_to_update.T



    def update_neighbour(self, index, point, distance):
        weights_to_update = self.weights[:, index]
        h = np.exp(-distance**2/0.5)
        weights_to_update = weights_to_update.T - self.learning_rate*h*(weights_to_update.T - point)
        self.weights[:, index] = weights_to_update.T


    def predict(self, data_point):
        differences = self.weights.T - data_point
        norms = np.linalg.norm(differences, axis=1)
        closest_index = np.argmin(norms)
        return closest_index

