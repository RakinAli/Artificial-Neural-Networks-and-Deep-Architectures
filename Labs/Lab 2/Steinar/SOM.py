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


class SOM_grid:
    def __init__(self, grid_size, dim, eta=0.2, eta_n=0.1, alpha=0.8):
        if not type(grid_size) is tuple:
            raise TypeError('grid size must be a tuple')
        
        self.grid_size = grid_size
        self.neurons = np.random.uniform(size=(grid_size[0], grid_size[1], dim))
        self.alpha = alpha
        self.eta = eta
        self.eta_n = eta_n


    def fit_data(self, data, epochs=20):
        previous_learning_rate = self.eta
        previous_learning_rate_n = self.eta_n

        for e in range(epochs):
            learning_rate = self.calculate_learning_rate(e, epochs, previous_learning_rate)
            learning_rate_n = self.calculate_learning_rate(e, epochs, previous_learning_rate_n)

            for d in range(data.shape[1]):

                data_point = data[:, d]
                differences = self.neurons - data_point
                norms = np.linalg.norm(differences, axis=2)

                closest_index = np.argmin(norms)
                closest_index = np.unravel_index(closest_index, norms.shape)

                self.update_neurons(closest_index, data_point, learning_rate, learning_rate_n)

            previous_learning_rate = learning_rate
            previous_learning_rate_n = learning_rate_n


    def calculate_learning_rate(self, k, k_max, eta_prev):
        return self.alpha * eta_prev**(k/k_max)


    def update_neurons(self, closest_index, data_point, learning_rate, learning_rate_n):
        neighbours = self.get_neighbours(closest_index)

        # Update the winning node
        update_neuron = self.neurons[closest_index]
        update_neuron = update_neuron - learning_rate*(update_neuron- data_point)
        self.neurons[closest_index] = update_neuron

        # Update the neighbours
        for index in neighbours:
            update_neuron = self.neurons[index]
            update_neuron = update_neuron - learning_rate_n*(update_neuron- data_point)
            self.neurons[index] = update_neuron


    def get_neighbours(self, index):
        neighbours = [self.right(index), self.up(index), self.down(index), self.left(index)]
        neighbours = list(filter(lambda a: a is not None, neighbours))

        return neighbours



    def right(self, index):
        x, y = index

        if x + 1 >= self.grid_size[0]:
            return None
        
        return (x + 1, y)
    
    def left(self, index):
        x, y = index

        if x - 1 < 0:
            return None
        
        return (x-1, y)
    
    def down(self, index):
        x, y = index

        if y + 1 >= self.grid_size[1]:
            return None
        
        return (x, y+1)

    def up(self, index):
        x, y = index

        if y - 1 < 0:
            return None
        
        return (x, y-1)
