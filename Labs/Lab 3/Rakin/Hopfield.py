import random
import numpy as np
from sklearn.base import BaseEstimator


def calculate_energy(weights,states):
    # Calculate the first part of the energy equation: -1/2 * sum(w_ij * s_i * s_j)
    energy_interaction = -np.dot(states, np.dot(weights, states.T))


    # Total energy is the sum of interaction energy and bias energy
    total_energy = energy_interaction 
    return total_energy


def _sign(x):
    return np.where(x >= 0, 1, -1)


class Hopfield(BaseEstimator):
    weights: np.array
    max_iterations: int
    bias: float
    zero_diagonal: bool
    random_weights: bool
    symmetric_weights: bool
    sparsity: float
    prediction_method: str
    energy_per_iteration = []
    self_connection = bool

    def __init__(
        self,
        max_iterations: int = 100,
        bias: float = 0,
        zero_diagonal: bool = True,
        random_weights: bool = False,
        symmetric_weights: bool = False,
        sparsity: float = 0,
        prediction_method: str = "batch",
        self_connection: bool = True,
    ):
        self.max_iterations = max_iterations
        self.bias = bias
        self.zero_diagonal = zero_diagonal
        self.random_weights = random_weights
        self.symmetric_weights = symmetric_weights
        self.sparsity = sparsity
        self.prediction_method = prediction_method
        self.energy_per_iteration = []
        self.self_connection = self_connection

    def fit(self, X, y=None):
        # Convert X to a numpy array if it isn't already
        X = np.array(X)

        # Calculate the average activity (rho)
        rho = np.mean(X)

        # Number of features (neurons)
        features = X.shape[1]

        # Initialize weights based on the presence of random weights
        if self.random_weights:
            weights = np.random.normal(size=(features, features))
        else:
            # Calculate the weights according to the learning rule for sparse patterns
            # Adjust the patterns by the average activity
            X_adjusted = X - rho
            # Apply the outer product to get the weights
            weights = (X_adjusted.T @ X_adjusted) / X.shape[0]

        # Apply zero diagonal if required
        if self.zero_diagonal or not self.self_connection:
            np.fill_diagonal(weights, 0)

        # Make weights symmetric if required
        if self.symmetric_weights:
            weights = (weights + weights.T) / 2

        # Finally, assign the calculated weights to self.weights
        self.weights = weights

    def predict(self, X) -> np.array:
        prediction = X.copy()  # Make a copy to avoid modifying the original input
        self.energy_per_iteration = []

        if self.prediction_method == "batch":
            for _ in range(self.max_iterations):
                # Calculate the state update for all neurons simultaneously
                net_input = prediction @ self.weights
                if self.bias is not None:  # Check if bias is applied
                    net_input -= self.bias

                # Incorporate sparsity in the prediction update
                if self.sparsity > 0.0:
                    prediction = 0.5 + 0.5 * np.sign(net_input)  # Adjusted for sparsity
                else:
                    prediction = np.sign(net_input)  # Original approach without sparsity

                # Calculate and store the energy
                energy = calculate_energy(
                    self.weights, prediction,
                )
                self.energy_per_iteration.append(energy)

        elif self.prediction_method == "sequential":
            for _ in range(self.max_iterations):
                for i in random.sample(range(prediction.shape[1]), prediction.shape[1]):
                    # Update the state of one neuron at a time, in a random order
                    net_input = prediction @ self.weights[:, i]
                    if self.bias is not None:  # Check if bias is applied
                        net_input -= self.bias

                    # Incorporate sparsity in the prediction update for each neuron
                    if self.sparsity > 0.0:
                        prediction[:, i] = 0.5 + 0.5 * np.sign(net_input)  # Adjusted for sparsity
                    else:
                        prediction[:, i] = np.sign(net_input)  # Original approach without sparsity

                # Calculate and store the energy
                energy = calculate_energy(
                    self.weights, prediction,
                )
                self.energy_per_iteration.append(energy)

        else:
            raise ValueError("Invalid prediction method")

        return prediction

    def get_energy(self):
        return self.energy_per_iteration
