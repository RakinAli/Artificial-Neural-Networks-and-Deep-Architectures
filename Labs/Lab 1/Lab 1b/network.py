import numpy as np
import matplotlib.pyplot as plt
import data_generator as dg


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# This is the class for the two-layer perceptron
class n_layer_perceptron:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []

        # Initialize the weights for each layer
        for i in range(len(layer_sizes) - 1):
            # The shape of the weight matrix should be (layer_sizes[i] + 1, layer_sizes[i + 1])
            # +1 for the bias term for all layers
            weight = np.random.randn(layer_sizes[i] + 1, layer_sizes[i + 1])
            self.weights.append(weight)
            print(f"Weight {i} shape: {weight.shape}")  # Add this print statement

    # Forward pass
    def forward(self, input_data):
        # Add bias term to the input data: append 1 to each input sample
        input_with_bias = np.hstack([input_data, np.ones((input_data.shape[0], 1))])
        activations = [input_with_bias]

        for i, weight in enumerate(self.weights):
            print(
                f"Shape before dot product at layer {i}: {activations[-1].shape}, weight shape: {weight.shape}"
            )

            net_input = np.dot(activations[-1], weight)
            activation = sigmoid(net_input)

            # Do not add bias term for the output layer
            if i < len(self.weights) - 1:
                # Append 1 for the bias term in the next layer
                activation = np.hstack([activation, np.ones((activation.shape[0], 1))])
            activations.append(activation)

        return activations[-1]

    # Train the network
    # Train the network

    # Train the network


    def train(self, training_data, training_targets, epochs, learning_rate, alpha=0.9):
        error_list = []
        momentum_weights = [np.zeros_like(w) for w in self.weights]

        for epoch in range(epochs):
            activations = self.forward(training_data)

            # Output layer error
            output_error = training_targets - activations[-1]
            delta = output_error * sigmoid_derivative(activations[-1])

            # Update momentum and weights for the output layer
            momentum_weights[-1] = alpha * momentum_weights[-1] - learning_rate * np.dot(
                activations[-2].T, delta
            )
            self.weights[-1] += momentum_weights[-1]

            # Backpropagation of error to hidden layers
            for i in reversed(range(len(self.weights) - 1)):
                # Get the activations for the current layer, excluding the bias term
                current_activations = activations[i][:, :-1] if i > 0 else activations[i]

                # Calculate delta for the current layer
                delta = np.dot(delta, self.weights[i + 1].T) * sigmoid_derivative(
                    current_activations
                )
                delta = delta[:, :-1]  # Exclude the bias term from the delta calculation

                # Update momentum and weights for the current layer
                prev_activations = activations[i - 1] if i > 1 else training_data
                momentum_weights[i] = alpha * momentum_weights[i] - learning_rate * np.dot(
                    prev_activations.T, delta
                )
                self.weights[i] += momentum_weights[i]

            # Compute the mean squared error
            mse = np.mean(np.square(output_error))
            error_list.append(mse)
            print(f"Epoch {epoch+1}/{epochs}, MSE: {mse}")

        return self.weights, error_list

    def predict(self, input_data):
        activations = self.forward(input_data)
        return activations[-1]


def main():
    # Generate the data
    generator = dg.data_generator(mA=[1.5, 0.5], sigmaA=0.5, mB=[-1.5, 0.5], sigmaB=0.5)
    (training_data, training_targets), (
        validation_data,
        validation_targets,
    ) = generator.generate_data(which_Class="AB", training_percentage=0.8)
    print(f"Shape of training data before adding bias: {training_data.shape}")

    print(f"Shape of training data: {training_data[0]}")

    input_size = training_data.shape[1]  # Number of features in the data
    output_size = 1  # Assuming binary classification
    hidden_layers = [10, 10]
    layer_sizes = [input_size] + hidden_layers + [output_size]

    # Initialize the network
    network = n_layer_perceptron(layer_sizes)

    # Train the network
    epochs = 100
    learning_rate = 0.01
    error_list = network.train(training_data, training_targets, epochs, learning_rate)

    # Plot the error
    plt.figure()
    plt.plot(error_list)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Training Error")
    plt.show()

    # Compute the accuracy on the training data
    predictions = network.predict(training_data)
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    accuracy = np.mean(predictions == training_targets)
    print("Accuracy on the training data: ", accuracy)


if __name__ == "__main__":
    main()
