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
        self.biases = []

        # Initialize the weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            # The shape of the weight matrix should be (layer_sizes[i], layer_sizes[i + 1])
            weight = np.random.randn(
                layer_sizes[i] + 1, layer_sizes[i + 1]
            )  # +1 for the bias term
            self.weights.append(weight)

    # Forward pass
    def forward(self, input_data):
        # Add bias term to the input data: append 1 to each input sample
        input_with_bias = np.hstack([input_data, np.ones((input_data.shape[0], 1))])
        activations = [input_with_bias]

        # Compute the activations for each layer
        for weight in self.weights[:-1]:  # All but the last layer include bias
            net_input = np.dot(activations[-1], weight)
            activation = sigmoid(net_input)
            # Append 1 for the bias term in the next layer
            activation_with_bias = np.hstack([activation, np.ones((activation.shape[0], 1))])
            activations.append(activation_with_bias)

        # For the last layer, we do not add a bias term since it is already included in the weights
        final_input = np.dot(activations[-1], self.weights[-1])
        final_activation = sigmoid(final_input)
        activations.append(final_activation)

        return activations[-1]  # The last set of activations is the output of the network

    # Train the network
    def train(self, training_data, training_targets, epochs, learning_rate, alpha=0.9):
        # Initialize the momentum terms for weights
        momentum_weights = [np.zeros_like(w) for w in self.weights]

        # Add bias term to the training data
        training_data_with_bias = np.hstack([training_data, np.ones((training_data.shape[0], 1))])

        for epoch in range(epochs):
            # Forward pass
            activations = self.forward(training_data_with_bias)
            activations.insert(0, training_data_with_bias)

            # Output layer error
            output_error = training_targets - activations[-1]
            delta = output_error * sigmoid_derivative(activations[-1])

            # Update momentum and weights for the output layer
            momentum_weights[-1] = alpha * momentum_weights[-1] - learning_rate * activations[-2].T.dot(delta)
            self.weights[-1] += momentum_weights[-1]

            # Backpropagation of error to hidden layers
            for i in reversed(range(len(self.weights) - 1)):
                delta = (self.weights[i + 1].dot(delta.T) * sigmoid_derivative(activations[i + 1].T)).T
                delta = delta[:, :-1]  # Remove delta for bias

                # Update momentum and weights for hidden layers
                momentum_weights[i] = alpha * momentum_weights[i] - learning_rate * activations[i].T.dot(delta)
                self.weights[i] += momentum_weights[i]

            # Compute the mean squared error
            mse = np.mean(np.square(output_error))
            print(f"Epoch {epoch+1}/{epochs}, MSE: {mse}")


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

    # Plotting training data
    generator.plot_data(training_data, training_targets)

    # Initialize the 2-layer perceptron
    input_size = training_data.shape[1]
    output_size = 1
    hidden_layers = [10, 10]
    layer_sizes = [input_size] + hidden_layers + [output_size]

    # Initialize the network
    network = n_layer_perceptron(layer_sizes)

    # Train the network
    epochs = 100
    learning_rate = 0.01
    weights, biases, error_list = network.train(
        training_data, training_targets, epochs, learning_rate
    )

    # Plot the error
    plt.figure()
    plt.plot(error_list)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Training error")
    plt.show()

    # Compute the accuracy on the training data
    predictions = network.predict(training_data)
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    accuracy = np.mean(predictions == training_targets)
    print("Accuracy on the training data: ", accuracy)


if __name__ == "__main__":
    main()
