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

        # Initialize the weights and biases
        for layer in range(len(layer_sizes) - 1):
            weights = np.random.normal(layer_sizes[layer], layer_sizes[layer + 1])
            biases = np.random.normal(layer_sizes[layer + 1])
            self.weights.append(weights)
            self.biases.append(biases)

    # Forward pass
    def forward(self, input_data):
        # Add bias term to the input data: append 1 to each input sample
        input_with_bias = np.hstack([input_data, np.ones((input_data.shape[0], 1))])
        activations = [input_with_bias]

        # Compute the activations for each layer
        for weight, bias in zip(self.weights, self.biases):
            net_input = np.dot(activations[-1], weight) + bias
            activation = sigmoid(net_input)
            # Append 1 for the bias term in the next layer
            activation_with_bias = np.hstack(
                [activation, np.ones((activation.shape[0], 1))]
            )
            activations.append(activation_with_bias)

        # Remove the last bias term added (not needed for the output layer)
        final_activations = activations[-1][:, :-1]
        return final_activations
    
    def train(self, training_data, training_targets, epochs, learning_rate, alpha=0.9, do_batch=True):
        # Initialize the momentum weights
        momentum_weights = [np.zeros(weight.shape) for weight in self.weights]
        momentum_bias = [np.zeros(bias.shape) for bias in self.biases]

        # First shuffle the data
        permutation = np.random.permutation(len(training_data))
        training_data = training_data[permutation]
        training_targets = training_targets[permutation]

        # Store the error for each epoch
        error_list = []
  
        for epoch in range(epochs):
            activations = self.forward(training_data)

            # Output delta
            delta = activations[-1] - training_targets * sigmoid_derivative(activations[-1])

            # Update momentum for the output layer
            momentum_weights[-1] = alpha * momentum_weights[-1] - learning_rate * activations[-2].T.dot(delta)
            momentum_bias[-1] = alpha * momentum_bias[-1] - learning_rate * np.sum(delta, axis=0)

            # Apply update to weights and biases
            self.weights[-1] += momentum_weights[-1]
            self.biases[-1] += momentum_bias[-1]

            # Backpropagate the error
            for layer in reversed(range(len(self.weights) -1)):
                delta_hidden = (self.weights[layer + 1].dot(delta.T)).T * sigmoid_derivative(activations[layer + 1])

                # Update momentum for the hidden layer
                momentum_weights[layer] = alpha * momentum_weights[layer] - learning_rate * activations[layer].T.dot(delta_hidden)
                momentum_bias[layer] = alpha * momentum_bias[layer] - learning_rate * np.sum(delta_hidden, axis=0)

                # Apply update to weights and biases
                self.weights[layer] += momentum_weights[layer]
                self.biases[layer] += momentum_bias[layer]

                # Update delta for the next layer   
                delta = delta_hidden
            # Compute the error (mean squared error)
            error = np.mean((activations[-1] - training_targets) ** 2)
            error_list.append(error)
            # Return the weights, biases and error
        return self.weights, self.biases, error_list
    
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
    hidden_layers =[10,10]
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
