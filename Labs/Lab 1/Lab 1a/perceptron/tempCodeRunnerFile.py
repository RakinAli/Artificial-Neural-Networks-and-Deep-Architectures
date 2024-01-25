
    # Add the bias to the dataset
    dataset = np.vstack((dataset, np.ones(dataset.shape[1])))

    return dataset, targets

# Create a main 
if __name__ == "__main__":
    dataset, label = generate_data(False)
    init_weights = np.random.normal(size=(dataset.shape[0]))

    updated_w, error_list = perceptron_learning(dataset, label, init_weights, 1, 100)

    # Plot the error 