import numpy as np
import matplotlib.pyplot as plt


class data_generator:
    # Number of samples
    SAMPLES = 100

    # Initialize the data generator
    def __init__(self, mA, sigmaA, mB, sigmaB):
        # Mean and standard deviation of class A
        self.mA = mA
        self.sigmaA = sigmaA
        # Mean and standard deviation of class B
        self.mB = mB
        self.sigmaB = sigmaB

    # Method to shuffle two arrays simultaneously
    def simultaneous_shuffle(self, a, b):
        assert len(a) == len(b)
        permutation = np.random.permutation(len(a))
        return a[permutation], b[permutation]
    
    def simultaneous_shuffle_columns(self, a, b):
        assert a.shape[1] == b.shape[1]
        permutation = np.random.permutation(a.shape[1])
        return a[:, permutation], b[:, permutation]

    def generate_class_data(self, samplesA, samplesB):
        # Initialize arrays for class A and B
        classA = np.zeros((2, samplesA))
        classB = np.zeros((2, samplesB))
        targetA = np.ones((samplesA, 1))
        targetB = np.zeros((samplesB, 1))

        # Generate data for class A
        classA[0] = np.random.normal(loc=self.mA[0], scale=self.sigmaA, size=samplesA)
        classA[1] = np.random.normal(loc=self.mA[1], scale=self.sigmaA, size=samplesA)

        # Generate data for class B
        classB[0] = np.random.normal(loc=self.mB[0], scale=self.sigmaB, size=samplesB)
        classB[1] = np.random.normal(loc=self.mB[1], scale=self.sigmaB, size=samplesB)

        # Combine the data from both classes
        dataset = np.concatenate((classA.T, classB.T))
        targets = np.concatenate((targetA, targetB))

        # Shuffle the combined dataset and targets
        dataset, targets = self.simultaneous_shuffle(dataset, targets)
        return dataset, targets

    # Method to generate the data
    def generate_data(self, which_Class=None, training_percentage=None):
        np.random.seed(2)

        if isinstance(training_percentage, list):
            training_samplesA = int(training_percentage[0] * self.SAMPLES)
            training_samplesB = int(training_percentage[1] * self.SAMPLES)
            validation_samplesA = self.SAMPLES - training_samplesA
            validation_samplesB = self.SAMPLES - training_samplesB
        else:
            training_samples = int(training_percentage * self.SAMPLES)
            validation_samples = self.SAMPLES - training_samples
            if which_Class == "A":
                training_samplesA = training_samples
                validation_samplesA = validation_samples
                training_samplesB = 0
                validation_samplesB = 0
            elif which_Class == "B":
                training_samplesB = training_samples
                validation_samplesB = validation_samples
                training_samplesA = 0
                validation_samplesA = 0
            else:  # If neither A nor B is specified
                training_samplesA = training_samples
                training_samplesB = training_samples
                validation_samplesA = validation_samples
                validation_samplesB = validation_samples

        # Generate training data
        training_data, training_targets = self.generate_class_data(
            training_samplesA, training_samplesB
        )

        # Generate validation data
        validation_data, validation_targets = self.generate_class_data(
            validation_samplesA, validation_samplesB
        )

        return (training_data, training_targets), (validation_data, validation_targets)
    
    # Method to create linearly unseparable data
    def generate_data_unseparable(self, samplesA, samplesB):
        # Initialize arrays for class A and B
        classA = np.zeros((2, samplesA))
        classB = np.zeros((2, samplesB))
        targetA = np.ones((samplesA, 1))
        targetB = -np.ones((samplesB, 1))

        # Generate data for class A
        classA[0, 0:int(samplesA / 2)] = np.random.normal(loc=-self.mA[0], scale=self.sigmaA, size=int(samplesA / 2))
        classA[0, int(samplesA / 2):] = np.random.normal(loc=self.mA[0], scale=self.sigmaA, size=int(samplesA / 2))
        classA[1] = np.random.normal(loc=self.mA[1], scale=self.sigmaA, size=samplesA)

        # Generate data for class B
        classB[0] = np.random.normal(loc=self.mB[0], scale=self.sigmaB, size=samplesB)
        classB[1] = np.random.normal(loc=self.mB[1], scale=self.sigmaB, size=samplesB)

        # Combine the data from both classes
        dataset = np.concatenate((classA.T, classB.T))
        targets = np.concatenate((targetA, targetB))

        # Shuffle the combined dataset and targets
        dataset, targets = self.simultaneous_shuffle(dataset, targets)
        return dataset, targets

    # Method to plot the data
    def plot_data(self, dataset, targets):
        # Separate the dataset into two classes based on targets
        classA = dataset[targets.flatten() == 1]
        classB = dataset[targets.flatten() == 0]

        # Plotting the data
        plt.figure(figsize=(8, 6))
        plt.scatter(
            classA[:, 0], classA[:, 1], c="blue", label="Class A", edgecolors="k"
        )
        plt.scatter(
            classB[:, 0], classB[:, 1], c="red", label="Class B", edgecolors="k"
        )
        plt.legend()
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Data Distribution")
        plt.axis("equal")
        plt.show()

    
    def split_data(self, data, targets, lambda_class1, lambda_class2, ratio_class1_val, ratio_class2_val):
        # filter out the classes
        class_1_data = np.array([data[:, x] for x in range(targets.shape[1]) if lambda_class1(targets[0, x])]).T
        class_1_targets = np.array([targets[:, x] for x in range(targets.shape[1]) if lambda_class1(targets[0, x])]).T
        class_2_data = np.array([data[:, x] for x in range(targets.shape[1]) if lambda_class2(targets[0, x])]).T
        class_2_targets = np.array([targets[:, x] for x in range(targets.shape[1]) if lambda_class2(targets[0, x])]).T

        class_1_data, class_1_targets = self.simultaneous_shuffle_columns(class_1_data, class_1_targets)
        class_2_data, class_2_targets = self.simultaneous_shuffle_columns(class_2_data, class_2_targets)

        nr_in_class_1_val = int(ratio_class1_val * class_1_data.shape[1])
        nr_in_class_2_val = int(ratio_class2_val * class_2_data.shape[1])

        # split data        
        class_1_val_data = class_1_data[:, 0:nr_in_class_1_val]
        class_1_train_data = class_1_data[:, nr_in_class_1_val:]
        class_1_val_targets = class_1_targets[:, 0:nr_in_class_1_val]
        class_1_train_targets = class_1_targets[:, nr_in_class_1_val:]

        class_2_val_data = class_2_data[:, 0:nr_in_class_2_val]
        class_2_train_data = class_2_data[:, nr_in_class_2_val:]
        class_2_val_targets = class_2_targets[:, 0:nr_in_class_2_val]
        class_2_train_targets = class_2_targets[:, nr_in_class_2_val:]

        # Put the two classes together
        validation_data = np.c_[class_1_val_data, class_2_val_data]
        validation_targets = np.c_[class_1_val_targets, class_2_val_targets]

        train_data = np.c_[class_1_train_data, class_2_train_data]
        train_targets = np.c_[class_1_train_targets, class_2_train_targets]

        return train_data, train_targets, validation_data, validation_targets



        



    	




"""    
# Parameters for the data generator
mA = [1.0, 0.3]
sigmaA = 0.2
mB = [0.0, -0.1]
sigmaB = 0.3

# Create an instance of the data generator
dg = data_generator(mA, sigmaA, mB, sigmaB)

# Test 1: Single percentage value
(training_data_1, training_targets_1), (
    validation_data_1,
    validation_targets_1,
) = dg.generate_data(which_Class="AB", training_percentage=0.2)

dg.plot_data(training_data_1, training_targets_1)

# Test 2: List of two values
(training_data_2, training_targets_2), (
    validation_data_2,
    validation_targets_2,
) = dg.generate_data(which_Class="AB", training_percentage=[0.9, 0.1])

# Plot the training data
dg.plot_data(training_data_2, training_targets_2)


# Check sizes
sizes_1 = (len(training_targets_1), len(validation_targets_1))
sizes_2 = (len(training_targets_2), len(validation_targets_2))

sizes_1, sizes_2
"""
