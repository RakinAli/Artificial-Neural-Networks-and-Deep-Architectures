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

    # Method to generate the data
    def generate_data(self, which_Class=None, percentage=None):
        np.random.seed(2)

        # Adjust the number of samples based on the percentage provided
        if which_Class == "AB" and percentage is not None:
            samplesA = int(percentage[0] * self.SAMPLES)
            samplesB = int (percentage[1] * self.SAMPLES)
        else:
            samplesA = (
                self.SAMPLES if which_Class != "A" else int(percentage * self.SAMPLES)
            )
            samplesB = (
                self.SAMPLES if which_Class != "B" else int(percentage * self.SAMPLES)
            )

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


# Parameters from the image
mA = [1.0, 0.3]
sigmaA = 0.2
mB = [0.0, -0.1]
sigmaB = 0.3

# Create an instance of the data generator
dg = data_generator(mA, sigmaA, mB, sigmaB)

# Now we can call the generate_data method with the class and percentage to generate a subset of data.
# For example, to generate 50% of class A data, we would call:
dataset, targets = dg.generate_data(which_Class="A", percentage=0.5)

# Plot the data
dg.plot_data(dataset, targets)

# Checking the size of the generated data to ensure it's correct as per the requirement
print(f"Class A data points: {np.sum(targets == 1)}")
print(f"Class B data points: {np.sum(targets == 0)}")
