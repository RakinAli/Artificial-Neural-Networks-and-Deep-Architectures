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
    def generate_data(self):
        np.random.seed(2)

        classA = np.zeros((2, self.SAMPLES))
        classB = np.zeros((2, self.SAMPLES))
        targetA = np.ones((self.SAMPLES, 1))
        targetB = np.zeros((self.SAMPLES, 1))

        classA[0] = np.random.normal(
            loc=self.mA[0], scale=self.sigmaA, size=self.SAMPLES
        )
        classA[1] = np.random.normal(
            loc=self.mA[1], scale=self.sigmaA, size=self.SAMPLES
        )

        classB[0] = np.random.normal(
            loc=self.mB[0], scale=self.sigmaB, size=self.SAMPLES
        )
        classB[1] = np.random.normal(
            loc=self.mB[1], scale=self.sigmaB, size=self.SAMPLES
        )

        dataset = np.concatenate((classA.T, classB.T))
        targets = np.concatenate((targetA, targetB))
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


if __name__ == "__main__":
    # Choose paramters mA, SigmA and mB, SigmB so that it is linearly seperable

    # 3.1.1 Linearly seperable data
    mA = [1.5, 0.5]
    sigmaA = 0.5
    mB = [-1.5, 0.5]
    sigmaB = 0.5

    dg = data_generator(mA, sigmaA, mB, sigmaB)
    dataset, targets = dg.generate_data()
    dg.plot_data(dataset, targets)
