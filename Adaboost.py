import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


def load_data():
    """
    Loads the data from the specified file.

    Returns:
        A tuple containing:
            - coordinates: A NumPy array of points (x, y coordinates).
            - labels: A NumPy array of corresponding labels.
    """
    # Define the file path
    file_path = 'circle_separator.txt'
    # Lists to store coordinates and labels
    coordinates = []
    labels = []

    # Read the text file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into coordinates and label
            parts = line.strip().split()
            x = float(parts[0])
            y = float(parts[1])
            point = (x, y)
            label = int(parts[2])
            coordinates.append(point)
            labels.append(int(label))
    return np.array(coordinates), np.array(labels)


def plot_data_with_decision_boundary(classifiers, alphas, X_train, y_train, file_path, num_of_iterations,
                                     num_classifiers=8):
    """
    Plot the data points along with the decision boundaries of the specified number of ensemble classifiers.

    Args:
        classifiers: List of selected classifiers.
        alphas: List of corresponding classifier weights.
        X_train: Input features of the training set.
        y_train: True labels of the training set.
        file_path: File path to the data file.
        num_classifiers: Number of classifiers to plot decision boundaries for (default is 8).
    """

    # Load the data from the file
    data = np.loadtxt(file_path)
    coordinates, labels = data[:, :2], data[:, 2]

    # Define the plot boundaries based on the range of the training data
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    # Generate a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Plot each classifier's decision boundary
    for alpha, classifier in zip(alphas[:num_classifiers], classifiers[:num_classifiers]):
        Z = predict(np.c_[xx.ravel(), yy.ravel()], classifier)
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, alpha=0.4)

    # Plot the training data
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', cmap=plt.cm.coolwarm, label='Training Data', s=25)

    # Plot the data points from the file
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, marker='x', cmap=plt.cm.coolwarm, label='File Data',
                s=25)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f"Data Points and Decision Boundaries on run {num_of_iterations}")
    plt.legend()
    plt.show()

# Define hypothesis set using pairs of points
def generate_rules(X_train):
    """
     Generates all possible lines defined by pairs of points in the training set.
     Args:
         X_train: A NumPy array of training points (x, y coordinates).

     Returns:
         A list of pairs of (m, b) representing the slopes and intercepts of the lines.
     """
    rules = []
    for i in range(len(X_train)):
        for j in range(i + 1, len(X_train)):
            (x1, y1), (x2, y2) = X_train[i], X_train[j]
            if x1 != x2:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                rules.append((m, b))

    return rules


def predict(X, h):
    """
    Predicts labels for data points using the given line rule (m, b).

    Args:
        X: A NumPy array of data points (x, y coordinates).
        h: A tuple (m, b) representing the line rule.

    Returns:
        A NumPy array of predicted labels (1 or -1).
    """
    predictions = []
    for x in X:
        m, b = h[0], h[1]

        if x[1] >= m * x[0] + b:  # if it's above the line
            prediction = 1
        else:
            prediction = -1
        predictions.append(prediction)
    return np.array(predictions)


# Compute the indicator function [h(x_i) != y_i]
def compute_weighted_error(predictions, true_labels, weights):
    """
    Calculates the weighted error for a given hypothesis and data set.

    Args:
        predictions: A NumPy array of predicted labels.
        true_labels: A NumPy array of true labels.
        weights: A NumPy array of data point weights.

    Returns:
        The weighted error as a float.
    """
    # Initialize the weighted error
    weighted_error = 0

    # Ensure all arrays have the same length
    assert len(predictions) == len(true_labels) == len(weights)

    # Loop over the indices of the arrays
    for i in range(len(predictions)):
        # Get the prediction, true label, and weight for this index
        pred = predictions[i]
        true_label = true_labels[i]
        weight = weights[i]

        # If the prediction doesn't match the true label, it's an error
        if pred != true_label:
            error = 1
        else:
            error = 0

        # Add the weighted error for this data point to the total weighted error
        weighted_error += weight * error

    return weighted_error


class Adaboost:

    def __init__(self, x, y, iterations, k_most_important):
        self.x = x  # data
        self.y = y  # label [-1 ,1]
        self.iterations = iterations  # number of run to run Adaboost
        self.k_most_important = k_most_important  # number rules
        self.empirical_errors = np.zeros(
            shape=k_most_important)  # save the empirical error for each run for doing the average later
        self.true_errors = np.zeros(
            shape=k_most_important)  # save the true errors for each run for doing the average later
        self.classifiers = []  # save the rules of the current run of Adaboost
        self.alphas = []  # save the wights of the current run of Adaboost

    def get_classifiers(self):
        return self.classifiers

    def get_alphas(self):
        return self.alphas

    def run_adaboost(self):

        # Each run is One run of Adaboost
        for run in range(self.iterations):
            print(f"Enter the: {run} iteration")
            # Split the data randomly into ½ test (T) and ½ train (S).
            X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.5, train_size=0.5,
                                                                random_state=42 + run)

            # Use the points of S (not T) to define the hypothesis set of lines.
            rules = generate_rules(X_train)

            # 1. Initialize point weights 𝐷_1 (𝑥_𝑖 )=1/𝑛
            weights = np.ones(len(X_train)) / len(X_train)
            # reset the chosen rules and wights
            self.classifiers = []
            self.alphas = []
            # 2. For iteration t=1,…,k
            for t in range(self.k_most_important):

                # 3.Compute weighted error for each h ∈ H:
                weighted_errors = []
                for h in rules:
                    predictions = predict(X_train, h)
                    weighted_errors.append(compute_weighted_error(predictions, y_train, weights))

                # 4.Select classifier with min weighted error
                min_wight_error_index = np.argmin(weighted_errors)
                selected_classifier = rules[min_wight_error_index]

                # 5. Set classifier weight 𝛼_𝑡 based on its error
                # 𝛼_𝑡=1/2  ln〖(1−𝜖_𝑡 (ℎ_𝑡))/(𝜖_𝑡 (ℎ_𝑡))〗
                alpha_t = 0.5 * np.log(
                    (1 - weighted_errors[min_wight_error_index]) / weighted_errors[min_wight_error_index])

                # 6. Update weights
                weights = self.update_point_weights(weights, alpha_t, X_train, selected_classifier, y_train)

                # Add selected classifier and its weight to lists
                self.classifiers.append(selected_classifier)
                self.alphas.append(alpha_t)

                # compute the empirical error of the function Hk on S, and the true error of Hk on T
                self.evaluate_classifier(X_train, y_train, X_test, y_test, self.classifiers,
                                         self.alphas, t)
            if run == 0 or run == 10 or run == 20 or run == 30 or run == 40 or run == 49:
                plot_data_with_decision_boundary(adaboost.get_classifiers(), adaboost.get_alphas(), X_train, y_train,
                                                 'circle_separator.txt', num_of_iterations=run, num_classifiers=8)

    def evaluate_classifier(self, x_train, y_train, x_test, y_test, classifiers, alphas, index):
        """
        Evaluate the classifier H_k on the given training and test datasets.

        Args:
            X_train: Input features of the training set.
            y_train: True labels of the training set.
            X_test: Input features of the test set.
            y_test: True labels of the test set.
            classifiers: List of selected classifiers.
            alphas: List of corresponding classifier weights.
            index: the current index from 1..k
        """

        # Accumulate weighted predictions for each data point
        train_predictions = np.zeros(len(x_train))
        test_predictions = np.zeros(len(x_test))

        # 𝐻𝑘(𝑥)=𝑠𝑖𝑔𝑛(Σ𝛼𝑖ℎ𝑖(𝑥))
        for alpha, classifier in zip(alphas, classifiers):
            train_predictions += alpha * predict(x_train, classifier)
            test_predictions += alpha * predict(x_test, classifier)

        # Calculate empirical error:
        #  We compute the empirical error on the training set:
        # empirical_err = (number of misclassified examples) / (number of examples)
        H_k_empirical_err = np.sign(train_predictions)
        empirical_err = np.sum(H_k_empirical_err != y_train) / len(y_train)

        # Calculate true error:
        #  We compute the true error on the test set:
        #  true_err = (number of misclassified examples) / (number of examples)
        H_k_true_err = np.sign(test_predictions)
        true_err = np.sum(H_k_true_err != y_test) / len(y_test)

        # Accumulate errors over all runs
        self.empirical_errors[index] += empirical_err
        self.true_errors[index] += true_err

    def print_averaged_error(self):
        for k in range(self.k_most_important):
            print(
                f"k = {k}: Empirical Error = {np.round(self.empirical_errors[k] / self.iterations, 3)}, "
                f"True Error = {np.round(self.true_errors[k] / self.iterations, 3)}, "
                f"Difference (True - Empirical) = "
                f"{np.absolute((self.true_errors[k] / self.iterations) - (self.empirical_errors[k] / self.iterations))}")

    @staticmethod
    def update_point_weights(weights, alpha_t, X_train, selected_classifier, y_train):
        """
        Update point weights
                # 𝐷_(𝑡+1) (𝑥_𝑖 )=1/𝑍_𝑡  𝐷_𝑡 (𝑥_𝑖 ) 𝑒_^(−𝛼_𝑡 ℎ_𝑡 (𝑥_𝑖 ) 𝑦_𝑖 )
                # where Zt is a normalizing constant giving ∑_𝑖▒〖𝐷_(𝑡+1) (𝑥_𝑖 )=1〗
        """
        update_point_weights = weights * np.exp(-alpha_t * predict(X_train, selected_classifier) * y_train)
        normalizing_constant = np.sum(update_point_weights)
        update_point_weights /= normalizing_constant
        return update_point_weights







if __name__ == '__main__':
    # Load data
    x, y = load_data()
    # Number of iterations k
    iterations = 50
    # Number of most important classifiers to consider
    k_most_important = 8
    adaboost = Adaboost(x, y, iterations, k_most_important)
    adaboost.run_adaboost()
    adaboost.print_averaged_error()
