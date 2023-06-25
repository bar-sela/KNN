import numpy as np


# Function to calculate L1 distance between two points
def l1_distance(a, b):
    return np.sum(np.abs(a - b))


# Function to calculate L2 distance between two points
def l2_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# Function to calculate L-infinity distance between two points
def linf_distance(a, b):
    return np.max(np.abs(a - b))


# Class implementing the KNN algorithm
class model_KNN():

    def __init__(self, k_size, p_size, train_data, train_labels, test_data, test_labels):
        self.k_size = k_size
        self.p_size = p_size

        ### train & test : data and labels
        self.train_labels = train_labels
        self.train_data = train_data
        self.test_data = test_data
        self.test_labels = test_labels

        ### Utilize the distance function based on p_size
        if p_size == 1:
            self.distance_metric_function = l1_distance
        elif p_size == 2:
            self.distance_metric_function = l2_distance
        elif p_size == np.inf:
            self.distance_metric_function = linf_distance

    def train(self):
        predictions = []
        j = -1
        for j, test_point in enumerate(self.test_data):
            distances = []
            for i, train_point in enumerate(self.train_data):
                # Calculate the distance between the test point and each training point
                distance = self.distance_metric_function(test_point, train_point)
                distances.append((distance, self.train_labels[i], (i, j)))

            # Sort the distances in ascending order
            distances.sort(key=lambda x: x[0])

            # Select the k_size nearest neighbors
            neighbors = distances[:self.k_size]

            # Get the labels of the neighbors
            labels = [neighbor[1] for neighbor in neighbors]

            # Find the unique labels and their counts
            unique_labels, label_counts = np.unique(labels, return_counts=True)

            # Find the label with the highest count (majority label)
            majority_index = np.argmax(label_counts)
            majority_label = unique_labels[majority_index]

            # Append the majority label to the predictions list
            predictions.append(majority_label)

        return predictions
