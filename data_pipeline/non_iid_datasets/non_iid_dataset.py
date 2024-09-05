import numpy as np

class NonIIDDatasets:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def generate_non_iid_data(self, num_samples, num_features):
        # Generate a random dataset with the specified number of samples and features
        X = np.random.rand(num_samples, num_features)  # Ensure both arguments are integers
        y = np.random.randint(0, 2, size=(num_samples,))

        # Split the data into training and testing sets
        from sklearn.model_selection import train_test_split

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

        return train_X, test_X, train_y, test_y