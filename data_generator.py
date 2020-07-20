import numpy as np
from sklearn.model_selection import train_test_split


def generate_data(num_sample=100, num_feature=5, variance=2):
    """
    Generate data for regression and split to train and test
    return X_train, X_test, y_train, y_test
    """
    features = np.zeros((num_sample, num_feature))
    labels = np.arange(100)
    for i in range(num_sample):
        features[i, :] = np.random.uniform(i - variance, i + variance, num_feature)

    return train_test_split(features, labels, test_size=0.33, random_state=42)