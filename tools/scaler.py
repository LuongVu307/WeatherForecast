import numpy as np

class StandardScaler():
    def __init__(self, columns):
        # Initialize the StandardScaler with the columns to be scaled
        self.columns = columns

    def fit(self, X):
        # Compute the mean and standard deviation for each specified column
        self.mean = np.mean(X, axis=0)  # Mean of the data
        self.std = np.std(X, axis=0)  # Standard deviation of the data

    def transform(self, X):
        # Standardize the data by subtracting the mean and dividing by the standard deviation
        X = ((X - self.mean) / self.std)

        return X
    
    def reversed(self, X):
        # Reverse the standardization by multiplying by the standard deviation and adding the mean
        X = X * self.std + self.mean
        return X

    def fit_transform(self, X):
        # Apply the fit and transform operations in one step
        X_copy = X.copy()  # Make a copy of the data

        # Fit the model (compute mean and std) and transform the specified columns
        self.fit(X_copy[self.columns])
        X_copy[self.columns] = self.transform(X_copy[self.columns])

        return X_copy

class MinMaxScaler():
    def __init__(self, columns):
        # Initialize the MinMaxScaler with the columns to be scaled
        self.columns = columns

    def fit_transform(self, X):
        # Apply the fit and transform operations in one step
        X_copy = X.copy()  # Make a copy of the data

        # Fit the model (compute min and max values) and transform the specified columns
        self.fit(X_copy[self.columns])
        X_copy[self.columns] = self.transform(X_copy[self.columns])

        return X_copy

    def fit(self, X):
        # Calculate the minimum, maximum, and range for each specified column
        self.data_min_ = np.min(X, axis=0)  # Minimum values of the columns
        self.data_max_ = np.max(X, axis=0)  # Maximum values of the columns
        self.data_range_ = self.data_max_ - self.data_min_  # Range (max - min) for each column
        
        # To avoid division by zero, if range is zero, set it to 1
        self.data_range_[self.data_range_ == 0] = 1

        # Calculate the scale and minimum for transformation
        self.scale_ = 1 / self.data_range_  # Scaling factor
        self.min_ = - self.data_min_ * self.scale_  # Minimum offset for transformation

    def transform(self, X):
        # Scale the data to the range [0, 1] by multiplying by scale and adding min
        return X * self.scale_ + self.min_

    def reversed(self, X_scaled):
        # Reverse the scaling transformation by using the min and scale
        return (X_scaled - self.min_) / self.scale_
