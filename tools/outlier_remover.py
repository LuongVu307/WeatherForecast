import numpy as np


class IQRRemover:
    def __init__(self, alpha, columns):
        # Initialize the IQRRemover with the threshold multiplier (alpha) and the columns to apply the transformation
        self.alpha = alpha
        self.columns = columns

    def fit(self, X):
        # Compute the first (Q1) and third (Q3) quartiles for each specified column
        self.Q1 = X.quantile(0.25)  # Calculate the 25th percentile (Q1)
        self.Q3 = X.quantile(0.75)  # Calculate the 75th percentile (Q3)

        # Calculate the interquartile range (IQR) as the difference between Q3 and Q1
        self.IQR = self.Q3 - self.Q1

    def transform(self, X):
        # Filter the data by removing outliers beyond the lower and upper bounds determined by IQR and alpha
        lower_bound = self.Q1 - self.alpha * self.IQR  # Lower bound for outliers
        upper_bound = self.Q3 + self.alpha * self.IQR  # Upper bound for outliers

        # Keep only the values within the bounds
        filtered = X[(X >= lower_bound) & (X <= upper_bound)]

        return filtered

    def fit_transform(self, X):
        # Apply the fit and transform operations in one step
        X_copy = X.copy()  # Make a copy of the data

        # Fit the model (compute Q1, Q3, and IQR) and transform the specified columns
        self.fit(X_copy[self.columns])
        X_copy[self.columns] = self.transform(X_copy[self.columns])

        return X_copy


class ZscoreRemover:
    def __init__(self, threshold, columns):
        # Initialize the ZscoreRemover with the threshold value and the columns to apply the transformation
        self.threshold = threshold
        self.columns = columns

    def fit(self, X):
        # Calculate the mean and standard deviation for each specified column
        self.mean = X.mean()  # Mean of the data
        self.std = X.std()  # Standard deviation of the data

        # Compute the Z-score for each data point in the columns
        self.z_score = (X - self.mean) / self.std

    def transform(self, X):
        # Filter out values where the absolute Z-score exceeds the threshold
        return X[np.abs(self.z_score) <= self.threshold]  # Keep values with Z-score less than or equal to the threshold
    
    def fit_transform(self, X):
        # Apply the fit and transform operations in one step
        X_copy = X.copy()  # Make a copy of the data

        # Fit the model (compute mean, std, and Z-scores) and transform the specified columns
        self.fit(X_copy[self.columns])
        X_copy[self.columns] = self.transform(X_copy[self.columns])

        return X_copy
