import pandas as pd
import numpy as np


class IQRRemover:
    def __init__(self, alpha, columns):
        self.alpha = alpha
        self.columns = columns

    def fit(self, X):
        self.Q1 = X.quantile(0.25)
        self.Q3 = X.quantile(0.75)

        self.IQR = self.Q3 - self.Q1

    def transform(self, X):
        lower_bound = self.Q1 - self.alpha * self.IQR
        upper_bound = self.Q3 + self.alpha * self.IQR

        filtered = X[(X >= lower_bound) & (X <= upper_bound)]

        return filtered

    def fit_transform(self, X):
        X_copy = X.copy()

        self.fit(X_copy[self.columns])
        X_copy[self.columns] = self.transform(X_copy[self.columns])

        return X_copy

class ZscoreRemover:
    def __init__(self, threshold, columns):
        self.threshold = threshold
        self.columns = columns


    def fit(self, X):
        self.mean = X.mean()
        self.std = X.std()

        self.z_score = (X - self.mean) / self.std

    def transform(self, X):
        
        return X[np.abs(self.z_score) <= self.threshold]
    
    def fit_transform(self, X):
        X_copy = X.copy()

        self.fit(X_copy[self.columns])
        X_copy[self.columns] = self.transform(X_copy[self.columns])

        return X_copy

