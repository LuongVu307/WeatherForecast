import numpy as np
import pandas as pd

class StandardScaler():
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X):
        self.mean = X.mean()
        self.std = X.std()


    def transform(self, X):
        # scaling = lambda data: (data-self.mean) / self.std
        X = ((X-self.mean) /self.std)

        return X
    
    def reversed(self, X):
        X = X*self.std + self.mean
        return X

    def fit_transform(self, X):
        X_copy = X

        self.fit(X_copy[self.columns])
        # print(self.transform(X_copy[self.columns]))
        X_copy[self.columns] = self.transform(X_copy[self.columns])

        return X_copy

class MinMaxScaler():
    def __init__(self, columns):
        self.columns = columns

    def fit_transform(self, X):
        X_copy = X.copy()

        self.fit(X_copy[self.columns])
        # print(self.transform(X_copy[self.columns]))
        X_copy[self.columns] = self.transform(X_copy[self.columns])

        return X_copy

    def fit(self, X):
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        self.data_range_[self.data_range_ == 0] = 1
        
        self.scale_ = 1 / self.data_range_
        self.min_ = - self.data_min_ * self.scale_

    def transform(self, X):
        return X * self.scale_ + self.min_

    def reversed(self, X_scaled):
        return (X_scaled - self.min_) / self.scale_

