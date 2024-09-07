import numpy as np
import pandas as pd


class SimpleImputer:
    def __init__(self, strategy, columns=None):
        self.strategy = strategy
        self.columns = columns

    def fit(self, X):
        X_copy = X.copy()
        if self.strategy == 'mean':
            self.statistics_ = X_copy.mean()
        elif self.strategy == 'median':
            self.statistics_ = X_copy.median()
        elif self.strategy == 'most_frequent':
            self.statistics_ = X_copy.mode().iloc[0]

    def transform(self, X):
        for column in self.columns:
            if column in X.columns:
                X[column].fillna(self.statistics_[column], inplace=True)
            else:
                print(f"Column {column} is not in data")

        return X
    
    def fit_transform(self, X):
        X_copy = X.copy()
        self.fit(X_copy)
        return self.transform(X_copy)
    

class KNNImputer:
    def __init__(self, k, sample_size, ignore, type="euclidean"):
        self.k = k
        self.size = sample_size
        self.type = type
        self.ignore = ignore


    def fit(self, X):
        print(f"Number of rows before: {len(X)}")
        cols = set(X.columns)
        cols -= set(self.ignore)
        self.X_train = X.dropna(subset=cols)
        self.cols = list(cols)

        self.X_train = self.X_train[self.cols]
        print(f"Number of available rows: ", len(self.X_train))

    def transform(self, X):
        save = X
        X_copy = X[self.cols]
        X_copy = X_copy[X_copy.isna().any(axis=1)]
        missing_rows = X_copy.index

        for row in missing_rows:
            temp = X_copy.loc[[row]]
            missing_cols = temp.columns[temp.isna().any()].to_list()
            pred = self.predict(temp, missing_cols)
            # print(len(pred), len(missing_cols))

            save.loc[row, pred.index] = pred
            # print(X.loc[row, missing_cols])

        return save

    def predict(self, X, cols):
        neighbors = self.calculate_distance(X)
        return neighbors[cols]

    def calculate_distance(self, X):
        X = X[self.cols]
        X = X.to_numpy()
        if type(self.size) == int: 
            sample = self.X_train.sample(n=self.size, replace=True)
        elif type(self.size) == float:
            sample = self.X_train.sample(frac=self.size, replace=True)
        index = sample.index.to_numpy()
        value = sample.to_numpy()

        if self.type == "euclidean":
            distance = np.nansum(np.abs(value-X), axis=1)
        elif self.type == "manhattan":
            distance = np.nansum(np.sqrt(value**2 - X**2), axis=1)
        k_neighbors = self.X_train.loc[index[np.argsort(distance)[:self.k]]]

        return k_neighbors.mean().round(8)


    def fit_transform(self, X):
        X_copy = X.copy()
        self.fit(X_copy)
        X = self.transform(X_copy)
        return X