import numpy as np

class SimpleImputer:
    def __init__(self, strategy, columns=None):
        # Initializes the SimpleImputer with the strategy ('mean', 'median', 'most_frequent') and the optional columns to apply 
        # the transformation
        self.strategy = strategy
        self.columns = columns

    def fit(self, X):
        X_copy = X.copy()
        # Based on the strategy, compute the statistics (mean, median, or most frequent) for each column
        if self.strategy == 'mean':
            self.statistics_ = np.mean(X_copy, axis=0)  # Calculate mean for each column
        elif self.strategy == 'median':
            self.statistics_ = X_copy.median()  # Calculate median for each column
        elif self.strategy == 'most_frequent':
            self.statistics_ = X_copy.mode().iloc[0]  # Get the most frequent value for each column

    def transform(self, X):
        # Fill missing values in the specified columns with the computed statistics
        for column in self.columns:
            if column in X.columns:
                # Replace missing values (NaNs) in the column with the calculated statistic (mean/median/frequent value)
                X[column].fillna(self.statistics_[column], inplace=True)
            else:
                print(f"Column {column} is not in data")  # Warn if the column is not present in the data

        return X
    
    def fit_transform(self, X):
        # Fit and transform the data in one step
        X_copy = X.copy()
        self.fit(X_copy)
        return self.transform(X_copy)
    

class KNNImputer:
    def __init__(self, k, sample_size, ignore, type="euclidean"):
        # Initializes KNNImputer with the number of neighbors (k), sample size, columns to ignore, and distance metric 
        # (euclidean or manhattan)
        self.k = k
        self.size = sample_size
        self.type = type
        self.ignore = ignore

    def fit(self, X):
        # Prepare the training data by removing rows with missing values in columns not specified for ignoring
        cols = set(X.columns)
        cols -= set(self.ignore)  # Remove the columns specified for ignoring from the list of columns to consider
        self.cols = list(cols)  # Save the columns that will be used for prediction
        self.X_train = X[self.cols]  # Use only the relevant columns for the training set

    def transform(self, X):
        # Impute missing values in the dataset using KNN
        save = X  # Keep a copy of the original dataframe
        X_copy = X[self.cols]  # Focus on the relevant columns
        X_copy = X_copy[X_copy.isna().any(axis=1)]  # Filter rows with missing values
        missing_rows = X_copy.index  # Get the indices of rows with missing values

        # For each row with missing values, predict the missing values based on the k-nearest neighbors
        for row in missing_rows:
            temp = X_copy.loc[[row]]  # Select the row with missing values
            missing_cols = temp.columns[temp.isna().any()].to_list()  # Get columns with missing values
            pred = self.predict(temp, missing_cols)  # Predict missing values based on k-NN
            # print("TEMMP: \n", temp, "\n")

            # Update the original dataframe with the predicted values
            for col in missing_cols:
                save.loc[row, col] = pred[col]
                
        return save

    def predict(self, X, cols):

        # Predict missing values by calculating the distances between rows
        neighbors = self.calculate_distance(X)
        prediction = {}
        
        for col in cols:
            final_neighbor = []

            
            valid_indices = set(self.X_train[col].dropna().index)
            final_neighbor.extend(neighbor for neighbor in neighbors if neighbor in valid_indices)
            final_neighbor = final_neighbor[:min(self.k, len(final_neighbor))]

            prediction[col] = np.mean(self.X_train.loc[final_neighbor, col].values)

        return prediction
        
    def calculate_distance(self, X):
        # Calculate distances between the input data (X) and the training data (X_train)

        X_train_processed = self.X_train.drop(X.index)
        X = X[self.cols]  # Focus on the relevant columns
        X = X.to_numpy()  # Convert to numpy array for distance calculations
        if type(self.size) == int: 
            sample = X_train_processed.sample(n=min(self.size, len(X_train_processed)), replace=False)  # Randomly sample from the training data
        elif type(self.size) == float:
            sample = X_train_processed.sample(frac=min(self.size, 1), replace=False)  # Sample a fraction of the data
        index = sample.index.to_numpy()  # Get the indices of the sampled rows
        value = sample.to_numpy()  # Get the values of the sampled rows

        # Calculate the distance between the input data (X) and the sampled data
        if self.type == "manhattan":
            distance = np.nansum(np.abs(value - X), axis=1)  # Manhattan distance
        elif self.type == "euclidean":
            # print(np.nansum((value - X) ** 2, axis=1))
            distance = np.sqrt(np.nansum((value - X) ** 2, axis=1))  # Euclidean distance

        return index[np.argsort(distance)]

    def fit_transform(self, X):
        # Fit and transform the data in one step using KNN imputation
        X_copy = X.copy()
        self.fit(X_copy)
        X = self.transform(X_copy)
        return X
