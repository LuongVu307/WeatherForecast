class OneHotEncoder:
    def __init__(self, drop=False, min_frequency=None, max_categories=None):
        # Initializes the OneHotEncoder with options for dropping columns, 
        # setting minimum frequency for categories, and limiting the number of categories
        self.drop = drop
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def transform(self, data, columns, type_encode="category", 
                  list_split=None, split_punc=None):
        df = data.copy()  # Create a copy of the input dataframe to avoid modifying the original data
        length = len(df)
        
        # One-hot encoding for multi-category columns (e.g., values with multiple categories in each cell)
        if type_encode == "multicategory":
            for col in columns:
                # Initialize new columns for each item in the list_split (if provided)
                if list_split != None:
                    for item in list_split:
                        name = f"{col}_{item.replace(' ', '')}"
                        df[name] = [0] * length  # Set initial values as 0

            # For each row, split the column values and mark the corresponding categories as 1
            for col in columns:
                for id, value in zip(df.index, df[col]):
                    items = value.split(split_punc)  # Split based on punctuation
                    for item in items:
                        name = f"{col}_{item.replace(' ', '')}"
                        df.loc[id, name] = 1  # Set the category column to 1

        # One-hot encoding for categorical columns (each value becomes its own column)
        if type_encode == "category":
            for col in columns:
                list_split = df[col].unique()  # Get unique values for the column
                for item in list_split:
                    name = f"{col}_{item.replace(' ', '')}"
                    df[name] = [0] * length  # Set initial values as 0

            # For each row, set the corresponding category column to 1
            for col in columns:
                for id, value in zip(df.index, df[col]):
                    name = f"{col}_{value.replace(' ', '')}"
                    df.loc[id, name] = 1

        # Drop the original columns if specified
        if self.drop == True:
            df.drop(columns=columns, inplace=True)

        return df  # Return the transformed dataframe


class BinaryEncoder:
    def __init__(self):
        pass

    def transform(self, data, columns, values):
        df = data.copy()  # Create a copy of the input dataframe to avoid modifying the original data
        length = len(df)

        # For each column and its corresponding value, create new binary columns
        for col, value in zip(columns, values):
            name = f"{col}_{value}"  # Construct the new column name
            df[name] = [0] * length  # Set initial values as 0

            # For each row, if the value matches, set the new column to 1
            for id, item in zip(df.index, df[col]):
                if value == item:
                    df.loc[id, name] = 1  # Mark the column as 1 for the matching value

        return df  # Return the transformed dataframe
