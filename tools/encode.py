import numpy as np

class OneHotEncoder:
    def __init__(self, drop=False, min_frequency=None, max_categories=None):
        self.drop = drop
        self.min_frequency = min_frequency
        self.max_categories = max_categories


    def transform(self, data, columns, type_encode="category", list_split=None, split_punc=None):
        df = data.copy()
        length = len(df)
        if type_encode == "multicategory":
            for col in columns:
                if list_split != None:
                    for item in list_split:
                        name = f"{col}_{item.replace(' ', '')}"
                        df[name] = [0]*length

            for col in columns:
                for id, value in zip(df.index, df[col]):
                    items = value.split(split_punc)
                    for item in items:
                        name = f"{col}_{item.replace(' ', '')}"
                        df.loc[id, name] = 1


        if type_encode == "category":
            for col in columns:
                list_split = df[col].unique()
                for item in list_split:
                    name = f"{col}_{item.replace(' ', '')}"
                    df[name] = [0]*length

            for col in columns:
                for id, value in zip(df.index, df[col]):
                    name = f"{col}_{value.replace(' ', '')}"
                    df.loc[id, name] = 1

                    
        if self.drop == True:
            df.drop(columns=columns, inplace=True)

        return df



class BinaryEncoder:
    def __init__(self):
        pass

    def transform(self, data, columns, values):
        df = data.copy()
        length = len(df)

        for col, value in zip(columns, values):
            name = f"{col}_{value}"
            df[name] = [0]*length

            for id, item in zip(df.index, df[col]):
                if value == item:
                    df.loc[id, name] = 1

        return df