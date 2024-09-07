import pandas as pd
from os.path import join
import os 
from encode import OneHotEncoder
import sys
from scaler import StandardScaler, MinMaxScaler
from impute import SimpleImputer, KNNImputer
from outlier_remover import IQRRemover, ZscoreRemover


import numpy as np
import pandas as pd

columns=[str(i) for i in range(5)]
random_array = np.random.rand(10, 5) * np.random.rand(10, 5) * 100

dct = {i : random_array[:, j] for i, j in zip(columns, range(random_array.shape[1]))}

df = pd.DataFrame(dct)
print(df)
scaler = StandardScaler(columns=columns) 
print(scaler.fit_transform(df))
print(df)
print(np.array(scaler.reversed(df)))