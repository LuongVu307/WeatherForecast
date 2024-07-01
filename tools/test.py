import pandas as pd
from os.path import join
import os 
from encode import OneHotEncoder

MAIN_DIR = "D:/Projects/WeatherForecast/dataset2"


data = pd.DataFrame()
for file in os.listdir(MAIN_DIR):
    if file.endswith(".csv"):
        try:
            file_path = join(MAIN_DIR, file)
            df = pd.read_csv(file_path, index_col="datetime")
            data = pd.concat([data, df])
        except Exception:
            print(file_path)
            sys.exit()
data = data.sort_index()
data.drop(columns="Unnamed: 0", inplace=True)

keep_cols = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin',
       'feelslike', 'dew', 'humidity', 'precip', 'precipprob', 'precipcover',
       'windgust', 'windspeed', 'winddir',
       'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
       'solarenergy', 'uvindex', 'severerisk', 'sunrise', 'sunset',
       'moonphase', 'conditions']

weather = data[keep_cols]


binned = weather.copy()
binned["winddir_bin"] = pd.cut(binned["winddir"], bins=[-1, 45, 90, 135, 180, 225, 270, 315, 360], labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

binned["winddir_bin"]
OneHotEncoder().transform(data=binned, columns=["winddir_bin"], type_encode="category")