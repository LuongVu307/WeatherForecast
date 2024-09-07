import pandas as pd
from tools.encode import OneHotEncoder, BinaryEncoder

def convert_time(time_str):
    # Split the time string into hours, minutes, and seconds
    hours, minutes = map(int, time_str.split(':'))
    
    # Calculate the total number of seconds since midnight
    total_hour = hours + minutes/60
    
    return total_hour


def pipe0(data):
    data.set_index("datetime", inplace=True)
    keep_cols = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin',
       'feelslike', 'dew', 'humidity', 'precip', 'precipcover',
       'windgust', 'windspeed', 'winddir',
       'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
       'solarenergy', 'uvindex', 'sunrise', 'sunset',
       'moonphase', 'conditions']

    data = data[keep_cols]

    data["sunrise"] = pd.to_datetime(data.sunrise)
    data["sunrise"] = data["sunrise"].dt.strftime('%H:%M')
    data["sunset"] = pd.to_datetime(data.sunset)
    data["sunset"] = data["sunset"].dt.strftime('%H:%M')

    data['sunrise'] = data['sunrise'].apply(convert_time)
    data['sunset'] = data['sunset'].apply(convert_time)

    int_col = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin',
       'feelslike', 'dew', 'humidity', 'precip', 'precipcover',
       'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
       'sunrise', 'sunset', 'moonphase', 'windgust', 'solarradiation', 'solarenergy', 'uvindex']

    data[int_col] = data[int_col].astype(float)
    
    drop_col = ["windgust", "solarradiation", "solarenergy", "uvindex"]

    data1 = data.drop(columns=drop_col)
    data2 = data


    return data1, data2



class Pipeline:
    def __init__(self, scaler, imputer, remover):
        self.scaler = scaler
        self.imputer = imputer
        self.remover = remover

    def fit(self, X):
        binned = X.copy()
        # binned["winddir_bin"] = pd.cut(binned["winddir"], bins=[-1, 45, 90, 135, 180, 225, 270, 315, 360], labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        # binned.drop(columns=["winddir"], inplace=True)
        
        encoded = OneHotEncoder(drop=True).transform(data=binned, columns=["conditions"], type_encode="multicategory", list_split=['Clear', 'Fog', 'Overcast', 'Partially cloudy', 'Rain', 'Snow'], split_punc=", ")
        # encoded = OneHotEncoder(drop=True).transform(data=encoded, columns=["winddir_bin"], type_encode="category")
        encoded = BinaryEncoder().transform(data=encoded, columns=["precip"], values=[0])

        encoded["date"] = encoded.index
        encoded["date"] = pd.to_datetime(encoded["date"])
        encoded["day"] = encoded["date"].dt.day
        encoded["month"] = encoded["date"].dt.month + encoded.day/30
        # # print(encoded.month)
        # encoded["spring"] = ((encoded["month"] == 1) | (encoded["month"] == 2) | (encoded["month"] == 3)).astype(int)
        # encoded["summer"] = ((encoded["month"] == 4) | (encoded["month"] == 5) | (encoded["month"] == 6)).astype(int)
        # encoded["autumn"] = ((encoded["month"] == 7) | (encoded["month"] == 8) | (encoded["month"] == 9)).astype(int)
        # encoded["winter"] = ((encoded["month"] == 10) | (encoded["month"] == 11) | (encoded["month"] == 12)).astype(int)
        
        encoded.drop(columns=["date"], inplace=True)
        
        self.encoded = encoded

    def fit_transform(self, X):
        self.fit(X)
        # print(encoded.columns)
        # self.dropping = self.encoded.nunique()[self.encoded.nunique() <= 1].index.tolist()

        # self.encoded.drop(columns=self.dropping, inplace=True)

        
        filtered = self.remover.fit_transform(self.encoded)

        imputed = self.imputer.fit_transform(filtered)
        
        scaled = self.scaler.fit_transform(imputed)
        # print(scaled.month)
        # return scaled
        # processed = self.PCA.fit_transform(scaled)

        # print(self.PCA.components.shape)
        # print(scaled.shape)
        scaled = scaled.reindex(sorted(scaled.columns), axis=1)
        # print(scaled.columns)


        return scaled


    def transform(self, X):
        self.fit(X)

        # self.encoded.drop(columns=self.dropping, inplace=True)

        # print(self.encoded.columns)
        filtered = self.remover.transform(self.encoded)

        imputed = self.imputer.transform(filtered)
        scaled = self.scaler.transform(imputed)
        # return scaled
        # print(self.PCA.components.shape)
        # print(scaled.shape)
        # processed = self.PCA.transform(scaled)

        return scaled
        
