import pandas as pd
from tools.encode import OneHotEncoder, BinaryEncoder

# Function to convert time in 'HH:MM' format to total hours
def convert_time(time_str):
    # Split the time string into hours and minutes, then convert to integer
    hours, minutes = map(int, time_str.split(':'))
    
    # Calculate the total number of hours (including fractional part for minutes)
    total_hour = hours + minutes/60
    
    return total_hour


# Function to process and clean the input data
def pipe0(data):
    # Set 'datetime' as the index of the dataframe
    data.set_index("datetime", inplace=True)

    # Define a list of columns to keep in the data
    keep_cols = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin',
                 'feelslike', 'dew', 'humidity', 'precip', 'precipcover',
                 'windgust', 'windspeed', 'winddir',
                 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
                 'solarenergy', 'uvindex', 'sunrise', 'sunset',
                 'moonphase', 'conditions']

    # Keep only the specified columns
    data = data[keep_cols]

    # Convert 'sunrise' and 'sunset' columns to datetime format and extract time in 'HH:MM'
    data["sunrise"] = pd.to_datetime(data.sunrise)
    data["sunrise"] = data["sunrise"].dt.strftime('%H:%M')
    data["sunset"] = pd.to_datetime(data.sunset)
    data["sunset"] = data["sunset"].dt.strftime('%H:%M')

    # Convert 'sunrise' and 'sunset' times to total hours
    data['sunrise'] = data['sunrise'].apply(convert_time)
    data['sunset'] = data['sunset'].apply(convert_time)

    # List of columns to convert to float
    int_col = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin',
               'feelslike', 'dew', 'humidity', 'precip', 'precipcover',
               'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
               'sunrise', 'sunset', 'moonphase', 'windgust', 'solarradiation', 'solarenergy', 'uvindex']

    # Convert the specified columns to float type
    data[int_col] = data[int_col].astype(float)
    
    # List of columns to drop
    drop_col = ["windgust", "solarradiation", "solarenergy", "uvindex"]

    # Create two datasets: one with the dropped columns and one with the full data
    data1 = data.drop(columns=drop_col)
    data2 = data

    # Return both datasets
    return data1, data2


# Define a Pipeline class for data processing and transformation
class Pipeline:
    def __init__(self, scaler, imputer, remover):
        # Initialize with a scaler, imputer, and column remover
        self.scaler = scaler
        self.imputer = imputer
        self.remover = remover

    # Method to fit the pipeline to the data
    def fit(self, X):
        binned = X.copy()
        
        # One-hot encode 'conditions' column with predefined categories
        encoded = OneHotEncoder(drop=True).transform(data=binned, columns=["conditions"], type_encode="multicategory", 
                                                     list_split=['Clear', 'Fog', 'Overcast', 'Partially cloudy', 'Rain', 'Snow'], split_punc=", ")
        
        # Apply binary encoding on the 'precip' column
        encoded = BinaryEncoder().transform(data=encoded, columns=["precip"], values=[0])

        # Add 'day' and 'month' features based on the datetime index
        encoded["date"] = encoded.index
        encoded["date"] = pd.to_datetime(encoded["date"])
        encoded["day"] = encoded["date"].dt.day
        encoded["month"] = encoded["date"].dt.month + encoded.day/30
        
        # Drop the original 'date' column
        encoded.drop(columns=["date"], inplace=True)
        
        # Store the encoded data
        self.encoded = encoded

    # Method to fit and transform the data
    def fit_transform(self, X):
        self.fit(X)

        # Apply feature removal, imputation, and scaling in sequence
        filtered = self.remover.fit_transform(self.encoded)
        imputed = self.imputer.fit_transform(filtered)
        scaled = self.scaler.fit_transform(imputed)
        
        # Ensure columns are sorted after scaling
        scaled = scaled.reindex(sorted(scaled.columns), axis=1)

        # Return the processed and scaled data
        return scaled

    # Method to transform the data using the fitted pipeline
    def transform(self, X):
        self.fit(X)

        # Apply feature removal, imputation, and scaling in sequence without fitting
        filtered = self.remover.transform(self.encoded)
        imputed = self.imputer.transform(filtered)
        scaled = self.scaler.transform(imputed)

        # Return the transformed data
        return scaled
