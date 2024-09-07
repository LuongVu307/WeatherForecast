from models.model import load_model
from models.preprocessing import pipe0
from datetime import date, timedelta
import requests
import pandas as pd
import sys
import numpy as np
import csv


def download(start, end, key):
    # print(start, end- timedelta(days=10))

    response = requests.request("GET", f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/coventry/{start}/{end}?unitGroup=metric&include=days&key={key}&contentType=csv")
    if response.status_code!=200:
        print('Unexpected Status code: ', response.status_code)
        sys.exit()

    CSVText = csv.reader(response.text.splitlines(), delimiter=',',quotechar='"')
    local_file_path = f'data.csv'

    # Write the content of the response to the local file
    with open(local_file_path, 'w') as local_file:
        local_file.write(response.text)

def predict():
    end = date.today()
    start = end - timedelta(days=6)
    key = "95FET9DFCX93LKCKJHKDMSJHQ"

    if pd.read_csv("data.csv").datetime.to_list()[-1] != str(end):
        download(start, end, key)


    data = pd.read_csv("data.csv")
    model1, model2 = load_model("model1"), load_model("model2")
    pipe1, pipe2, scaler1, scaler2 = load_model("pipe1"), load_model("pipe2"), load_model("scaler1"), load_model("scaler2")
    data1, data2 = pipe0(data)

    columns = []

    prediction = []

    X1 = pipe1.fit_transform(data1)
    X1 = np.array(X1).reshape(1, -1)
    pred1 = model1.predict(X1).reshape(3, -1)

    columns1 = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'humidity',
                    'precip', 'windspeed', 'cloudcover', 'visibility', 'sunrise', 'sunset', 'moonphase']

    columns += columns1
    for i in range(3):
        prediction.append(scaler1.reversed(pred1[i]).tolist())


    X2 = pipe2.fit_transform(data2)
    X2 = np.array(X2).reshape(1, -1)
    pred2 = model2.predict(X2).reshape(3, -1)

    columns2 = ['windgust', 'uvindex']
    columns += columns2
    for i in range(3):
        prediction[i] += list(scaler2.reversed(pred2[i]))

    data = pd.DataFrame(prediction, columns=columns)
    data["sunset"] = data["sunset"]/24
    data["sunrise"] = data["sunrise"]/24

    # print(data)

    return data

def get_today():
    end = date.today()
    start = end - timedelta(days=6)
    key = "95FET9DFCX93LKCKJHKDMSJHQ"

    #https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/coventry/today?unitGroup=metric&include=current%2Chours&key=95FET9DFCX93LKCKJHKDMSJHQ&contentType=json
    if pd.read_csv("data.csv").datetime.to_list()[-1] != str(end):
        download(start, end, key)


    data = pd.read_csv("data.csv")

    # keep_cols = ["tempmin", "tempmax", "temp", "humidity", "precip", "windspeed", "cloudcover", "uvindex", "feelslikemax", "feelslikemin", "feelslike", "dew", "windgust", "sealevelpressure", "visibility", "sunrise", "sunset", "winddir"]
    keep_cols = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 
                 'humidity', 'precip', 'windspeed', 'cloudcover', 'visibility', 'sunrise', 
                 'sunset', 'moonphase', 'windgust', 'uvindex', 'dew']
    data["uvindex"] = data["uvindex"]/2
    # print("col", data[keep_cols].columns)

    weather = np.array(data[keep_cols].iloc[[-1]])


    # print(weather)
    return weather[0]

def get_history_data():
    data = pd.read_csv("data.csv")

    keep_cols = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 
                 'precip', 'windspeed','cloudcover', 'visibility', 'sunrise', 'sunset', 'moonphase', 
                 'windgust', 'uvindex']
    data["uvindex"] = data["uvindex"]/2

    data["sunset"] = pd.to_datetime(data["sunset"])
    data["sunrise"] = pd.to_datetime(data["sunrise"])

    convert_time = lambda x: (int(x.strftime("%H")) + int(x.strftime("%M"))/60)/24

    data["sunrise"] = data["sunrise"].apply(convert_time)
    data["sunset"] = data["sunset"].apply(convert_time)
    return data[keep_cols]