import pandas as pd
import requests
import sys
import os
import csv


while True:
  list_data = list(os.listdir("data"))

  year = int(min(list_data)[:-4]) -1 
  key = "" #Input the key
          
  response = requests.request("GET", f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/coventry/{year}-01-01/{year}-12-31?unitGroup=metric&include=days&key={key}&contentType=csv")
  if response.status_code!=200:
    print('Unexpected Status code: ', response.status_code)
    sys.exit()
  

  # Parse the results as CSV
  CSVText = csv.reader(response.text.splitlines(), delimiter=',',quotechar='"')
          

  data = list(CSVText)

  header = data[0]
  rows = data[1:]

  # Create DataFrame
  df = pd.DataFrame(rows, columns=header)

  # Display the DataFrame (optional)
  df.to_csv(f"data/{year}.csv")
  df
