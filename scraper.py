import pandas as pd
import requests
import sys
import os
import csv
import codecs


while True:
  list_data = list(os.listdir("dataset2"))

  year = int(min(list_data)[:-4]) -1 
          
  response = requests.request("GET", f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/coventry/{year}-01-01/{year}-12-31?unitGroup=metric&include=days&key=95FET9DFCX93LKCKJHKDMSJHQ&contentType=csv")
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
  df.to_csv(f"dataset2/{year}.csv")
  df
