import pandas as pd

data_9am = pd.read_csv("weathernoNA.csv")
data_9am = data_9am[['Date','Location','Rainfall','Evaporation','Sunshine','WindGustDir',
                     'WindGustSpeed','WindDir9am','WindSpeed9am','Humidity9am','Pressure9am',
                     'Cloud9am','Temp9am','RainToday']]
data_9am.to_csv("weather9am.csv", index=False)
print(data_9am)
