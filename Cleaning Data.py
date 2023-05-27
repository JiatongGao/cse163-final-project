import pandas as pd
import numpy as np
df = pd.read_csv("weatherAUS.csv")
# Deal with NA
# Replace NaN with the mean value of that column
missing_values = df.isnull().sum()
print(missing_values)
df_num = df.apply(pd.to_numeric, errors='coerce')
df_num = df_num.fillna(df_num.mean())
df_num[['Date','Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']] =\
df[['Date','Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']]
df = df_num
df['Date'] = pd.to_datetime(df['Date'])# Convert date form
print(df)

