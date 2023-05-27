import pandas as pd
df = pd.read_csv("weatherAUS.csv")
# Deal with NA
missing_values = df.isnull().sum() 
