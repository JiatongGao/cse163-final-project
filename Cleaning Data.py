import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# Split the data into features (X) and target variable (y)
min = df['MinTemp']
max = df['MaxTemp']
average = (min + max) / 2
df['Temperature'] = average
data = df.loc[:, ['Temperature', 'Rainfall', 'Evaporation','Sunshine', 'WindGustSpeed']]

print(data)
X = data.drop('Temperature', axis=1)  # Features matrix
y = data['Temperature']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the feature importance scores
feature_importance = np.abs(model.coef_)

# Create a DataFrame to store the factors and their importance scores
importance_df = pd.DataFrame({'Factor': X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the factors and their importance scores
print(importance_df)

# Determine the factor with the strongest influence
strongest_factor = importance_df['Factor'].iloc[0]
