import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("weathernoNA.csv")
# Split the data into features (X) and target variable (y)
data = df.loc[:, ['Temp9am', 'Rainfall', 'Evaporation',
                  'Sunshine', 'WindGustSpeed', 'Humidity9am',
                  'WindSpeed9am', 'Pressure9am', 'Cloud9am',]]

X = data.drop('Temp9am', axis=1)  # Features matrix
y = data['Temp9am']  # Target variable

split_point = int(len(data)*0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the feature importance scores
feature_importance = np.abs(model.coef_)

# Create a DataFrame to store the factors and their importance scores
importance_df = pd.DataFrame({'Factor': X.columns,
                              'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the factors and their importance scores
print(importance_df)

# Determine the factor with the strongest influence
strongest_factor = importance_df['Factor'].iloc[0]
print(strongest_factor)

# plot the pie plot
fig, ax = plt.subplots(figsize=(8, 6))
explode = [0.1] + [0] * (len(importance_df) - 1)
# Explode the first slice for emphasis
colors = plt.cm.Set3(np.linspace(0, 1, len(importance_df)))
# Generate a color map for the slices
ax.pie(importance_df['Importance'],
       labels=importance_df['Factor'], explode=explode,
       colors=colors, autopct='%1.1f%%', shadow='True', startangle=45)
ax.set_title('Correlation of factors and temperature at 9 a.m.', color='red')
ax.legend(loc='best')
ax.axis('equal')  # Ensure a circular pie chart
plt.tight_layout()  # Adjust spacing between subplots
plt.savefig('factors9am.png', bbox_inches='tight')
