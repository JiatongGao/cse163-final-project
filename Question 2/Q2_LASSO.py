import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# 读取数据集
data = pd.read_csv('weather9am.csv')

encoder = LabelEncoder()
data['Location'] = encoder.fit_transform(data['Location'])
data['WindGustDir'] = encoder.fit_transform(data['WindGustDir'])
data['WindDir9am'] = encoder.fit_transform(data['WindDir9am'])
data['RainToday'] = encoder.fit_transform(data['RainToday'])
data['RainTomorrow'] = encoder.fit_transform(data['RainTomorrow'])

scaler = MinMaxScaler()
numeric_features = ['Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
                    'WindSpeed9am','Humidity9am','Pressure9am','Cloud9am','Temp9am']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

features = data.drop(['Date', 'Temp9am'], axis=1)
target = data['Temp9am']
split_point = int(len(data)*0.8)
X_train, X_test = features[:split_point], features[split_point:]
y_train, y_test = target[:split_point], target[split_point:]

#select an alpha
alpha_lasso = 10**np.linspace(-3,1,100)
lasso = Lasso()
coefs_lasso = []

for i in alpha_lasso:
    lasso.set_params(alpha = i)
    lasso.fit(X_train, y_train)
    coefs_lasso.append(lasso.coef_)
    
plt.figure(figsize=(12,10))
ax = plt.gca()
ax.plot(alpha_lasso, coefs_lasso)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights: scaled coefficients')
plt.title('Lasso regression coefficients Vs. alpha')
plt.legend(data.drop('Temp9am',axis=1, inplace=False).columns)
plt.show()