import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score



data = pd.read_csv('weather9am.csv')

encoder = LabelEncoder()
data['Location'] = encoder.fit_transform(data['Location'])
data['WindGustDir'] = encoder.fit_transform(data['WindGustDir'])
data['WindDir9am'] = encoder.fit_transform(data['WindDir9am'])
data['RainToday'] = encoder.fit_transform(data['RainToday'])


scaler = MinMaxScaler()
numeric_features = ['Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
                    'WindSpeed9am','Humidity9am','Pressure9am','Cloud9am','Temp9am']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

features = data.drop(['Date','Location','Temp9am'], axis=1)
target = data['Temp9am']
split_point = int(len(data)*0.8)
X_train, X_test = features[:split_point], features[split_point:]
y_train, y_test = target[:split_point], target[split_point:]

#select an alpha
alpha_lasso = 10**np.linspace(-4,0,100)
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
plt.legend(features.columns)
plt.savefig('alpha_graph_9am.png', bbox_inches='tight')

coefs_list = []
for i in alpha_lasso:
    lasso = Lasso(alpha = i)
    model_lasso = lasso.fit(X_train, y_train)
    coef = pd.Series(model_lasso.coef_,index=X_train.columns)
    coefs_list.append(coef[coef != 0].abs().sort_values(ascending=False))

coef_table = pd.concat(coefs_list, axis=1)
coef_table.columns = alpha_lasso
coef_table.to_csv('alpha_coef_table_9am.csv', index=True)
fea = X_train.columns
a = pd.DataFrame()
a['feature'] = fea
a['importance'] = coef.values

a = a.sort_values('importance',ascending = False)
plt.figure(figsize=(12,8))
plt.barh(a['feature'],a['importance'])
plt.title('the importance features')
plt.savefig('importance_of_feature_9am.png', bbox_inches='tight')


