import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


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
alpha_lasso = 10**np.linspace(-5,0,100)
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
plt.show()

# K-cross
alpha_lasso = 10 ** np.linspace(-4, 0, 100)
cv_errors = []
kf = KFold(n_splits=2, shuffle=True,random_state=42)
for alpha in alpha_lasso:
    lasso = Lasso(alpha=alpha)
    fold_errors = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        lasso.fit(X_train_fold, y_train_fold)
        y_pred = lasso.predict(X_val_fold)
        fold_error = mean_squared_error(y_val_fold, y_pred)
        fold_errors.append(fold_error)

    

best_alpha_index = np.argmin(fold_errors)
best_alpha = alpha_lasso[best_alpha_index]
best_cv_error = np.min(fold_errors)

print("Best alpha:", best_alpha)
print("Best CV error:", best_cv_error)


# use best LASSO
lasso = Lasso(alpha=best_alpha)
lasso.fit(X_train, y_train)

coef_abs = np.abs(lasso.coef_)
feature_importance = coef_abs / np.max(coef_abs) 

plt.figure(figsize=(10, 6))
plt.bar(X_train.columns, feature_importance)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance based on Lasso Regression (alpha={})'.format(best_alpha))
plt.xticks(rotation=90)
plt.show()


