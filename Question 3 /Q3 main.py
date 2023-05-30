
from Question_3_predict_func import predict_temp
import pandas as pd
#import data
cities = ['Sydney','Melbourne']
for city in cities:
    mae_ensemble, rmse_ensemble, accuracy_ensemble  = predict_temp(city)
    print(f"{city} - MAE: {mae_ensemble:.2f}")
    print(f"{city} - RMSE: {rmse_ensemble:.2f}")
    print(f"{city} - Accuracy: {accuracy_ensemble:.2f}%")
