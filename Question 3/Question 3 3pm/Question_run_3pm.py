
from Question_3_predict_3pm import predict_temp_3pm
import pandas as pd
import csv
cities = ['Sydney', 'Melbourne','Brisbane','Perth','Canberra','Adelaide']
data_9am = []
for city in cities:
    mae_ensemble, r2_ensemble, accuracy_ensemble = predict_temp_3pm(city)
    data_9am.append([mae_ensemble, r2_ensemble, accuracy_ensemble])

csv_file = 'stat_data_3pm.csv'
