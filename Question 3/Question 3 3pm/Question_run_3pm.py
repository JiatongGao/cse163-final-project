from Question_3_predict_3pm import predict_temp_3pm
import csv
cities = ['Sydney', 'Melbourne', 'Brisbane',
          'Perth', 'Canberra', 'Adelaide']
data_3pm = []
for city in cities:
    mae_ensemble, r2_ensemble, accuracy_ensemble = predict_temp_3pm(city)
    data_3pm.append([mae_ensemble, r2_ensemble, accuracy_ensemble])

csv_file = 'stat_data_3pm.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Cities', 'MAE', 'R2', 'Accuracy'])
    for i in range(len(cities)):
        writer.writerow([cities[i]] + data_3pm[i])
