from predict_temp_func import predict_temp
#import data

cities = ['Sydney','Melbourne']
for city in cities:
    mae_nn, rmse_nn, accuracy_nn = predict_temp(city)
    print(f"{city} - MAE: {mae_nn:.2f}")
    print(f"{city} - RMSE: {rmse_nn:.2f}")
    print(f"{city} - Accuracy: {accuracy_nn:.2f}%")
