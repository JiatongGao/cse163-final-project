from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


df = pd.read_csv('weather3pm')

city = 'Melbourne'
data = df[df['Location'] == city][['Date', 'Evaporation','Rainfall', 'Pressure9am','Humidity9am', 'WindSpeed9am','Temp9am']]
features = data[['Evaporation', 'Rainfall','Pressure9am', 'Humidity9am', 'WindSpeed9am']].values
target = data[['Temp9am']].values

split_point = int(len(data)*0.8)
X_train, X_test = features[:split_point], features[split_point:]
y_train, y_test = target[:split_point], target[split_point:]

class TemperaturePredict(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TemperaturePredict, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        predicted_output = self.fc3(output)
        return predicted_output

