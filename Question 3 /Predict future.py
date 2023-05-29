import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import data
df = pd.read_csv("weatherAUS.csv")
# Deal with NA
# Replace NaN with the mean value of that column
missing_values = df.isnull().sum()
df_num = df.apply(pd.to_numeric, errors='coerce')
df_num = df_num.fillna(df_num.mean())
df_num[['Date','Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']] =\
df[['Date','Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']]
df = df_num
df['Date'] = pd.to_datetime(df['Date'])# Convert date form
city = 'Albury'
df = df[(df['Location'] == city)]

###############

data = df[['Date','Evaporation','Sunshine','Pressure9am','Humidity9am','Temp9am']]
# Extract features and target
features = data[['Evaporation','Sunshine','Pressure9am','Humidity9am']].values
target = data[['Temp9am']].values
# standarized 
scaler = StandardScaler()
features = scaler.fit_transform(features)
print(features)
# divide train and test (20% test and 80% target)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# convert to tensor
X_train = torch.tensor(X_train, dtype= torch.float32)
X_test = torch.tensor(X_test, dtype= torch.float32)
y_train = torch.tensor(y_train, dtype= torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


class TemperaturePredict(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TemperaturePredict, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc1(output)
        output = self.relu(output)
        predicted_output = self.fc2(output)
        return predicted_output

# An example
input_size = X_train.shape[1]
hidden_size = 128
num_layers = 3

model = TemperaturePredict(input_size, hidden_size, num_layers)

# loss
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for i in range(0, len(X_train), batch_size):
        batch_inputs = X_train[i:i+batch_size]
        batch_labels = y_train[i:i+batch_size]

        optimizer.zero_grad()
        
        outputs = model(batch_inputs.unsqueeze(1))
        loss = criterion(outputs.squeeze(), batch_labels.squeeze())
        
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.unsqueeze(1))

predictions = test_outputs.squeeze().numpy()
true_labels = y_test.numpy()

comparison = np.column_stack((predictions, true_labels))

# Make a csv

df_comparison = pd.DataFrame(comparison, columns=['Predictions', 'True Labels'])


# Make a plot
x = range(len(predictions))


plt.plot(x, predictions, label='Predictions')
plt.plot(x, true_labels, label='True Labels')


plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Temperature')



# test
absolute_errors = np.abs(predictions - true_labels)
mae = np.mean(absolute_errors)
rmse = np.sqrt(np.mean(absolute_errors ** 2))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")