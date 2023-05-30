import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv("weathernoNA.csv")
def predict_temp(city):
    data = df[df['Location'] == city][['Date', 'Evaporation', 'Sunshine', 'Pressure9am', 'Humidity9am', 'Temp9am']]
    features = data[['Evaporation', 'Sunshine', 'Pressure9am', 'Humidity9am']].values
    target = data[['Temp9am']].values
    # standarized 
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # divide train and test (20% test and 80% target)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    # convert to tensor
    X_train = torch.tensor(X_train, dtype= torch.float32)
    X_test = torch.tensor(X_test, dtype= torch.float32)
    y_train = torch.tensor(y_train, dtype= torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

# data augment
    def data_augmentation(inputs, labels):
        augmented_inputs = torch.flip(inputs, dims=[0])
        augmented_labels = labels.clone()
        return torch.cat((inputs, augmented_inputs)), torch.cat((labels, augmented_labels))

    # Build the neutral network model
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

####
    input_size = X_train.shape[1]
    hidden_size = 128
    num_layers = 3
    weight_decay = 0.03

    model = TemperaturePredict(input_size, hidden_size, num_layers)

# loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

# train the model
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0.0
    
        for batch_inputs, batch_labels in train_loader:
            # data augmentation 
            batch_inputs, batch_labels = data_augmentation(batch_inputs, batch_labels)

            optimizer.zero_grad()
        
            outputs = model(batch_inputs.unsqueeze(1))
            loss = criterion(outputs.squeeze(), batch_labels.squeeze())
        
        ###
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss += weight_decay * l2_reg
        
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
    
    # Use neutral network to predict
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test.unsqueeze(1))

    predictions_nn = test_outputs.squeeze().numpy()
    true_labels = y_test.numpy()

    


# Make a csv

#df_comparison = pd.DataFrame(comparison, columns=['Predictions', 'True Labels'])


# Make a plot
#x = range(len(predictions))


#plt.plot(x, predictions, label='Predictions')
#plt.plot(x, true_labels, label='True Labels')


#plt.legend()
#plt.xlabel('Sample Index')
#plt.ylabel('Temperature')



# test
    absolute_errors = np.abs(predictions_nn - true_labels)
    mae_nn = np.mean(absolute_errors)
    rmse_nn = np.sqrt(np.mean(absolute_errors ** 2))
    accuracy_nn = 100 - (mae_nn / np.mean(true_labels)) * 100

    

    return mae_nn, rmse_nn, accuracy_nn