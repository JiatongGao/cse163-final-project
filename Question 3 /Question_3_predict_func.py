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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

def predict_temp(city):
    df = pd.read_csv("weathernoNA.csv")
    data = df[df['Location'] == city][['Date', 'Evaporation', 'Sunshine', 'Pressure9am', 'Humidity9am', 'Temp9am']]
    features = data[['Evaporation', 'Sunshine', 'Pressure9am', 'Humidity9am']].values
    target = data[['Temp9am']].values
    # standarized 
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # divide train and test (20% test and 80% target)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    # convert to tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)
    y_train_np = np.array(y_train)
    

    # data augment
    def data_augmentation(inputs, labels):
        augmented_inputs = torch.flip(inputs, dims=[0])
        augmented_labels = labels.clone()
        return torch.cat((inputs, augmented_inputs)), torch.cat((labels, augmented_labels))

    # Build the neural network model
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

    input_size = X_train_tensor.shape[1]
    hidden_size = 128
    num_layers = 3
    weight_decay = 0.03

    model = TemperaturePredict(input_size, hidden_size, num_layers)

    # loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    # train the model
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
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
        
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss += weight_decay * l2_reg
        
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
    
    # Use neural network to predict
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.unsqueeze(1))

    predictions_nn = test_outputs.squeeze().numpy()
    true_labels = y_test_tensor.numpy()

    # Train and test the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1)
    rf_model.fit(X_train_np, y_train_np)
    rf_predictions = rf_model.predict(X_test_np)

    # Convert the predictions to numpy array
    predictions_rf = np.array(rf_predictions)
    
    

    # Ensemble predictions
    stacked_features = np.column_stack((predictions_nn, predictions_rf))
    meta_model = LinearRegression()
    meta_model.fit(stacked_features, true_labels)
    ensemble_predictions = meta_model.predict(stacked_features)

    # Calculate evaluation metrics for the ensemble model
    absolute_errors_ensemble = np.abs(ensemble_predictions - true_labels)
    mae_ensemble = np.mean(absolute_errors_ensemble)
    rmse_ensemble = np.sqrt(np.mean(absolute_errors_ensemble ** 2))
    accuracy_ensemble = 100 - (mae_ensemble / np.mean(true_labels)) * 100

    # Calculate evaluation metrics for the neural network model
    absolute_errors_nn = np.abs(predictions_nn - true_labels)
    mae_nn = np.mean(absolute_errors_nn)
    rmse_nn = np.sqrt(np.mean(absolute_errors_nn ** 2))
    accuracy_nn = 100 - (mae_nn / np.mean(true_labels)) * 100

    return mae_ensemble, rmse_ensemble, accuracy_ensemble
