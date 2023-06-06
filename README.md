# Weather Analysis Through Main Cities Of Australia And Future Trend
Jiatong Gao, Edward Wu, Mia Wang
# For Question 1
You should download "weatherAUS.csv" and install matplotlib.pyplot and pandas libraries.
# For Question 2
You should download "weather3pm.csv" and "weather9am.csv". There are several libraries you need to install. Pandas, numpy, matplotlib.pyplot and Lasso from sklearn.linear_model. LabelEncoder, MinMaxScaler from sklearn.preprocessing.
KFold from sklearn.model_selection import.
Mean_squared_error from sklearn.metrics import.
# For Question 3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
