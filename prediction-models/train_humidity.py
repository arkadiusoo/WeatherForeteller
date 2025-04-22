import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------ PARAMETERS ------------------------
CSV_PATH        = 'max_planck_weather_ts.csv'
LATITUDE        = 53.567555
LONGITUDE       = 9.9749697
REFERENCE_UTC   = pd.Timestamp('2025-04-25T00:00:00')
WINDOW_SIZE     = 48  # number of past hours to use as input
PREDICT_HORIZON = 24  # number of hours to forecast
BATCH_SIZE      = 32
EPOCHS          = 10
LEARNING_RATE   = 0.001
API_TIMEOUT     = 10  # seconds

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------- DATA LOADING AND PREPARATION ---------------------
df = pd.read_csv(CSV_PATH, delimiter=',')
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
df.set_index('Date Time', inplace=True)

# Aggregate to hourly means
hourly_df = df.resample('h').mean()
humidity_series = hourly_df['rh (%)'].dropna()

# Scale data to [0, 1]
scaler = MinMaxScaler(feature_range=(0,1))
values = humidity_series.values.reshape(-1, 1)
scaled_values = scaler.fit_transform(values)

# Construct sequences for training
X, y = [], []
for i in range(WINDOW_SIZE, len(scaled_values)):
    X.append(scaled_values[i-WINDOW_SIZE:i, 0])
    y.append(scaled_values[i, 0])
X = np.array(X)  # shape: (samples, WINDOW_SIZE)
y = np.array(y)  # shape: (samples,)

# Convert to PyTorch tensors
tensor_X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (samples, WINDOW_SIZE, 1)
tensor_y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (samples, 1)
dataset  = TensorDataset(tensor_X, tensor_y)
loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------- MODEL DEFINITION ----------------------
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)       # output shape: (batch, seq_len, hidden_size)
        last_step = output[:, -1, :]   # take the last time step
        return self.linear(last_step)  # shape: (batch, 1)

model     = LSTMForecast().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------------ TRAINING LOOP ------------------------
model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    epoch_loss /= len(dataset)
    print(f"Epoch {epoch}/{EPOCHS} — Loss: {epoch_loss:.6f}")

# ------------------ EVALUATION ON TRAINING SET ------------------
model.eval()
with torch.no_grad():
    preds = model(tensor_X.to(device)).cpu().numpy().flatten()
    actual = tensor_y.numpy().flatten()
    train_mse = mean_squared_error(actual, preds)
    print(f"\nTraining MSE: {train_mse:.6f}")

# ------------- SAVE MODEL AND SCALER TO DISK -------------
torch.save(model.state_dict(), 'saved-models/model_humidity.pth')
with open('saved-models/scaler_humidity.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Saved model to 'saved-models/model_humidity.pth' and scaler to 'saved-models/scaler_humidity.pkl'.")
