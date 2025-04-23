import os
import pickle

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from django.conf import settings

WINDOW_SIZE     = 48
PREDICT_HORIZON = 24
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def predict_from_csv(path: str):
    """
    (columns: 'Date Time', 'T (degC)')

    Returns:
        time_list: list[str] – times in "HH.MM" format
        temp_list: list[float] – predicted temperatures [°C]
    """
    # Data loading and preparation
    df = pd.read_csv(path, delimiter=',')
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df.set_index('Date Time', inplace=True)

    hourly = df.resample('h').mean()
    ts = hourly['T (degC)'].dropna()

    if len(ts) < WINDOW_SIZE:
        raise ValueError(f"Not enough data (min. {WINDOW_SIZE} hours required), only {len(ts)} available")

    # Load scaler and model
    models_dir = os.path.join(settings.BASE_DIR, 'saved-models')
    scaler_path = os.path.join(models_dir, 'scaler_temp.pkl')
    model_path  = os.path.join(models_dir, 'model_temp.pth')

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    model = LSTMForecast(input_size=1, hidden_size=16, num_layers=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Prepare input sequence
    vals = ts.values.reshape(-1, 1)
    scaled = scaler.transform(vals)

    window_seq = list(scaled[-WINDOW_SIZE:].flatten())

    # Iteratively predict PREDICT_HORIZON points
    preds_scaled = []
    for _ in range(PREDICT_HORIZON):
        x = np.array(window_seq[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            p = model(x_tensor).cpu().numpy().flatten()[0]
        preds_scaled.append(p)
        window_seq.append(p)

    # Invert scaling
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    # Generate list of times based on the last timestamp
    last_time = ts.index[-1]
    time_list = [
        (last_time + pd.Timedelta(hours=i + 1)).strftime('%H.%M')
        for i in range(PREDICT_HORIZON)
    ]
    temp_list = preds.tolist()

    return time_list, temp_list
