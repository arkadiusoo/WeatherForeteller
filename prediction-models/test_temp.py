import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn

# ------------------------ PARAMETERS ------------------------
LATITUDE        = 53.567555
LONGITUDE       = 9.9749697
WINDOW_SIZE     = 48  # input window size in hours
PREDICT_HORIZON = 24  # forecast horizon in hours
API_TIMEOUT     = 10  # seconds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------- MODEL DEFINITION ----------------------
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        return self.linear(last_step)

# ---------------------- LOAD MODEL AND SCALER ----------------------
model = LSTMForecast().to(device)
model.load_state_dict(torch.load('saved-models/model_temp.pth', map_location=device))
model.eval()

with open('saved-models/scaler_temp.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --------------------- HISTORICAL TEST FUNCTION ---------------------
def test_historical_date(date_str):
    """
    date_str: 'YYYY-MM-DD', e.g. '2024-11-01'
    """
    reference_time = pd.Timestamp(f"{date_str}T00:00:00")
    start_date = (reference_time - pd.Timedelta(hours=WINDOW_SIZE)).date().isoformat()
    end_date   = (reference_time + pd.Timedelta(hours=PREDICT_HORIZON)).date().isoformat()

    url = (
        f"https://archive-api.open-meteo.com/v1/era5"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly=temperature_2m"
        f"&start_date={start_date}&end_date={end_date}"
    )
    data = requests.get(url, timeout=API_TIMEOUT).json()['hourly']
    times = pd.to_datetime(data['time'])
    temperatures = np.array(data['temperature_2m'])

    # Extract input window
    past_mask = (times > reference_time - pd.Timedelta(hours=WINDOW_SIZE)) & (times <= reference_time)
    window_temperatures = temperatures[past_mask]
    if len(window_temperatures) != WINDOW_SIZE:
        raise RuntimeError(f"Input window missing data: {len(window_temperatures)} of {WINDOW_SIZE} points")

    # True future temperatures
    future_mask = (times > reference_time) & (times <= reference_time + pd.Timedelta(hours=PREDICT_HORIZON))
    true_temperatures  = temperatures[future_mask]
    future_times = times[future_mask]

    # LSTM iterative forecasting
    scaled_window = scaler.transform(window_temperatures.reshape(-1, 1)).flatten()
    seq = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    preds_scaled = []
    with torch.no_grad():
        for _ in range(PREDICT_HORIZON):
            p = model(seq).cpu().item()
            preds_scaled.append(p)
            next_input = torch.tensor([[[p]]], dtype=torch.float32).to(device)
            seq = torch.cat((seq[:, 1:, :], next_input), dim=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    # Align lengths and filter NaNs
    length = min(len(true_temperatures), len(preds))
    true_temperatures = true_temperatures[:length]
    preds = preds[:length]
    valid = ~np.isnan(true_temperatures) & ~np.isnan(preds)
    true_temperatures = true_temperatures[valid]
    preds = preds[valid]
    plot_times = future_times[:length][valid]

    mse = mean_squared_error(true_temperatures, preds)
    print(f"\n=== Historical test for {date_str} ===")
    print(f"Data points used: {len(true_temperatures)}, MSE: {mse:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(plot_times, true_temperatures, marker='o', label='Open-Meteo historical data')
    plt.plot(plot_times, preds, marker='x', label='LSTM forecast')
    plt.title(f"LSTM vs. Open-Meteo historical data on {date_str}")
    plt.xlabel('Time (UTC)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------ SINGLE-DATE TEST ------------------------
def test_for_date(date_str):
    """
    date_str: 'YYYY-MM-DD', e.g. '2025-04-22'
    """
    reference_time = pd.Timestamp(f"{date_str}T00:00:00")

    # 1) Fetch past WINDOW_SIZE hours from Open‑Meteo
    past_start = (reference_time - pd.Timedelta(hours=WINDOW_SIZE)).date().isoformat()
    past_end   = reference_time.date().isoformat()
    url_past = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        "&hourly=temperature_2m"
        f"&start_date={past_start}&end_date={past_end}"
        "&timezone=UTC"
    )
    past_data = requests.get(url_past, timeout=API_TIMEOUT).json()['hourly']
    times_past = pd.to_datetime(past_data['time'], utc=True).tz_convert(None)
    temperatures_past = np.array(past_data['temperature_2m'])
    mask = (times_past > reference_time - pd.Timedelta(hours=WINDOW_SIZE)) & (times_past <= reference_time)
    window_temperatures = temperatures_past[mask]
    assert len(window_temperatures) == WINDOW_SIZE, f"Expected {WINDOW_SIZE} points, got {len(window_temperatures)}"

    # Prepare input tensor
    scaled_window = scaler.transform(window_temperatures.reshape(-1, 1)).flatten()
    seq = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    # Generate forecast
    preds_scaled = []
    with torch.no_grad():
        for _ in range(PREDICT_HORIZON):
            p = model(seq).cpu().numpy().flatten()[0]
            preds_scaled.append(p)
            next_input = torch.tensor([[[p]]], dtype=torch.float32).to(device)
            seq = torch.cat((seq[:, 1:, :], next_input), axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    # Retrieve Open‑Meteo forecast
    future_start = reference_time.date().isoformat()
    future_end   = (reference_time + pd.Timedelta(hours=PREDICT_HORIZON)).date().isoformat()
    url_future = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        "&hourly=temperature_2m"
        f"&start_date={future_start}&end_date={future_end}"
        "&timezone=UTC"
    )
    future_data = requests.get(url_future, timeout=API_TIMEOUT).json()['hourly']
    times_fut = pd.to_datetime(future_data['time'], utc=True).tz_convert(None)
    temperatures_fut = np.array(future_data['temperature_2m'])
    mask_fut = (times_fut > reference_time) & (times_fut <= reference_time + pd.Timedelta(hours=PREDICT_HORIZON))
    true_temperatures = temperatures_fut[mask_fut]
    forecast_times = times_fut[mask_fut]

    # Compute MSE and plot comparison
    mse = mean_squared_error(true_temperatures, preds)
    print(f"\n=== Forecast test for {date_str} ===")
    print(f"MSE for next {PREDICT_HORIZON} hours: {mse:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(forecast_times, true_temperatures, marker='o', label='Open-Meteo forecast')
    plt.plot(forecast_times, preds, marker='x', label='LSTM forecast')
    plt.title(f"LSTM vs. Open-Meteo on {date_str}")
    plt.xlabel('Time (UTC)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------- RUN TESTS -----------------------
test_historical_date('2024-11-01')
for dt in ['2025-04-22', '2025-04-25', '2025-05-05']:
    test_for_date(dt)
