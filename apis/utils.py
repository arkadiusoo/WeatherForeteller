import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests

from django.conf import settings

WINDOW_SIZE = 48
PREDICT_HORIZON = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------- MODEL DEFINITION ----------------------
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(
            x)  # output shape: (batch, seq_len, hidden_size)
        last_step = output[:, -1, :]  # take the last time step
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
    df['Date Time'] = pd.to_datetime(df['Date Time'],
                                     format='%d.%m.%Y %H:%M:%S')
    df.set_index('Date Time', inplace=True)

    time_list, temp_list, hum_list = predictor(df)
    return time_list, temp_list, hum_list


def getCityData(city: str):
    api_key = "d0c5fc84e556463389a220217251205"
    now = datetime.now()
    today = now.date()
    yesterday = (now - timedelta(days=1)).date()
    yesterday_2 = (now - timedelta(days=2)).date()

    url_yesterday_2 = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={yesterday_2}"
    url_yesterday = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={yesterday}"
    url_today = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={today}"

    response_yesterday_2 = requests.get(url_yesterday_2)
    response_yesterday = requests.get(url_yesterday)
    response_today = requests.get(url_today)

    if response_yesterday.status_code != 200:
        raise Exception(
            f"Error fetching yesterday's data: {response_yesterday.status_code} - {response_yesterday.text}")
    if response_today.status_code != 200:
        raise Exception(
            f"Error fetching today's data: {response_today.status_code} - {response_today.text}")
    if response_yesterday_2.status_code != 200:
        raise Exception(
            f"Error fetching yesterday's data: {response_yesterday_2.status_code} - {response_yesterday_2.text}")

    data_yesterday_2 = response_yesterday_2.json()
    data_yesterday = response_yesterday.json()
    data_today = response_today.json()

    hours_yesterday_2 = data_yesterday_2['forecast']['forecastday'][0]['hour']
    hours_yesterday = data_yesterday['forecast']['forecastday'][0]['hour']
    hours_today = data_today['forecast']['forecastday'][0]['hour']

    hours_today_filtered = [
        h for h in hours_today
        if int(h['time'][-5:-3]) <= now.hour
    ]

    combined_hours = hours_yesterday_2 + hours_yesterday + hours_today_filtered

    last_48_hours = combined_hours[-48:]
    time = []
    temp = []
    humidity = []

    for i in last_48_hours:
        time.append(i['time'])
        temp.append(i['temp_c'])
        humidity.append(i['humidity'])

    df = pd.DataFrame(
        {'Date Time': time, 'T (degC)': temp, 'humidity': humidity})
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%Y-%m-%d %H:%M')
    df.set_index('Date Time', inplace=True)

    time_list, temp_list, hum_list = predictor(df)
    return time_list, temp_list, hum_list


def predictor(ts):
    models_dir = os.path.join(settings.BASE_DIR,
                              'prediction-models/saved-models')
    # Paths for temperature
    scaler_temp_path = os.path.join(models_dir, 'scaler_temp.pkl')
    model_temp_path = os.path.join(models_dir, 'model_temp.pth')
    # Paths for humidity
    scaler_hum_path = os.path.join(models_dir, 'scaler_humidity.pkl')
    model_hum_path = os.path.join(models_dir, 'model_humidity.pth')

    # Load scalers
    with open(scaler_temp_path, 'rb') as f:
        scaler_temp = pickle.load(f)
    with open(scaler_hum_path, 'rb') as f:
        scaler_hum = pickle.load(f)

    # Load models
    model_temp = LSTMForecast(input_size=1, hidden_size=16, num_layers=1).to(
        DEVICE)
    model_temp.load_state_dict(
        torch.load(model_temp_path, map_location=DEVICE))
    model_temp.eval()

    model_hum = LSTMForecast(input_size=1, hidden_size=16, num_layers=1).to(
        DEVICE)
    model_hum.load_state_dict(torch.load(model_hum_path, map_location=DEVICE))
    model_hum.eval()

    # Prepare and scale input sequences
    vals_temp = ts['T (degC)'].values.reshape(-1, 1)
    print(ts)
    vals_hum = ts['humidity'].values.reshape(-1, 1)
    scaled_temp = scaler_temp.transform(vals_temp).flatten().tolist()
    scaled_hum = scaler_hum.transform(vals_hum).flatten().tolist()

    # Rolling window predictions
    preds_temp_scaled = []
    preds_hum_scaled = []
    seq_temp = scaled_temp.copy()
    seq_hum = scaled_hum.copy()

    for _ in range(PREDICT_HORIZON):
        # Temperature prediction
        x_temp = np.array(seq_temp[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
        t_temp = torch.tensor(x_temp, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            p_temp = model_temp(t_temp).cpu().numpy().flatten()[0]
        preds_temp_scaled.append(p_temp)
        seq_temp.append(p_temp)

        # Humidity prediction
        x_hum = np.array(seq_hum[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
        t_hum = torch.tensor(x_hum, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            p_hum = model_hum(t_hum).cpu().numpy().flatten()[0]
        preds_hum_scaled.append(p_hum)
        seq_hum.append(p_hum)

    # Invert scaling
    preds_temp = scaler_temp.inverse_transform(
        np.array(preds_temp_scaled).reshape(-1, 1)
    ).flatten().tolist()
    preds_hum = scaler_hum.inverse_transform(
        np.array(preds_hum_scaled).reshape(-1, 1)
    ).flatten().tolist()

    # Build time indices for forecasts
    last_time = ts.index[-1]
    time_list = [
        (last_time + pd.Timedelta(hours=i + 1)).strftime('%H.%M')
        for i in range(PREDICT_HORIZON)
    ]

    return time_list, preds_temp, preds_hum
