from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import numpy as np
from binance import Client
from sklearn.preprocessing import MinMaxScaler
import os
from models.lstm_model import LSTMModel

app = FastAPI()

# ================== CORS SETTINGS ==================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== LSTM MODEL LOADER ==================
def load_model(interval="1m"):
    """Dynamically load model based on interval."""
    model_path = f"models/models/lstm_{interval}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {interval} not found. Please train it first using train.py")

    input_size = 1
    hidden_size = 128
    num_layers = 2
    output_size = 1

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# ================== PRICE FETCHER ==================
def get_prices(symbol="BTCUSDT", interval="1m", limit=100):
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    closes = np.array([float(k[4]) for k in klines])
    return closes

# ================== PREDICTION ROUTE ==================
@app.get("/prices")
def get_predictions(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("1m"),   # e.g. "1m", "5m", "30m"
    predict_steps: int = Query(30) # how many steps to predict ahead
):
    try:
        # Load Model
        model = load_model(interval)

        # Get data
        prices = get_prices(symbol, interval)
        real_price = float(prices[-1])

        # Scale data
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(prices.reshape(-1, 1))

        # Prepare input (lookback = 60)
        lookback = 60
        seq = scaled[-lookback:]
        input_seq = torch.tensor(seq.reshape(1, lookback, 1), dtype=torch.float32)

        preds = []
        curr_seq = input_seq.clone()

        # Predict multiple steps
        for _ in range(predict_steps):
            with torch.no_grad():
                next_val = model(curr_seq)
                preds.append(next_val.item())
                # shift sequence left and append new prediction
                new_seq = torch.cat([curr_seq[:, 1:, :], next_val.reshape(1, 1, 1)], dim=1)
                curr_seq = new_seq

        # Inverse scale
        pred_scaled = np.array(preds).reshape(-1, 1)
        predicted_prices = scaler.inverse_transform(pred_scaled).flatten().tolist()

        return {
            "symbol": symbol,
            "interval": interval,
            "real": real_price,
            "predicted": predicted_prices
        }

    except Exception as e:
        return {"error": str(e)}

# ================== ROOT ROUTE ==================
@app.get("/")
def root():
    return {"message": "Crypto Predictor API with Interval Support is running!"}
