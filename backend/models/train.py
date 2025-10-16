import torch
import torch.nn as nn
import numpy as np
from binance import Client
from sklearn.preprocessing import MinMaxScaler
import argparse
import os
from lstm_model import LSTMModel

# ================== ARGUMENT PARSER ==================
parser = argparse.ArgumentParser()
parser.add_argument("--symbol", default="BTCUSDT", help="Crypto symbol")
parser.add_argument("--interval", default="1m", help="Binance interval: 1m, 5m, 15m, 1h, etc.")
parser.add_argument("--lookback", type=int, default=60, help="How many data points to look back")
parser.add_argument("--predict_steps", type=int, default=60, help="How many steps to predict ahead")
args = parser.parse_args()

# ================== FETCH DATA ==================
print(f"⏳ Fetching {args.symbol} {args.interval} data from Binance...")
client = Client()
klines = client.get_klines(symbol=args.symbol, interval=args.interval, limit=1000)
prices = np.array([float(k[4]) for k in klines]).reshape(-1, 1)
print(f"✅ Retrieved {len(prices)} prices for {args.symbol} ({args.interval})")

# ================== NORMALIZE DATA ==================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)

X, y = [], []
for i in range(len(scaled) - args.lookback):
    X.append(scaled[i:i + args.lookback])
    y.append(scaled[i + args.lookback])
X, y = np.array(X), np.array(y)

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

# ================== MODEL SETUP ==================
input_size = 1
hidden_size = 128
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ================== TRAIN MODEL ==================
print("🚀 Starting training...")
epochs = 30
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_t)
    loss = criterion(output, y_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.8f}")

# ================== SAVE MODEL ==================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), f"models/lstm_{args.interval}.pth")

# ================== SAVE SCALER ==================
np.save("models/scaler_data.npy", [scaler.data_min_, scaler.data_max_])
print(f"✅ Model trained for {args.interval} and saved.")
