import redis
import torch
from models.lstm_model import LSTMPricePredictor

# Load trained model
model = LSTMPricePredictor(input_size=10, hidden_size=64, num_layers=2, output_size=1)
model.load_state_dict(torch.load("models/lstm.pth", map_location="cpu"))
model.eval()

# Redis connection
r = redis.Redis(host="localhost", port=6379, db=0)

def get_latest_window(seq_length=20):
    """Fetch the latest price sequence from Redis"""
    prices = r.lrange("BTC_USDT_prices", 0, seq_length-1)
    prices = [float(p) for p in reversed(prices)]  # oldest -> newest
    return prices

def predict_next_price(seq_length=20):
    prices = get_latest_window(seq_length)
    if len(prices) < seq_length:
        print("Not enough data yet.")
        return None

    # Convert to model input
    # e.g., split into windows of size INPUT_SIZE=10
    input_tensor = torch.tensor([prices[i:i+10] for i in range(seq_length-10+1)], dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # batch dimension

    with torch.no_grad():
        prediction = model(input_tensor).item()
    return prediction

if __name__ == "__main__":
    import time
    while True:
        pred = predict_next_price()
        if pred:
            print(f"Predicted next BTC/USDT price: {pred}")
        time.sleep(5)  # every 5 seconds
