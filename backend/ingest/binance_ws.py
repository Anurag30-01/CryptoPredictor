import asyncio
import json
import websockets
import redis

# Redis connection
r = redis.Redis(host="localhost", port=6379, db=0)

BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@trade"
MAX_PRICES = 500  # Sliding window size

async def binance_price_stream():
    async with websockets.connect(BINANCE_WS) as ws:
        print("Connected to Binance WebSocket")
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            price = float(data["p"])
            r.lpush("BTC_USDT_prices", price)
            r.ltrim("BTC_USDT_prices", 0, MAX_PRICES-1)  # keep last 200 prices
            print(f"Price pushed to Redis: {price}")

if __name__ == "__main__":
    asyncio.run(binance_price_stream())
