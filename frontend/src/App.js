import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import zoomPlugin from "chartjs-plugin-zoom";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  Filler,
  TimeScale,
} from "chart.js";

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  Filler,
  zoomPlugin,
  TimeScale
);

const App = () => {
  const [prices, setPrices] = useState([]);
  const [predicted, setPredicted] = useState([]);
  const [minutes, setMinutes] = useState("1m");
  const chartRef = useRef(null);

  // Fetch Real + Predicted Prices
  useEffect(() => {
  let lastPredictionTime = 0;

  const fetchData = async () => {
    try {
      const res = await axios.get(`http://127.0.0.1:8000/prices?minutes=${minutes}`);
      const real = res.data.real;
      const preds = res.data.predicted || [];

      // Add the latest real price
      setPrices((prev) => [...prev.slice(-10), { time: new Date(), value: real }]);

      // Update predictions only if enough time has passed
      const now = Date.now();
      const intervalMs =
        minutes === "1m" ? 60_000 :
        minutes === "5m" ? 5 * 60_000 :
        30 * 60_000;

      if (now - lastPredictionTime > intervalMs) {
        lastPredictionTime = now;

        setPredicted(
          preds.map((p, i) => ({
        time: new Date(now + (i + 1) * 1000),
        value: p,
          }))
        );
      }
    } catch (err) {
      console.warn("Error fetching data:", err);
    }
  };

  const id = setInterval(fetchData, 1000);
  return () => clearInterval(id);
}, [minutes]);


  // Chart.js data
  const data = {
    labels: [
      ...prices.map((p) => p.time.toLocaleTimeString()),
      ...predicted.map((p) => p.time.toLocaleTimeString()),
    ],
    datasets: [
      {
        label: "Real Price",
        data: prices.map((p) => p.value),
        borderColor: "#00FFFF",
        backgroundColor: "rgba(0,255,255,0.15)",
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointRadius: 0,
      },
      {
        label: "Predicted Price",
        data: [
          ...new Array(prices.length).fill(null),
          ...predicted.map((p) => p.value),
        ],
        borderColor: "rgba(255,165,0,0.9)",
        backgroundColor: "rgba(255,165,0,0.2)",
        borderWidth: 2,
        borderDash: [6, 4],
        tension: 0.4,
        fill: true,
        pointRadius: 0,
      },
    ],
  };

  // Chart Options
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index",
      intersect: false,
    },
    animation: {
      duration: 600,
      easing: "easeInOutQuart",
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Time (s)",
          color: "#ccc",
          font: { size: 13 },
        },
        ticks: {
          color: "#aaa",
          maxTicksLimit: 10,
        },
        grid: {
          color: "rgba(255,255,255,0.05)",
        },
      },
      y: {
        title: {
          display: true,
          text: "Price (USD)",
          color: "#ccc",
          font: { size: 13 },
        },
        ticks: {
          color: "#aaa",
        },
        grid: {
          color: "rgba(255,255,255,0.05)",
        },
      },
    },
    plugins: {
      legend: {
        labels: { color: "#fff" },
      },
      tooltip: {
        enabled: true,
        mode: "index",
        intersect: false,
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: $${ctx.formattedValue}`,
        },
      },
      zoom: {
        zoom: {
          wheel: { enabled: true },
          pinch: { enabled: true },
          drag: { enabled: true },
          mode: "x",
        },
        pan: {
          enabled: true,
          mode: "x",
        },
        limits: {
          x: { min: "original", max: "original" },
        },
      },
    },
  };

  return (
    <div className="bg-[#0b0b0b] text-white h-screen p-6 flex flex-col">
      <h1 className="text-2xl font-bold mb-4 text-cyan-400">
        Real-Time Crypto Price Predictor
      </h1>

      {/* Interval Selector */}
      <div className="flex gap-3 mb-4 text-black items-center">
        <label htmlFor="minutes" className="text-white">
          Interval:
        </label>
        <select
          id="minutes"
          value={minutes}
          onChange={(e) => setMinutes(e.target.value)}
          className="p-2 rounded bg-[#1e293b] text-white"
        >
          <option value="1m">Next 1 Minute</option>
          <option value="5m">Next 5 Minutes</option>
          <option value="30m">Next 30 Minutes</option>
        </select>

        <button
          onClick={() => chartRef.current.resetZoom()}
          className="bg-gray-700 hover:bg-gray-600 text-white p-2 rounded"
        >
          Reset Zoom
        </button>
      </div>

      {/* Chart Container */}
      <div className="flex-1 bg-[#121212] rounded-xl p-4 shadow-lg">
        <Line ref={chartRef} data={data} options={options} />
      </div>

      <p className="text-gray-400 mt-4 text-center text-sm">
        Interval: {minutes} • Auto-refresh every second • Scroll or pinch to zoom
      </p>
    </div>
  );
};

export default App;
