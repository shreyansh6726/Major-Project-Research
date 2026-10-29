# Stock Trend & Price Predictor

A desktop stock-prediction application built with Python, Tkinter, XGBoost, LightGBM, pandas, and Yahoo Finance data through `yfinance`.

The application lets a user select a supported company and produces a report containing:

- predicted next-day direction (up or down)
- model confidence
- predicted return magnitude
- latest available closing price

## Features

- Tkinter desktop interface
- Support for Apple, Microsoft, Alphabet, Amazon, Tesla, NVIDIA, and Meta
- Automatic feature engineering using returns, lagged returns, moving averages, volatility, RSI, MACD, and Bollinger Bands
- XGBoost classifier for direction prediction
- LightGBM regressor for expected return prediction
- Cached models saved under `models/`
- Automatic model training when a cached model is not available

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Tkinter (usually included with standard Python installations)
- Internet access when downloading Yahoo Finance data

Install the Python packages from a virtual environment:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install joblib lightgbm matplotlib numpy pandas xgboost yfinance
```

`matplotlib` is included in `requirements.txt` for the embedded post-prediction chart. If it is not installed, the rest of the application still runs and the chart area displays an installation notice.

If PowerShell blocks activation, run the application with the virtual-environment interpreter directly:

```powershell
venv\Scripts\python.exe app.py
```

## Run the application

From the project directory:

```powershell
py app.py
```

Select a company and click **Load Model & Predict**. The first run for a company can take longer because the application downloads ten years of data and trains both models. Later runs reuse the cached model and download six months of recent data. Results include a visual chart with the recent closing-price trend, 20-day average, and predicted next-day price marker.

## Project structure

```text
.
├── app.py              # Tkinter UI, data processing, training, and prediction
├── models/             # Local cached model files; ignored by Git
├── requirements.txt    # Dependency specification file
├── AGENTS.md           # Guidance for contributors and coding agents
└── README.md           # Project documentation
```

## Important notes

- `yfinance` requires `period="6mo"` for six months. `period="6m"` is invalid.
- The duration selector currently changes the report label only; the trained model predicts the next trading day.
- Predictions are experimental estimates, not financial advice.
- Yahoo Finance data can be delayed, unavailable, or rate-limited.
- Trained `.pkl` files are local artifacts and are excluded from version control.

## Troubleshooting

### Invalid period error

Use `6mo`, not `6m`, in `yf.download`.

### Empty or failed downloads

Check the ticker, internet connection, Yahoo Finance availability, and whether the system clock is correct. Retry after a short delay if Yahoo Finance rate-limits requests.

### Tkinter is missing

Install a standard Python distribution that includes Tkinter. On Linux, install the distribution's `python3-tk` package.

