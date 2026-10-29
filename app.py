import os
import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

# =====================================================================
# POPULAR STOCKS WITH RESPECTIVE TICKERS (IN COMMENTS)
# =====================================================================
POPULAR_STOCKS = {
    "Apple Inc.": "AAPL",  # Apple Ticker
    "Microsoft Corp.": "MSFT",  # Microsoft Ticker
    "Alphabet Inc. (Google)": "GOOGL",  # Google Ticker
    "Amazon.com Inc.": "AMZN",  # Amazon Ticker
    "Tesla Inc.": "TSLA",  # Tesla Ticker
    "NVIDIA Corp.": "NVDA",  # NVIDIA Ticker
    "Meta Platforms Inc.": "META",  # Meta Ticker
}

# Ensure models directory exists
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def engineer_features(df):
  """Feature engineering pipeline computing RSI, MACD, Bollinger Bands, and Lags."""
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  # Returns & Lags
  df["Return"] = df["Close"].pct_change()
  df["Lag_1"] = df["Return"].shift(1)
  df["Lag_2"] = df["Return"].shift(2)
  df["SMA_10"] = df["Close"].rolling(window=10).mean() / df["Close"] - 1
  df["SMA_50"] = df["Close"].rolling(window=50).mean() / df["Close"] - 1
  df["Volatility"] = df["Return"].rolling(window=10).std()

  # RSI (14 period)
  delta = df["Close"].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  df["RSI"] = 100 - (100 / (1 + rs))

  # MACD
  exp12 = df["Close"].ewm(span=12, adjust=False).mean()
  exp26 = df["Close"].ewm(span=26, adjust=False).mean()
  df["MACD"] = exp12 - exp26
  df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
  df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

  # Bollinger Bands
  window = 20
  rolling_mean = df["Close"].rolling(window=window).mean()
  rolling_std = df["Close"].rolling(window=window).std()
  df["BB_Upper"] = rolling_mean + (rolling_std * 2)
  df["BB_Lower"] = rolling_mean - (rolling_std * 2)
  df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (
      df["BB_Upper"] - df["BB_Lower"]
  )
  df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / rolling_mean

  return df


def train_and_save_models(ticker, model_path):
  """Downloads last 10 years of data, trains XGBoost & LightGBM, and saves them."""
  status_label.config(
      text=f"Downloading last 10 years of data for {ticker}..."
  )
  root.update()

  # Fetch 10 years of historical data from yfinance
  df = yf.download(ticker, period="10y")
  df = engineer_features(df)

  # Targets
  df["Target_Class"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
  df["Target_Reg"] = df["Close"].pct_change().shift(-1)
  df = df.dropna()

  features = [
      "Lag_1",
      "Lag_2",
      "SMA_10",
      "SMA_50",
      "Volatility",
      "Volume",
      "RSI",
      "MACD",
      "MACD_Signal",
      "MACD_Hist",
      "BB_Position",
      "BB_Width",
  ]

  X = df[features]
  y_class = df["Target_Class"]
  y_reg = df["Target_Reg"]

  status_label.config(text=f"Training models for {ticker} (please wait)...")
  root.update()

  # Train optimized XGBoost Classifier (Direction)
  clf = xgb.XGBClassifier(
      n_estimators=100,
      max_depth=3,
      learning_rate=0.01,
      subsample=0.8,
      random_state=42,
      n_jobs=-1,
  )
  clf.fit(X, y_class)

  # Train optimized LightGBM Regressor (Magnitude/Return)
  reg = lgb.LGBMRegressor(
      n_estimators=60,
      max_depth=3,
      num_leaves=21,
      learning_rate=0.02,
      random_state=42,
      verbose=-1,
      n_jobs=-1,
  )
  reg.fit(X, y_reg)

  # Pack and save models
  models_package = {"classifier": clf, "regressor": reg, "features": features}
  joblib.dump(models_package, model_path)
  return models_package, df


def run_prediction():
  """Handles selection, checks model cache, trains if missing, and predicts."""
  selected_company_name = company_combobox.get()
  selected_duration = duration_combobox.get()

  if not selected_company_name:
    messagebox.showerror("Error", "Please select a company first!")
    return

  ticker = POPULAR_STOCKS[selected_company_name]
  model_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")

  try:
    # Check if pre-trained model exists in models/ directory
    if os.path.exists(model_path):
      status_label.config(
          text=f"Loading existing model for {ticker} from models/..."
      )
      root.update()
      models_package = joblib.load(model_path)
      clf = models_package["classifier"]
      reg = models_package["regressor"]
      features = models_package["features"]

      # Fetch recent data to build latest feature row
      # yfinance uses "6mo" for six months ("6m" is not a valid period).
      df = yf.download(ticker, period="6mo")
      df = engineer_features(df)
    else:
      status_label.config(
          text=f"Model for {ticker} not found. Training new model..."
      )
      root.update()
      models_package, df = train_and_save_models(ticker, model_path)
      clf = models_package["classifier"]
      reg = models_package["regressor"]
      features = models_package["features"]

    # Extract the latest available row for prediction
    latest_data = df.iloc[[-1]][features]

    # Predict direction & return magnitude
    direction_pred = clf.predict(latest_data)[0]
    direction_prob = clf.predict_proba(latest_data)[0]
    return_pred = reg.predict(latest_data)[0]

    direction_text = "UP 📈" if direction_pred == 1 else "DOWN 📉"
    confidence = (
        direction_prob[1] if direction_pred == 1 else direction_prob[0]
    ) * 100

    # Format result text display
    result_string = (
        f"=== Prediction Report for {ticker} ===\n"
        f"• Horizon / Duration: {selected_duration}\n"
        f"• Predicted Direction: {direction_text} (Confidence: {confidence:.2f}%)\n"
        f"• Expected Return: {return_pred * 100:.2f}%\n"
        f"• Latest Close Price: ${df['Close'].iloc[-1]:.2f}\n"
        f"• Status: Success"
    )

    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, result_string)
    result_text.config(state=tk.DISABLED)

    status_label.config(text="Done! Model ready.")

  except Exception as e:
    messagebox.showerror("Error", f"An error occurred: {str(e)}")
    status_label.config(text="Execution failed.")


# =====================================================================
# TKINTER GUI LAYOUT SETUP
# =====================================================================
root = tk.Tk()
root.title("Stock Trend & Price Predictor")
root.geometry("540x440")
root.config(bg="#f8f9fa")

title_lbl = tk.Label(
    root,
    text="XGBoost + LightGBM Stock Predictor",
    font=("Arial", 15, "bold"),
    bg="#f8f9fa",
    fg="#212529",
)
title_lbl.pack(pady=15)

frame = tk.Frame(root, bg="#f8f9fa")
frame.pack(pady=10)

# Company Selection Dropdown
lbl_comp = tk.Label(
    frame, text="Select Company:", font=("Arial", 11), bg="#f8f9fa", anchor="w"
)
lbl_comp.grid(row=0, column=0, sticky="w", padx=10, pady=8)

company_combobox = ttk.Combobox(
    frame,
    values=list(POPULAR_STOCKS.keys()),
    width=28,
    state="readonly",
    font=("Arial", 10),
)
company_combobox.grid(row=0, column=1, padx=10, pady=8)
company_combobox.current(0)

# Duration Selection Dropdown
lbl_dur = tk.Label(
    frame, text="Select Duration:", font=("Arial", 11), bg="#f8f9fa", anchor="w"
)
lbl_dur.grid(row=1, column=0, sticky="w", padx=10, pady=8)

duration_combobox = ttk.Combobox(
    frame,
    values=[
        "Next Trading Day (1-Day)",
        "Short-Term Horizon (3-Day)",
        "Weekly Horizon (5-Day)",
    ],
    width=28,
    state="readonly",
    font=("Arial", 10),
)
duration_combobox.grid(row=1, column=1, padx=10, pady=8)
duration_combobox.current(0)

# Trigger Button
predict_button = tk.Button(
    root,
    text="Load Model & Predict",
    command=run_prediction,
    bg="#0d6efd",
    fg="white",
    font=("Arial", 11, "bold"),
    padx=12,
    pady=6,
)
predict_button.pack(pady=12)

# Status info label
status_label = tk.Label(
    root, text="Ready", font=("Arial", 9, "italic"), bg="#f8f9fa", fg="#6c757d"
)
status_label.pack(pady=2)

# Result text box
result_text = tk.Text(
    root,
    height=7,
    width=60,
    font=("Consolas", 10),
    state=tk.DISABLED,
    bg="#ffffff",
    fg="#212529",
)
result_text.pack(pady=10)

root.mainloop()
