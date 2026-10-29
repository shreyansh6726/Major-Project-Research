import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
import yfinance as yf

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

POPULAR_STOCKS = {
    "Apple Inc.": "AAPL", "Microsoft Corp.": "MSFT",
    "Alphabet Inc. (Google)": "GOOGL", "Amazon.com Inc.": "AMZN",
    "Tesla Inc.": "TSLA", "NVIDIA Corp.": "NVDA",
    "Meta Platforms Inc.": "META",
}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

COLORS = {
    "navy": "#12233F", "navy_dark": "#0B172A", "blue": "#2563EB",
    "blue_dark": "#1D4ED8", "teal": "#0F766E", "green": "#15803D",
    "red": "#B91C1C", "ink": "#172033", "muted": "#64748B",
    "line": "#E2E8F0", "background": "#F4F7FB", "card": "#FFFFFF",
}
FEATURES = [
    "Lag_1", "Lag_2", "SMA_10", "SMA_50", "Volatility", "Volume",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist", "BB_Position", "BB_Width",
]


def engineer_features(df):
    """Compute the technical indicators used by both models."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["Lag_1"] = df["Return"].shift(1)
    df["Lag_2"] = df["Return"].shift(2)
    df["SMA_10"] = df["Close"].rolling(10).mean() / df["Close"] - 1
    df["SMA_50"] = df["Close"].rolling(50).mean() / df["Close"] - 1
    df["Volatility"] = df["Return"].rolling(10).std()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp12 - exp26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = rolling_mean + rolling_std * 2
    df["BB_Lower"] = rolling_mean - rolling_std * 2
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / rolling_mean
    return df


def train_and_save_models(ticker, model_path):
    df = engineer_features(yf.download(ticker, period="10y", progress=False))
    df["Target_Class"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df["Target_Reg"] = df["Close"].pct_change().shift(-1)
    df = df.dropna()
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01,
                            subsample=0.8, random_state=42, n_jobs=-1)
    clf.fit(df[FEATURES], df["Target_Class"])
    reg = lgb.LGBMRegressor(n_estimators=60, max_depth=3, num_leaves=21,
                            learning_rate=0.02, random_state=42, verbose=-1, n_jobs=-1)
    reg.fit(df[FEATURES], df["Target_Reg"])
    package = {"classifier": clf, "regressor": reg, "features": FEATURES}
    joblib.dump(package, model_path)
    return package, df


def calculate_prediction(company_name):
    ticker = POPULAR_STOCKS[company_name]
    model_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
    if os.path.exists(model_path):
        package = joblib.load(model_path)
        df = engineer_features(yf.download(ticker, period="6mo", progress=False))
    else:
        package, df = train_and_save_models(ticker, model_path)
    df = df.dropna(subset=package["features"])
    latest = df.iloc[[-1]][package["features"]]
    direction = int(package["classifier"].predict(latest)[0])
    probabilities = package["classifier"].predict_proba(latest)[0]
    expected_return = float(package["regressor"].predict(latest)[0])
    return (ticker, df, direction, float(probabilities[direction]) * 100,
            expected_return, float(df["Close"].iloc[-1]))


def make_card(parent, padding=18):
    card = tk.Frame(parent, bg=COLORS["card"], highlightthickness=1,
                    highlightbackground=COLORS["line"])
    inner = tk.Frame(card, bg=COLORS["card"])
    inner.pack(fill="both", expand=True, padx=padding, pady=padding)
    return card, inner


def set_status(text, success=False):
    status_label.config(text=text, fg=COLORS["green"] if success else "#CBD5E1")


def clear_chart(message="Your price visualization will appear here after prediction."):
    for widget in chart_host.winfo_children():
        widget.destroy()
    tk.Label(chart_host, text=message, bg=COLORS["card"], fg=COLORS["muted"],
             font=("Segoe UI", 10)).pack(expand=True)


def show_chart(df, ticker, expected_return):
    if not MATPLOTLIB_AVAILABLE:
        clear_chart("Install matplotlib to enable the embedded chart.")
        return
    for widget in chart_host.winfo_children():
        widget.destroy()
    close = pd.to_numeric(df.tail(90)["Close"], errors="coerce").dropna()
    average = close.rolling(20).mean()
    predicted_price = float(close.iloc[-1]) * (1 + expected_return)
    figure = Figure(figsize=(9.5, 3.2), dpi=100, facecolor=COLORS["card"])
    axis = figure.add_subplot(111)
    axis.set_facecolor(COLORS["card"])
    axis.grid(axis="y", color="#E8EEF6", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(COLORS["line"])
    axis.tick_params(colors=COLORS["muted"], labelsize=8, length=0)
    axis.set_title(f"{ticker} · Recent price trend", loc="left", pad=12,
                   color=COLORS["ink"], fontsize=12, fontweight="bold")
    x = list(range(len(close)))
    line, = axis.plot([], [], color=COLORS["blue"], linewidth=2.3, label="Close")
    avg_line, = axis.plot([], [], color=COLORS["teal"], linewidth=1.4,
                          linestyle="--", label="20-day average")
    axis.axhline(float(close.iloc[-1]), color="#CBD5E1", linewidth=1, linestyle=":")
    axis.scatter([len(close) - 1], [float(close.iloc[-1])], color=COLORS["blue"], s=32, zorder=5)
    axis.scatter([len(close) - 1], [predicted_price], color=COLORS["green"], s=44,
                 marker="D", zorder=6, label="Predicted next day")
    axis.legend(frameon=False, loc="upper left", fontsize=8, ncol=3)
    canvas = FigureCanvasTkAgg(figure, master=chart_host)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def reveal(index=2):
        if index > len(close):
            return
        line.set_data(x[:index], close.iloc[:index])
        avg_line.set_data(x[:index], average.iloc[:index])
        axis.relim()
        axis.autoscale_view()
        canvas.draw_idle()
        root.after(14, lambda: reveal(index + 3))
    reveal()


def on_prediction_complete(result=None, error=None):
    progress.stop()
    predict_button.config(state=tk.NORMAL, text="Load model & predict")
    if error:
        set_status("Prediction failed")
        messagebox.showerror("Unable to complete prediction", str(error))
        return
    ticker, df, direction, confidence, expected_return, latest_close = result
    direction_value.config(text="UP" if direction else "DOWN",
                           fg=COLORS["green"] if direction else COLORS["red"])
    confidence_value.config(text=f"{confidence:.1f}%")
    return_value.config(text=f"{expected_return * 100:+.2f}%")
    price_value.config(text=f"${latest_close:,.2f}")
    summary_value.config(text=f"{ticker} forecast complete · next trading day")
    set_status("Prediction ready", True)
    show_chart(df, ticker, expected_return)


def prediction_worker(company_name):
    try:
        root.after(0, lambda: set_status(f"Downloading data for {POPULAR_STOCKS[company_name]} …"))
        result = calculate_prediction(company_name)
        root.after(0, lambda: on_prediction_complete(result=result))
    except Exception as exc:
        root.after(0, lambda: on_prediction_complete(error=exc))


def run_prediction():
    company_name = company_combobox.get()
    if not company_name:
        messagebox.showerror("Select a company", "Please select a company first.")
        return
    predict_button.config(state=tk.DISABLED, text="Preparing prediction …")
    progress.start(12)
    set_status("Starting prediction …")
    threading.Thread(target=prediction_worker, args=(company_name,), daemon=True).start()


root = tk.Tk()
root.title("Market Insight · Stock Trend Predictor")
root.geometry("1120x780")
root.minsize(900, 680)
root.configure(bg=COLORS["background"])
style = ttk.Style(root)
style.theme_use("clam")
style.configure("TCombobox", fieldbackground=COLORS["card"], background=COLORS["card"],
                foreground=COLORS["ink"], bordercolor=COLORS["line"], padding=8)
style.configure("Modern.Horizontal.TProgressbar", troughcolor="#E8EEF6",
                background=COLORS["blue"], lightcolor=COLORS["blue"], darkcolor=COLORS["blue"])

header = tk.Frame(root, bg=COLORS["navy"], height=110)
header.pack(fill="x")
header.pack_propagate(False)
header_inner = tk.Frame(header, bg=COLORS["navy"])
header_inner.pack(fill="both", expand=True, padx=34, pady=22)
tk.Label(header_inner, text="MARKET INSIGHT", bg=COLORS["navy"], fg="#93C5FD",
         font=("Segoe UI", 9, "bold")).pack(anchor="w")
tk.Label(header_inner, text="Stock Trend Predictor", bg=COLORS["navy"], fg="white",
         font=("Segoe UI", 24, "bold")).pack(anchor="w")
status_label = tk.Label(header_inner, text="Ready", bg=COLORS["navy"], fg="#CBD5E1",
                        font=("Segoe UI", 10))
status_label.place(relx=1, rely=0.54, anchor="e")

main = tk.Frame(root, bg=COLORS["background"])
main.pack(fill="both", expand=True, padx=28, pady=24)
controls_card, controls = make_card(main, 20)
controls_card.pack(fill="x", pady=(0, 18))
tk.Label(controls, text="Forecast workspace", bg=COLORS["card"], fg=COLORS["ink"],
         font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
tk.Label(controls, text="Select a company and generate a current model-backed outlook.",
         bg=COLORS["card"], fg=COLORS["muted"], font=("Segoe UI", 9)).grid(
             row=1, column=0, columnspan=3, sticky="w", pady=(4, 16))
tk.Label(controls, text="Company", bg=COLORS["card"], fg=COLORS["muted"],
         font=("Segoe UI", 9, "bold")).grid(row=2, column=0, sticky="w")
company_combobox = ttk.Combobox(controls, values=list(POPULAR_STOCKS.keys()), width=28,
                                state="readonly", font=("Segoe UI", 10))
company_combobox.grid(row=3, column=0, sticky="ew", padx=(0, 22), pady=(6, 0))
company_combobox.current(0)
tk.Label(controls, text="Forecast horizon", bg=COLORS["card"], fg=COLORS["muted"],
         font=("Segoe UI", 9, "bold")).grid(row=2, column=1, sticky="w")
duration_combobox = ttk.Combobox(
    controls, values=["Next Trading Day (1-Day)", "Short-Term Horizon (3-Day)",
                      "Weekly Horizon (5-Day)"], width=28, state="readonly",
    font=("Segoe UI", 10))
duration_combobox.grid(row=3, column=1, sticky="ew", padx=(0, 22), pady=(6, 0))
duration_combobox.current(0)
predict_button = tk.Button(controls, text="Load model & predict", command=run_prediction,
                           bg=COLORS["blue"], fg="white", activebackground=COLORS["blue_dark"],
                           activeforeground="white", relief="flat", bd=0, padx=22, pady=11,
                           font=("Segoe UI", 10, "bold"), cursor="hand2")
predict_button.grid(row=3, column=2, sticky="e", pady=(6, 0))
predict_button.bind("<Enter>", lambda event: predict_button.config(bg=COLORS["blue_dark"]))
predict_button.bind("<Leave>", lambda event: predict_button.config(bg=COLORS["blue"]))
controls.columnconfigure(0, weight=1)
controls.columnconfigure(1, weight=1)
progress = ttk.Progressbar(controls, style="Modern.Horizontal.TProgressbar", mode="indeterminate")
progress.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(18, 0))

summary_card, summary = make_card(main, 18)
summary_card.pack(fill="x", pady=(0, 18))
summary_value = tk.Label(summary, text="No prediction generated yet", bg=COLORS["card"],
                         fg=COLORS["muted"], font=("Segoe UI", 9))
summary_value.pack(anchor="w", pady=(0, 12))
metrics = tk.Frame(summary, bg=COLORS["card"])
metrics.pack(fill="x")
metric_values = []
for index, (label, value) in enumerate([
    ("Predicted direction", "—"), ("Confidence", "—"),
    ("Expected return", "—"), ("Latest close", "—")]):
    block = tk.Frame(metrics, bg=COLORS["card"])
    block.grid(row=0, column=index, sticky="w", padx=(0, 72 if index < 3 else 0))
    tk.Label(block, text=label.upper(), bg=COLORS["card"], fg=COLORS["muted"],
             font=("Segoe UI", 8, "bold")).pack(anchor="w")
    value_label = tk.Label(block, text=value, bg=COLORS["card"], fg=COLORS["ink"],
                           font=("Segoe UI", 18, "bold"))
    value_label.pack(anchor="w", pady=(5, 0))
    metric_values.append(value_label)
direction_value, confidence_value, return_value, price_value = metric_values

chart_card, chart_inner = make_card(main, 18)
chart_card.pack(fill="both", expand=True)
chart_title = tk.Frame(chart_inner, bg=COLORS["card"])
chart_title.pack(fill="x", pady=(0, 8))
tk.Label(chart_title, text="Price visualization", bg=COLORS["card"], fg=COLORS["ink"],
         font=("Segoe UI", 13, "bold")).pack(side="left")
tk.Label(chart_title, text="90 most recent sessions", bg=COLORS["card"], fg=COLORS["muted"],
         font=("Segoe UI", 9)).pack(side="right")
chart_host = tk.Frame(chart_inner, bg=COLORS["card"])
chart_host.pack(fill="both", expand=True)
clear_chart()

root.mainloop()
