# Agent and Contributor Guide

## Project overview

This is a small Tkinter desktop application for stock trend prediction. The main implementation is in `app.py`.

## Working conventions

- Keep changes focused and compatible with Windows PowerShell usage.
- Preserve the existing GUI workflow unless the task explicitly requests a redesign.
- Use four-space indentation in new Python code and keep functions small where practical.
- Prefer clear, standard-library solutions over adding unnecessary dependencies.
- Do not commit credentials, API keys, downloaded datasets, or generated model binaries.
- Keep user-facing error messages actionable.

## Data and model behavior

- Historical training data is downloaded with `yfinance` using `period="10y"`.
- Recent prediction data is downloaded using `period="6mo"`; this exact value is required by current `yfinance` versions.
- `engineer_features()` creates lagged returns, moving averages, volatility, RSI, MACD, and Bollinger Band features.
- XGBoost predicts direction and LightGBM predicts next-period return magnitude.
- Cached models are written to `models/<TICKER>_model.pkl`.
- The duration dropdown currently describes the UI horizon but does not change the model target.

## Validation checklist

Before handing off a change:

1. Run a syntax check with the project's active Python interpreter:

   ```powershell
   python -m py_compile app.py
   ```

2. Confirm all `yfinance` period values are valid (`1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, or `max`).
3. If data-fetching code changed, test one known ticker such as `TSLA` or `AAPL`.
4. Launch the GUI when practical and verify that the prediction button, status label, and result box still work.
5. Avoid committing files under `models/` unless explicitly requested.

## Scope boundaries

Do not make investment recommendations or describe predictions as guaranteed. If changing the prediction horizon, update the target construction, UI labels, documentation, and validation together.

## Project architecture and context

This repository is a desktop-based machine-learning application for a B.Tech final-year major project (Batch 2027, GNIOT). Its intended architecture is a hybrid dual-model pipeline for short-term stock direction and return-magnitude forecasting, with automated feature engineering, model persistence, and an interactive Tkinter GUI.

### Problem context

- Short-term financial prices contain substantial noise and sentiment-driven movement; high apparent accuracy should be treated skeptically.
- Financial time series are non-stationary. Prefer normalized ratios, percentage changes, bounded oscillators, and rolling indicators over raw absolute-price features where appropriate.
- Random train/test shuffling can leak future information into the past. Any validation or tuning change must preserve chronological ordering.
- A result above roughly 80% daily directional accuracy should be investigated for leakage or overfitting. A modest, repeatable edge around 51.5%–54% is more realistic in this problem domain.

### Technical stack and responsibilities

- Python 3.10+
- Yahoo Finance (`yfinance`) for daily OHLCV data, normally with a ten-year lookback for training
- NumPy and pandas for vectorized feature engineering
- XGBoost (`XGBClassifier`) for binary direction: Up = `1`, Down = `0`
- LightGBM (`LGBMRegressor`) for expected percentage-return magnitude
- Optuna for Bayesian hyperparameter optimization when tuning is implemented
- `TimeSeriesSplit` for strict chronological cross-validation when tuning or evaluating models
- `joblib` for cached model serialization at `models/{ticker}_model.pkl`
- Tkinter for the desktop interface

### Feature engineering contract

The core feature pipeline should retain these meanings:

1. `Return`, `Lag_1`, and `Lag_2`: daily percentage return and temporal lags.
2. `SMA_10` and `SMA_50`: moving averages expressed as deviation from the current close.
3. `Volatility`: rolling ten-period return standard deviation.
4. `RSI`: fourteen-period bounded momentum oscillator; 70 and 30 are useful reference levels.
5. `MACD`, `MACD_Signal`, and `MACD_Hist`: trend-following momentum features based on exponential averages.
6. `BB_Position` and `BB_Width`: normalized twenty-period Bollinger Band position and volatility bandwidth using two standard deviations.

### Modeling and validation guidance

The dual-model strategy separates two questions: whether price will move up or down, and how large the next percentage move may be. Direction should be evaluated with directional accuracy and a classification report; magnitude should be evaluated with RMSE and MAE.

For production-quality tuning or evaluation, use five chronological `TimeSeriesSplit` folds. Each fold must train only on earlier observations and validate on later observations. Do not replace this with shuffled K-fold validation. When adding Optuna, ensure every trial uses the same chronological folds and that the final test period remains untouched until the end.

### Model caching and GUI behavior

The GUI should check `models/` for `{ticker}_model.pkl` before training. Existing models should be loaded lazily to avoid repeated training latency. If a model is missing, the app may download data, train, serialize the model, and then render the result. Keep long-running downloads and training off the Tkinter event loop so progress indicators and the interface remain responsive.

### Current implementation note

The current application already uses the dual-model design, feature pipeline, joblib caching, background prediction, and graphical interface. Optuna-based Bayesian tuning and `TimeSeriesSplit` cross-validation are architecture goals described by this context; if they are introduced, update the notebook, README, requirements, and validation checklist together so the documentation matches the executable code.

