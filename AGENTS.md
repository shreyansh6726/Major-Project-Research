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

