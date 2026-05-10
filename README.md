# Part 4 - End-to-End Stock Scanner Application (app.py)

# Requirements
- Python 3.x
- yfinance
- scikit-learn
- pandas
- numpy
- stock_scanner.db (created by setup_db.py)
- breakout_model.pkl (created by ml_model.py)

pip install yfinance scikit-learn pandas numpy

# How to Run

python app.py

The program will launch an interactive menu. All files must be in the same directory.

# Required run order (first time only)
Step 1 — build database and data lake (python setup_db.py)   
Step 2 — train ML model (python ml_model.py)
Step 3 — launch application (python app.py) 

# Menu options
[1] Run Scanner — select a saved criteria set and scan all stocks
[2] View Scan History — view the last 30 scan results with ML predictions
[3] Manage Watchlist — add or remove stocks from your watchlist
[4] Manage Alerts — configure and check price/volume threshold alerts
[5] Update Data Pipeline — fetch the latest prices from Yahoo Finance
[6] Database Summary — view row counts and database status
[0] Exit

# Run Scanner example
Select option: 1

Select scan criteria:

  [1] Top Gainers - Aggressive
      >= 5.0% change | >= 2.0x rel vol | 20-day avg

  [2] Broad Daily Scan
      >= 1.5% change | >= 1.3x rel vol | 20-day avg

Enter number (or 0 to cancel): 2

Running: 'Broad Daily Scan'...
────────────────────────────────────────────────────────────

Results for: 'Broad Daily Scan'
Scanned 63 stocks | 2026-05-09 14:32:11

  #    Ticker    % Change   Rel Vol   ML Prediction                       Date
  ────────────────────────────────────────────────────────────────────────────
  1    MARA       8.42%      3.21x    CONTINUATION (68% confidence)       2026-05-08
  2    COIN       5.17%      2.84x    REVERSAL (54% confidence)           2026-05-08

2 stock(s) matched. Results saved to database.

# Alert configuration example
Select option: 4

  [A] Add alert   [D] Delete alert   [C] Check alerts now   [0] Back

Choice: A
  Ticker: NVDA
  Alert types: RELATIVE_VOLUME, PCT_CHANGE, PRICE_TARGET
  Alert type: RELATIVE_VOLUME
  Threshold value: 2.5

  Alert created for NVDA (RELATIVE_VOLUME >= 2.5).

# Notes
- The ML model is loaded once at startup and applied to every scan result automatically
- If breakout_model.pkl is missing, the app will still run but ML predictions will show as N/A
- Use option [5] Update Data Pipeline to add new trading days without rebuilding the database
- After a data pipeline update, re-run ml_model.py to retrain the model on the latest data
 
