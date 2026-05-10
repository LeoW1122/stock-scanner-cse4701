# Part 2 - Database Setup and Data Lake Population (setup_db.py)

# Requirements
- Python 3.x
- yfinance
- pandas
- numpy

pip install yfinance pandas numpy

# How to Run

python setup_db.py

The program will automatically build the database, fetch real market data, and run the scanner.

# What it does
-Creates the SQLite database (stock_scanner.db) with all 7 tables from the ER schema
-Fetches 1 year of real OHLCV price data from Yahoo Finance for 60+ tickers
-Seeds a demo user (trader_leo), 5 scan criteria, a watchlist, and 2 alerts
-Runs the top gainers scanner and stores any matching results

# Example output
Schema created.

Fetching 1 year of data (2025-05-09 to 2026-05-09)
Universe: 63 tickers

  AAPL   : 251 days | $293.32   | Apple Inc.
  MSFT   : 251 days | $415.12   | Microsoft Corporation
  NVDA   : 251 days | $215.20   | NVIDIA Corporation
  ...

Data lake complete: 63 stocks, 15,813 price records.

Seeded: user (ID=1), 5 scan criteria, watchlist, 2 alerts.

── Running Scanner ──────────────────────────────────────────────

Criteria: 'Broad Daily Scan'
  Filter: >=1.5% change | >=1.3x relative volume | 20-day avg

  Ticker   % Change   Rel Vol  Date
  ─────────────────────────────────────────────────────
  MARA      8.42%      3.21x   2026-05-08
  COIN      5.17%      2.84x   2026-05-08

── Database Summary ─────────────────────────────────────────────
  STOCK             :   63 rows
  PRICE_HISTORY     : 15813 rows
  SCAN_CRITERIA     :    5 rows
  SCAN_RESULT       :    2 rows
  WATCHLIST         :    7 rows
  ALERT             :    2 rows

Database saved: stock_scanner.db
Next: run ml_model.py

# Notes
-This script deletes and recreates the database every time it runs
-If you want to preserve existing data, use the Update Data Pipeline option inside app.py instead
-Run this script before ml_model.py and app.py
