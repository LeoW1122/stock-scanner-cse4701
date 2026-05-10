import sqlite3
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

DB_PATH  = "stock_scanner.db"
MDL_PATH = "breakout_model.pkl"

#Helpers
def clear():
    os.system("cls" if os.name == "nt" else "clear")

def divider(char="─", width=60):
    print(char * width)

def header(title):
    divider("═")
    print(f"  {title}")
    divider("═")

def pause():
    input("\n  Press Enter to continue...")

#Database connection
def get_conn():
    if not os.path.exists(DB_PATH):
        print(f"\n  ERROR: Database not found at '{DB_PATH}'")
        print("  Please run setup_db.py first.\n")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

#ML model loader
def load_model():
    if not os.path.exists(MDL_PATH):
        print(f"\n  WARNING: Model file not found at '{MDL_PATH}'")
        print("  ML predictions will be unavailable. Run ml_model.py to train.\n")
        return None, None
    with open(MDL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["features"]

#Feature computation for ML
def compute_features(conn, stock_id, period=20):
    """Compute ML features for a stock from its recent price history."""
    c = conn.cursor()
    rows = c.execute("""
        SELECT Date, OpenPrice, ClosePrice, HighPrice, LowPrice, Volume
        FROM PRICE_HISTORY
        WHERE StockID = ?
        ORDER BY Date DESC
        LIMIT 25
    """, (stock_id,)).fetchall()

    if len(rows) < 6:
        return None

    df = pd.DataFrame(rows, columns=["Date","Open","Close","High","Low","Volume"])
    df = df.iloc[::-1].reset_index(drop=True)  # oldest first

    avg_vol_20 = df["Volume"].tail(20).mean()
    avg_vol_5  = df["Volume"].tail(5).mean()
    sma_20     = df["Close"].tail(20).mean()

    latest     = df.iloc[-1]
    prev       = df.iloc[-2]

    pct_change  = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
    rel_volume  = latest["Volume"] / avg_vol_20 if avg_vol_20 else 1
    vol_trend   = avg_vol_5 / avg_vol_20 if avg_vol_20 else 1
    price_mom_5 = ((latest["Close"] - df.iloc[-6]["Close"]) / df.iloc[-6]["Close"]) * 100 if len(df) >= 6 else pct_change
    atr_5       = (df["High"].tail(5) - df["Low"].tail(5)).mean() / latest["Close"] if latest["Close"] else 0.02
    above_sma20 = 1 if latest["Close"] > sma_20 else 0

    return [pct_change, rel_volume, vol_trend, price_mom_5, atr_5, above_sma20]

#ML prediction
def predict(model, features, feat_vector):
    if model is None or feat_vector is None:
        return "N/A (model not loaded)"
    try:
        X = np.array([feat_vector])
        prob = model.predict_proba(X)[0][1]
        label = "CONTINUATION" if prob >= 0.5 else "REVERSAL"
        return f"{label} ({prob:.0%} confidence)"
    except Exception:
        return "N/A (prediction error)"


# CORE FEATURES
#1. Run Scanner
def run_scanner(conn, model, features, user_id):
    clear()
    header("RUN SCANNER")
    c = conn.cursor()

    criteria_list = c.execute("""
        SELECT CriteriaID, CriteriaName, MinPctChange, MinRelativeVolume,
               TimePeriodDays, SectorFilter
        FROM SCAN_CRITERIA WHERE UserID = ?
    """, (user_id,)).fetchall()

    if not criteria_list:
        print("  No scan criteria found. Please configure criteria first.")
        pause()
        return

    # Show criteria menu
    print("\n  Select scan criteria:\n")
    for i, (cid, name, pct, rvol, days, sector) in enumerate(criteria_list, 1):
        sf = f" | Sector: {sector}" if sector else ""
        print(f"  [{i}] {name}")
        print(f"      >= {pct}% change | >= {rvol}x rel vol | {days}-day avg{sf}\n")

    choice = input("  Enter number (or 0 to cancel): ").strip()
    if not choice.isdigit() or int(choice) == 0:
        return
    idx = int(choice) - 1
    if idx < 0 or idx >= len(criteria_list):
        print("  Invalid selection.")
        pause()
        return

    cid, name, min_pct, min_rvol, period, sector = criteria_list[idx]

    print(f"\n  Running: '{name}'...")
    divider()

    if sector:
        stocks = c.execute(
            "SELECT StockID, Ticker FROM STOCK WHERE Sector = ?", (sector,)
        ).fetchall()
    else:
        stocks = c.execute("SELECT StockID, Ticker FROM STOCK").fetchall()

    matches = []
    for stock_id, ticker in stocks:
        rows = c.execute("""
            SELECT Date, ClosePrice, Volume FROM PRICE_HISTORY
            WHERE StockID = ? ORDER BY Date DESC LIMIT ?
        """, (stock_id, period + 1)).fetchall()

        if len(rows) < 2:
            continue

        latest_close = rows[0][1];  prior_close = rows[1][1]
        latest_vol   = rows[0][2]
        avg_vol      = sum(r[2] for r in rows[1:]) / len(rows[1:]) if len(rows) > 1 else 1

        pct_change = ((latest_close - prior_close) / prior_close) * 100 if prior_close else 0
        rel_volume = latest_vol / avg_vol if avg_vol else 0

        if pct_change >= min_pct and rel_volume >= min_rvol:
            feat_vec = compute_features(conn, stock_id, period)
            ml_pred  = predict(model, features, feat_vec)

            matches.append((ticker, pct_change, rel_volume, rows[0][0], ml_pred, stock_id))

            c.execute("""
                INSERT INTO SCAN_RESULT
                    (CriteriaID, StockID, ScannedAt, PctChange, RelativeVolume, MLPrediction)
                VALUES (?, ?, datetime('now'), ?, ?, ?)
            """, (cid, stock_id, round(pct_change, 2), round(rel_volume, 2), ml_pred))

    conn.commit()
    matches.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Results for: '{name}'")
    print(f"  Scanned {len(stocks)} stocks | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if matches:
        print(f"  {'#':<4} {'Ticker':<8} {'% Change':>10} {'Rel Vol':>9}  {'ML Prediction':<35} {'Date'}")
        divider()
        for i, (tk, pct, rvol, date, pred, _) in enumerate(matches, 1):
            print(f"  {i:<4} {tk:<8} {pct:>9.2f}%  {rvol:>8.2f}x  {pred:<35} {date}")
        print(f"\n  {len(matches)} stock(s) matched. Results saved to database.")
    else:
        print("  No stocks matched on the most recent trading day.")
        print("  This is normal for large-cap stocks — try lowering thresholds")
        print("  or wait for a high-volatility market session.")

    pause()

#2. View Top Gainers
def view_top_gainers(conn, user_id):
    clear()
    header("TOP GAINERS – SCAN HISTORY")
    c = conn.cursor()

    results = c.execute("""
        SELECT s.Ticker, sr.PctChange, sr.RelativeVolume,
               sr.ScannedAt, sr.MLPrediction, sc.CriteriaName
        FROM SCAN_RESULT sr
        JOIN STOCK s       ON sr.StockID    = s.StockID
        JOIN SCAN_CRITERIA sc ON sr.CriteriaID = sc.CriteriaID
        WHERE sc.UserID = ?
        ORDER BY sr.ScannedAt DESC, sr.PctChange DESC
        LIMIT 30
    """, (user_id,)).fetchall()

    if not results:
        print("\n  No scan results found. Run the scanner first.")
        pause()
        return

    print(f"\n  {'Ticker':<8} {'% Change':>10} {'Rel Vol':>9}  {'ML Prediction':<35} {'Criteria':<28} {'Scanned At'}")
    divider(width=130)
    for tk, pct, rvol, scanned, pred, crit in results:
        pct_str  = f"{pct:.2f}%" if pct is not None else "N/A"
        rvol_str = f"{rvol:.2f}x" if rvol is not None else "N/A"
        pred_str = pred if pred else "PENDING"
        print(f"  {tk:<8} {pct_str:>10} {rvol_str:>9}  {pred_str:<35} {crit:<28} {scanned}")

    pause()

#3. Manage Watchlist
def manage_watchlist(conn, user_id):
    clear()
    header("WATCHLIST")
    c = conn.cursor()

    while True:
        clear()
        header("WATCHLIST")
        watchlist = c.execute("""
            SELECT w.WatchlistID, s.Ticker, s.CompanyName, s.Sector,
                   ph.ClosePrice, w.AddedAt
            FROM WATCHLIST w
            JOIN STOCK s ON w.StockID = s.StockID
            LEFT JOIN PRICE_HISTORY ph ON ph.StockID = s.StockID
                AND ph.Date = (SELECT MAX(Date) FROM PRICE_HISTORY WHERE StockID = s.StockID)
            WHERE w.UserID = ?
            ORDER BY w.AddedAt DESC
        """, (user_id,)).fetchall()

        if watchlist:
            print(f"\n  {'#':<4} {'Ticker':<8} {'Company':<32} {'Sector':<22} {'Last Close':>10}  {'Added'}")
            divider()
            for i, (wid, tk, co, sec, close, added) in enumerate(watchlist, 1):
                close_str = f"${close:.2f}" if close else "N/A"
                print(f"  {i:<4} {tk:<8} {co:<32} {sec:<22} {close_str:>10}  {added[:10]}")
        else:
            print("\n  Your watchlist is empty.")

        print("\n  [A] Add stock   [R] Remove stock   [0] Back")
        action = input("\n  Choice: ").strip().upper()

        if action == "0":
            break
        elif action == "A":
            ticker = input("  Enter ticker symbol: ").strip().upper()
            row = c.execute("SELECT StockID FROM STOCK WHERE Ticker = ?", (ticker,)).fetchone()
            if not row:
                print(f"  '{ticker}' not found in database.")
            else:
                try:
                    c.execute("INSERT INTO WATCHLIST (UserID, StockID) VALUES (?, ?)",
                              (user_id, row[0]))
                    conn.commit()
                    print(f"  Added {ticker} to watchlist.")
                except sqlite3.IntegrityError:
                    print(f"  {ticker} is already on your watchlist.")
            pause()
        elif action == "R":
            if not watchlist:
                continue
            num = input("  Enter # to remove: ").strip()
            if num.isdigit() and 1 <= int(num) <= len(watchlist):
                wid = watchlist[int(num)-1][0]
                c.execute("DELETE FROM WATCHLIST WHERE WatchlistID = ?", (wid,))
                conn.commit()
                print(f"  Removed from watchlist.")
                pause()

#4. Manage Alerts
def manage_alerts(conn, user_id):
    clear()
    header("ALERTS")
    c = conn.cursor()

    while True:
        clear()
        header("ALERTS")
        alerts = c.execute("""
            SELECT a.AlertID, s.Ticker, a.AlertType, a.ThresholdValue, a.TriggeredAt
            FROM ALERT a
            JOIN STOCK s ON a.StockID = s.StockID
            WHERE a.UserID = ?
            ORDER BY a.AlertID DESC
        """, (user_id,)).fetchall()

        if alerts:
            print(f"\n  {'#':<4} {'Ticker':<8} {'Type':<20} {'Threshold':>12}  {'Triggered At'}")
            divider()
            for i, (aid, tk, atype, thresh, triggered) in enumerate(alerts, 1):
                trig_str = triggered if triggered else "Not yet triggered"
                print(f"  {i:<4} {tk:<8} {atype:<20} {thresh:>12.2f}  {trig_str}")
        else:
            print("\n  No alerts configured.")

        print("\n  [A] Add alert   [D] Delete alert   [C] Check alerts now   [0] Back")
        action = input("\n  Choice: ").strip().upper()

        if action == "0":
            break
        elif action == "A":
            ticker = input("  Ticker: ").strip().upper()
            row = c.execute("SELECT StockID FROM STOCK WHERE Ticker=?", (ticker,)).fetchone()
            if not row:
                print(f"  '{ticker}' not found.")
                pause()
                continue
            print("  Alert types: RELATIVE_VOLUME, PCT_CHANGE, PRICE_TARGET")
            atype  = input("  Alert type: ").strip().upper()
            thresh = input("  Threshold value: ").strip()
            try:
                c.execute("""
                    INSERT INTO ALERT (UserID, StockID, AlertType, ThresholdValue)
                    VALUES (?, ?, ?, ?)
                """, (user_id, row[0], atype, float(thresh)))
                conn.commit()
                print(f"  Alert created for {ticker} ({atype} >= {thresh}).")
            except ValueError:
                print("  Invalid threshold value.")
            pause()
        elif action == "D":
            if not alerts:
                continue
            num = input("  Enter # to delete: ").strip()
            if num.isdigit() and 1 <= int(num) <= len(alerts):
                aid = alerts[int(num)-1][0]
                c.execute("DELETE FROM ALERT WHERE AlertID=?", (aid,))
                conn.commit()
                print("  Alert deleted.")
                pause()
        elif action == "C":
            # Evaluate all alerts against latest price data
            print("\n  Checking alerts...\n")
            triggered_count = 0
            for aid, tk, atype, thresh, _ in alerts:
                sid = c.execute("SELECT StockID FROM STOCK WHERE Ticker=?", (tk,)).fetchone()[0]
                rows = c.execute("""
                    SELECT ClosePrice, Volume FROM PRICE_HISTORY
                    WHERE StockID=? ORDER BY Date DESC LIMIT 21
                """, (sid,)).fetchall()
                if len(rows) < 2:
                    continue
                close    = rows[0][0]
                vol      = rows[0][1]
                avg_vol  = sum(r[1] for r in rows[1:]) / len(rows[1:])
                prev_close = rows[1][0]
                pct      = ((close - prev_close) / prev_close) * 100 if prev_close else 0
                rvol     = vol / avg_vol if avg_vol else 0

                triggered = False
                if atype == "RELATIVE_VOLUME"  and rvol  >= thresh: triggered = True
                if atype == "PCT_CHANGE"        and pct   >= thresh: triggered = True
                if atype == "PRICE_TARGET"      and close >= thresh: triggered = True

                if triggered:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    c.execute("UPDATE ALERT SET TriggeredAt=? WHERE AlertID=?", (now, aid))
                    print(f"  TRIGGERED: {tk} {atype} = "
                          f"{rvol:.2f}x / {pct:.2f}% / ${close:.2f} (threshold: {thresh})")
                    triggered_count += 1

            conn.commit()
            if triggered_count == 0:
                print("  No alerts triggered on current data.")
            pause()

#5. Update Data Pipeline
def update_data_pipeline(conn):
    clear()
    header("UPDATE DATA PIPELINE")
    print("\n  This will fetch the latest price data from Yahoo Finance")
    print("  and update the database with any new trading days.\n")
    confirm = input("  Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        return

    c = conn.cursor()
    stocks = c.execute("SELECT StockID, Ticker FROM STOCK").fetchall()
    end   = datetime.today()
    start = end - timedelta(days=7)  # fetch last 7 days to catch any gaps

    print(f"\n  Fetching data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}...\n")
    updated = 0
    new_rows = 0

    for stock_id, ticker in stocks:
        try:
            tk   = yf.Ticker(ticker)
            hist = tk.history(start=start.strftime("%Y-%m-%d"),
                              end=end.strftime("%Y-%m-%d"), auto_adjust=True)
            if hist.empty:
                continue

            rows = []
            for date, row in hist.iterrows():
                rows.append((
                    stock_id, date.strftime("%Y-%m-%d"),
                    round(float(row["Open"]),  4),
                    round(float(row["Close"]), 4),
                    round(float(row["High"]),  4),
                    round(float(row["Low"]),   4),
                    int(row["Volume"])
                ))

            before = c.execute("SELECT COUNT(*) FROM PRICE_HISTORY WHERE StockID=?",
                               (stock_id,)).fetchone()[0]
            c.executemany("""
                INSERT OR IGNORE INTO PRICE_HISTORY
                    (StockID, Date, OpenPrice, ClosePrice, HighPrice, LowPrice, Volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            after = c.execute("SELECT COUNT(*) FROM PRICE_HISTORY WHERE StockID=?",
                              (stock_id,)).fetchone()[0]
            added = after - before
            if added > 0:
                print(f"  {ticker}: +{added} new day(s)")
                new_rows += added
                updated  += 1

        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")

    if new_rows > 0:
        print(f"\n  Update complete: {new_rows} new records added across {updated} stocks.")
        print("  Consider re-running ml_model.py to retrain with the latest data.")
    else:
        print("\n  Database is already up to date.")

    pause()

#6. Database Summary
def view_db_summary(conn):
    clear()
    header("DATABASE SUMMARY")
    c = conn.cursor()
    print()
    tables = ["STOCK","PRICE_HISTORY","SCAN_CRITERIA","SCAN_RESULT","WATCHLIST","ALERT","USER"]
    for tbl in tables:
        count = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:<18}: {count:>6} rows")

    # Latest price date
    latest = c.execute("SELECT MAX(Date) FROM PRICE_HISTORY").fetchone()[0]
    print(f"\n  Latest price date : {latest}")

    # Recent scan results
    recent = c.execute("""
        SELECT COUNT(*) FROM SCAN_RESULT
        WHERE ScannedAt >= date('now', '-7 days')
    """).fetchone()[0]
    print(f"  Scan results (7d) : {recent}")
    print(f"\n  Database file     : {os.path.abspath(DB_PATH)}")
    print(f"  Model file        : {os.path.abspath(MDL_PATH) if os.path.exists(MDL_PATH) else 'NOT FOUND'}")
    pause()


# MAIN MENU
def main():
    conn  = get_conn()
    model, features = load_model()

    # Get or create default user
    c = conn.cursor()
    user = c.execute("SELECT UserID, Username FROM USER LIMIT 1").fetchone()
    if not user:
        print("No users found. Please run setup_db.py first.")
        conn.close()
        sys.exit(1)
    user_id, username = user

    while True:
        clear()
        header("STOCK SCANNER ANALYSIS TOOL")
        print(f"\n  Logged in as: {username}")
        ml_status = "Loaded" if model else "Not loaded (run ml_model.py)"
        print(f"  ML Model    : {ml_status}")
        print(f"  Database    : {DB_PATH}\n")
        divider()
        print("\n  [1] Run Scanner")
        print("  [2] View Scan History / Top Gainers")
        print("  [3] Manage Watchlist")
        print("  [4] Manage Alerts")
        print("  [5] Update Data Pipeline (fetch latest prices)")
        print("  [6] Database Summary")
        print("  [0] Exit\n")
        divider()

        choice = input("\n  Select option: ").strip()

        if   choice == "1": run_scanner(conn, model, features, user_id)
        elif choice == "2": view_top_gainers(conn, user_id)
        elif choice == "3": manage_watchlist(conn, user_id)
        elif choice == "4": manage_alerts(conn, user_id)
        elif choice == "5": update_data_pipeline(conn)
        elif choice == "6": view_db_summary(conn)
        elif choice == "0":
            print("\n  Goodbye.\n")
            conn.close()
            break
        else:
            print("  Invalid option.")

if __name__ == "__main__":
    main()
