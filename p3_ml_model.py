import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import pickle, os

DB_PATH  = "stock_scanner.db"
MDL_PATH = "breakout_model.pkl"

def build_features(conn):
    """
    For each stock and each trading day, compute scanner features
    and label whether the NEXT day's close is higher (continuation=1).
    """
    c = conn.cursor()
    stocks = c.execute("SELECT StockID, Ticker FROM STOCK").fetchall()

    all_rows = []
    for sid, ticker in stocks:
        rows = c.execute("""
            SELECT Date, OpenPrice, ClosePrice, HighPrice, LowPrice, Volume
            FROM PRICE_HISTORY WHERE StockID=? ORDER BY Date ASC
        """, (sid,)).fetchall()

        if len(rows) < 25:
            continue

        df = pd.DataFrame(rows, columns=["Date","Open","Close","High","Low","Volume"])
        df["Date"] = pd.to_datetime(df["Date"])

        # Rolling metrics
        df["avg_vol_20"]   = df["Volume"].rolling(20).mean()
        df["avg_vol_5"]    = df["Volume"].rolling(5).mean()
        df["sma_20"]       = df["Close"].rolling(20).mean()
        df["pct_change"]   = df["Close"].pct_change() * 100
        df["rel_volume"]   = df["Volume"] / df["avg_vol_20"]
        df["vol_trend"]    = df["avg_vol_5"] / df["avg_vol_20"]
        df["price_mom_5"]  = df["Close"].pct_change(5) * 100
        df["true_range"]   = (df["High"] - df["Low"]).rolling(5).mean() / df["Close"]
        df["above_sma20"]  = (df["Close"] > df["sma_20"]).astype(int)

        # Label: did next day close higher?
        df["next_close"]   = df["Close"].shift(-1)
        df["label"]        = (df["next_close"] > df["Close"]).astype(int)
        df["Ticker"]       = ticker

        df = df.dropna()
        # Only keep days that would have triggered a scan (pct_change >= 3%, rel_vol >= 1.5x)
        triggered = df[(df["pct_change"] >= 3.0) | (df["rel_volume"] >= 1.5)]
        all_rows.append(triggered)

    full = pd.concat(all_rows, ignore_index=True)
    print(f"Feature dataset: {len(full)} triggered scan events across {len(stocks)} stocks")
    return full

FEATURES = ["pct_change","rel_volume","vol_trend","price_mom_5","true_range","above_sma20"]

def train_model(df):
    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print("\nModel Performance:")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Reversal","Continuation"]))
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    print("\n  Feature Importances:")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {feat:<18} {imp:.4f}  {bar}")

    return model, X_test, y_test

def update_scan_results(conn, model):
    """Apply ML predictions to all PENDING scan results in the DB."""
    c = conn.cursor()
    pending = c.execute("""
        SELECT sr.ResultID, sr.PctChange, sr.RelativeVolume,
               ph.Volume, s.StockID
        FROM SCAN_RESULT sr
        JOIN STOCK s ON sr.StockID = s.StockID
        JOIN PRICE_HISTORY ph ON ph.StockID = s.StockID
        WHERE sr.MLPrediction = 'PENDING'
        ORDER BY ph.Date DESC
        LIMIT 1
    """).fetchall()

    for row in pending:
        result_id, pct, rvol, vol, sid = row
        # Build a minimal feature vector with available data
        feat = np.array([[
            pct if pct else 0,
            rvol if rvol else 1,
            1.2,   # vol_trend placeholder
            pct * 0.6 if pct else 0,  # price_mom_5 approx
            0.02,  # true_range placeholder
            1      # assume above SMA on a big up day
        ]])
        prob = model.predict_proba(feat)[0][1]
        label = "CONTINUATION" if prob >= 0.5 else "REVERSAL"
        prediction = f"{label} ({prob:.0%} confidence)"
        c.execute("UPDATE SCAN_RESULT SET MLPrediction=? WHERE ResultID=?",
                  (prediction, result_id))

    conn.commit()
    print(f"\nUpdated {len(pending)} scan result(s) with ML predictions.")

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    df   = build_features(conn)

    if len(df) < 20:
        print("Not enough triggered events to train. Lowering threshold...")
        # If no events, use all rows
        c = conn.cursor()
        stocks = c.execute("SELECT StockID, Ticker FROM STOCK").fetchall()
        all_rows = []
        for sid, ticker in stocks:
            rows = c.execute("SELECT Date,OpenPrice,ClosePrice,HighPrice,LowPrice,Volume FROM PRICE_HISTORY WHERE StockID=? ORDER BY Date ASC",(sid,)).fetchall()
            if len(rows) < 25: continue
            df2 = pd.DataFrame(rows, columns=["Date","Open","Close","High","Low","Volume"])
            df2["avg_vol_20"] = df2["Volume"].rolling(20).mean()
            df2["avg_vol_5"]  = df2["Volume"].rolling(5).mean()
            df2["sma_20"]     = df2["Close"].rolling(20).mean()
            df2["pct_change"] = df2["Close"].pct_change() * 100
            df2["rel_volume"] = df2["Volume"] / df2["avg_vol_20"]
            df2["vol_trend"]  = df2["avg_vol_5"] / df2["avg_vol_20"]
            df2["price_mom_5"]= df2["Close"].pct_change(5) * 100
            df2["true_range"] = (df2["High"]-df2["Low"]).rolling(5).mean()/df2["Close"]
            df2["above_sma20"]= (df2["Close"]>df2["sma_20"]).astype(int)
            df2["next_close"] = df2["Close"].shift(-1)
            df2["label"]      = (df2["next_close"]>df2["Close"]).astype(int)
            df2["Ticker"]     = ticker
            all_rows.append(df2.dropna())
        df = pd.concat(all_rows, ignore_index=True)
        print(f"Using full dataset: {len(df)} rows")

    model, X_test, y_test = train_model(df)

    # Save model
    with open(MDL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": FEATURES}, f)
    print(f"\nModel saved: {MDL_PATH}")

    update_scan_results(conn, model)
    conn.close()
    print("ML pipeline complete.")
