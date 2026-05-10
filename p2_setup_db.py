import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
import os

DB_PATH = "stock_scanner.db"

# Ticker universe - Mix of large caps (market context), mid caps, and small/micro caps (the ones that actually trigger momentum scans).
# More can be added as the scanner works with any valid ticker.

TICKERS = [
    #Large cap tech (market context)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD",

    #Financials
    "JPM", "BAC", "GS",

    #Healthcare
    "JNJ", "PFE", "UNH",

    #Energy
    "XOM", "CVX",

    #ETFs
    "SPY", "QQQ", "IWM",

    #Crypto / Bitcoin adjacent
    "MSTR", "COIN", "HOOD", "MARA", "RIOT", "CLSK", "HUT",

    #AI / Tech small & mid caps
    "SOUN", "BBAI", "IONQ", "RXRX", "SMCI", "PLTR", "RBLX",
    "AFRM", "UPST", "SOFI", "U", "ACHR", "JOBY", "LILM",

    #Biotech momentum names
    "NVAX", "OCGN", "SAVA", "SRPT", "MRNA", "BNTX", "TDOC",

    #EV / Clean energy
    "NKLA", "WKHS", "BLNK", "PLUG", "FCEL", "BE", "ENPH",
    "SEDG", "RUN", "NOVA", "ARRY",

    #Gaming / consumer
    "DKNG", "PENN", "GENI",

    #Additional small cap momentum names
    "WOLF", "KSCP", "RCAT", "SPIR", "ACCD", "PHR", "DOCS",
]

# Schema
def create_schema(conn):
    conn.executescript("""
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS USER (
            UserID       INTEGER PRIMARY KEY AUTOINCREMENT,
            Username     TEXT NOT NULL UNIQUE,
            Email        TEXT NOT NULL UNIQUE,
            PasswordHash TEXT NOT NULL,
            CreatedAt    TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS STOCK (
            StockID     INTEGER PRIMARY KEY AUTOINCREMENT,
            Ticker      TEXT NOT NULL UNIQUE,
            CompanyName TEXT,
            Sector      TEXT,
            Exchange    TEXT,
            MarketCap   REAL
        );

        CREATE TABLE IF NOT EXISTS PRICE_HISTORY (
            PriceID    INTEGER PRIMARY KEY AUTOINCREMENT,
            StockID    INTEGER NOT NULL,
            Date       TEXT NOT NULL,
            OpenPrice  REAL,
            ClosePrice REAL,
            HighPrice  REAL,
            LowPrice   REAL,
            Volume     INTEGER,
            FOREIGN KEY (StockID) REFERENCES STOCK(StockID),
            UNIQUE(StockID, Date)
        );

        CREATE TABLE IF NOT EXISTS SCAN_CRITERIA (
            CriteriaID        INTEGER PRIMARY KEY AUTOINCREMENT,
            UserID            INTEGER NOT NULL,
            CriteriaName      TEXT NOT NULL,
            MinPctChange      REAL NOT NULL,
            MinRelativeVolume REAL NOT NULL,
            TimePeriodDays    INTEGER NOT NULL DEFAULT 20,
            SectorFilter      TEXT,
            FOREIGN KEY (UserID) REFERENCES USER(UserID)
        );

        CREATE TABLE IF NOT EXISTS SCAN_RESULT (
            ResultID       INTEGER PRIMARY KEY AUTOINCREMENT,
            CriteriaID     INTEGER NOT NULL,
            StockID        INTEGER NOT NULL,
            ScannedAt      TEXT NOT NULL DEFAULT (datetime('now')),
            PctChange      REAL,
            RelativeVolume REAL,
            MLPrediction   TEXT,
            FOREIGN KEY (CriteriaID) REFERENCES SCAN_CRITERIA(CriteriaID),
            FOREIGN KEY (StockID)    REFERENCES STOCK(StockID)
        );

        CREATE TABLE IF NOT EXISTS WATCHLIST (
            WatchlistID INTEGER PRIMARY KEY AUTOINCREMENT,
            UserID      INTEGER NOT NULL,
            StockID     INTEGER NOT NULL,
            AddedAt     TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (UserID)  REFERENCES USER(UserID),
            FOREIGN KEY (StockID) REFERENCES STOCK(StockID),
            UNIQUE(UserID, StockID)
        );

        CREATE TABLE IF NOT EXISTS ALERT (
            AlertID        INTEGER PRIMARY KEY AUTOINCREMENT,
            UserID         INTEGER NOT NULL,
            StockID        INTEGER NOT NULL,
            AlertType      TEXT NOT NULL,
            ThresholdValue REAL NOT NULL,
            TriggeredAt    TEXT,
            FOREIGN KEY (UserID)  REFERENCES USER(UserID),
            FOREIGN KEY (StockID) REFERENCES STOCK(StockID)
        );

        -- Indexes for query optimization
        CREATE INDEX IF NOT EXISTS idx_ph_stock_date ON PRICE_HISTORY(StockID, Date);
        CREATE INDEX IF NOT EXISTS idx_sr_criteria   ON SCAN_RESULT(CriteriaID);
        CREATE INDEX IF NOT EXISTS idx_sr_stock      ON SCAN_RESULT(StockID);
        CREATE INDEX IF NOT EXISTS idx_alert_user    ON ALERT(UserID);
    """)
    conn.commit()
    print("Schema created.\n")

#Fetch real data
def fetch_and_store(conn):
    c = conn.cursor()
    end   = datetime.today()
    start = end - timedelta(days=365)

    print(f"Fetching 1 year of data ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})")
    print(f"Universe: {len(TICKERS)} tickers\n")

    loaded   = 0
    skipped  = 0
    total_rows = 0

    for ticker in TICKERS:
        try:
            tk   = yf.Ticker(ticker)
            info = tk.info

            company  = info.get("longName") or info.get("shortName") or ticker
            sector   = info.get("sector", "Unknown")
            exchange = info.get("exchange", "Unknown")
            mktcap   = info.get("marketCap", None)

            c.execute("""
                INSERT OR IGNORE INTO STOCK (Ticker, CompanyName, Sector, Exchange, MarketCap)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker, company, sector, exchange, mktcap))
            conn.commit()

            stock_id = c.execute(
                "SELECT StockID FROM STOCK WHERE Ticker=?", (ticker,)
            ).fetchone()[0]

            hist = tk.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True
            )

            if hist.empty or len(hist) < 5:
                print(f"  {ticker:<8}: skipped (no data)")
                skipped += 1
                continue

            rows = [(
                stock_id,
                date.strftime("%Y-%m-%d"),
                round(float(row["Open"]),  4),
                round(float(row["Close"]), 4),
                round(float(row["High"]),  4),
                round(float(row["Low"]),   4),
                int(row["Volume"])
            ) for date, row in hist.iterrows()]

            c.executemany("""
                INSERT OR IGNORE INTO PRICE_HISTORY
                    (StockID, Date, OpenPrice, ClosePrice, HighPrice, LowPrice, Volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

            loaded     += 1
            total_rows += len(rows)
            print(f"  {ticker:<8}: {len(rows):>3} days | ${rows[-1][3]:<9.2f} | {company[:40]}")

        except Exception as e:
            print(f"  {ticker:<8}: ERROR - {e}")
            skipped += 1

    print(f"\nData lake: {loaded} stocks loaded, {skipped} skipped, {total_rows:,} price records.")
    return loaded

#Seed users and criteria
def seed_user_data(conn):
    c = conn.cursor()

    c.execute("""
        INSERT OR IGNORE INTO USER (Username, Email, PasswordHash)
        VALUES ('trader_leo', 'leo@stockscanner.com', 'bcrypt_hash_placeholder')
    """)
    conn.commit()
    uid = c.execute("SELECT UserID FROM USER WHERE Username='trader_leo'").fetchone()[0]

    """Scan criteria
    Aggressive: catches big movers (5%+ with 2x vol)
    Moderate:   catches mid moves (2%+ with 1.5x vol) — will fire most days
    Small cap:  tuned for volatile small caps (8%+ with 2x vol)
    Any move:   lowest bar (1.5%+ with 1.3x vol) — educational / broad scan"""

    c.executemany("""
        INSERT OR IGNORE INTO SCAN_CRITERIA
            (UserID, CriteriaName, MinPctChange, MinRelativeVolume, TimePeriodDays, SectorFilter)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [
        (uid, "Top Gainers - Aggressive",   5.0, 2.0, 20, None),
        (uid, "Moderate Movers - Any",      2.0, 1.5, 20, None),
        (uid, "Small Cap Momentum",         8.0, 2.0, 10, None),
        (uid, "Broad Daily Scan",           1.5, 1.3, 20, None),
        (uid, "Tech Breakout",              3.0, 2.0, 20, "Technology"),
    ])
    conn.commit()

    #Watchlist
    for tk in ("NVDA", "IONQ", "SOUN", "BBAI", "COIN", "MSTR", "PLTR"):
        row = c.execute("SELECT StockID FROM STOCK WHERE Ticker=?", (tk,)).fetchone()
        if row:
            c.execute("INSERT OR IGNORE INTO WATCHLIST (UserID, StockID) VALUES (?,?)",
                      (uid, row[0]))

    #Alert
    nvda = c.execute("SELECT StockID FROM STOCK WHERE Ticker='NVDA'").fetchone()
    if nvda:
        c.execute("""
            INSERT INTO ALERT (UserID, StockID, AlertType, ThresholdValue)
            VALUES (?, ?, 'RELATIVE_VOLUME', 2.5)
        """, (uid, nvda[0]))
    coin = c.execute("SELECT StockID FROM STOCK WHERE Ticker='COIN'").fetchone()
    if coin:
        c.execute("""
            INSERT INTO ALERT (UserID, StockID, AlertType, ThresholdValue)
            VALUES (?, ?, 'PCT_CHANGE', 5.0)
        """, (uid, coin[0]))

    conn.commit()
    print(f"\nSeeded: user (ID={uid}), 5 scan criteria, watchlist, 2 alerts.")
    return uid

#Scanner
def run_scanner(conn, user_id):
    c = conn.cursor()
    criteria = c.execute("""
        SELECT CriteriaID, CriteriaName, MinPctChange, MinRelativeVolume,
               TimePeriodDays, SectorFilter
        FROM SCAN_CRITERIA WHERE UserID=?
    """, (user_id,)).fetchall()

    print("\nRunning Scanner")
    total = 0

    for cid, name, min_pct, min_rvol, period, sector in criteria:
        print(f"\nCriteria: '{name}'")
        print(f"  Filter: >={min_pct}% change | >={min_rvol}x rel vol | {period}-day avg")

        if sector:
            stocks = c.execute(
                "SELECT StockID, Ticker FROM STOCK WHERE Sector=?", (sector,)
            ).fetchall()
        else:
            stocks = c.execute("SELECT StockID, Ticker FROM STOCK").fetchall()

        matches = []
        for sid, ticker in stocks:
            rows = c.execute("""
                SELECT Date, ClosePrice, Volume FROM PRICE_HISTORY
                WHERE StockID=? ORDER BY Date DESC LIMIT ?
            """, (sid, period + 1)).fetchall()

            if len(rows) < 2:
                continue

            pct  = ((rows[0][1] - rows[1][1]) / rows[1][1]) * 100 if rows[1][1] else 0
            avgv = sum(r[2] for r in rows[1:]) / len(rows[1:]) if len(rows) > 1 else 1
            rvol = rows[0][2] / avgv if avgv else 0

            if pct >= min_pct and rvol >= min_rvol:
                matches.append((ticker, pct, rvol, rows[0][0]))
                c.execute("""
                    INSERT INTO SCAN_RESULT
                        (CriteriaID, StockID, ScannedAt, PctChange, RelativeVolume, MLPrediction)
                    VALUES (?, ?, datetime('now'), ?, ?, 'PENDING')
                """, (cid, sid, round(pct, 2), round(rvol, 2)))

        conn.commit()
        matches.sort(key=lambda x: x[1], reverse=True)

        if matches:
            print(f"\n  {'Ticker':<8} {'% Change':>10} {'Rel Vol':>10}  {'Date'}")
            print(f"  {'-'*48}")
            for m in matches:
                print(f"  {m[0]:<8} {m[1]:>9.2f}%  {m[2]:>9.2f}x  {m[3]}")
            total += len(matches)
        else:
            print("  No matches on most recent trading day.")

    print(f"\nScanner complete. {total} total result(s) stored.")


if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Removed existing database.\n")

    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)

    loaded = fetch_and_store(conn)
    if loaded == 0:
        print("\nWARNING: No tickers loaded. Check internet connection.")
        conn.close()
        exit(1)

    uid = seed_user_data(conn)
    run_scanner(conn, uid)

    c = conn.cursor()
    print("\nDatabase Summary:")
    for tbl in ("STOCK","PRICE_HISTORY","SCAN_CRITERIA","SCAN_RESULT","WATCHLIST","ALERT"):
        print(f"  {tbl:<18}: {c.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]:>6} rows")

    conn.close()
    print(f"\nDatabase saved: {DB_PATH}")
    print("Next: run ml_model.py, then app.py")
