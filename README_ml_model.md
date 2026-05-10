# Part 3 - Machine Learning Model Training (ml_model.py)

# Requirements
- Python 3.x
- scikit-learn
- pandas
- numpy
- stock_scanner.db (must be created first by running setup_db.py)

pip install scikit-learn pandas numpy

# How to Run

python ml_model.py

The program will automatically read from the database, engineer features, train the model, and save it.

# What it does
-Reads PRICE_HISTORY from stock_scanner.db and builds a feature dataset of triggered scan events
-Engineers 6 features per event: price_mom_5, true_range, rel_volume, pct_change, vol_trend, above_sma20
-Trains a Random Forest Classifier to predict whether a stock will continue rising the next day
-Saves the trained model to breakout_model.pkl
-Updates any PENDING scan results in the database with ML predictions

# Example output
Feature dataset: 739 triggered scan events across 23 stocks

── Model Performance ────────────────────────────────────────────
  Accuracy : 0.5608
  ROC-AUC  : 0.5881

  Classification Report:
                precision  recall  f1-score  support
  Reversal          0.53    0.51      0.52       69
  Continuation      0.59    0.61      0.60       79

  Confusion Matrix:
    TN=35  FP=34
    FN=31  TP=48

  Feature Importances:
    price_mom_5        0.2207  ████████
    true_range         0.2127  ████████
    rel_volume         0.1963  ███████
    pct_change         0.1886  ███████
    vol_trend          0.1721  ██████
    above_sma20        0.0097

Model saved: breakout_model.pkl

Updated 2 scan result(s) with ML predictions.
ML pipeline complete.

# Notes
-setup_db.py must be run before this script or it will fail to find stock_scanner.db
-Re-run this script any time new price data is added to retrain the model on the latest data
-The saved model file (breakout_model.pkl) is required by app.py at runtime
