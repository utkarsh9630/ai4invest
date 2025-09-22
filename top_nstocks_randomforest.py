#!/usr/bin/env python3 # needs the input file from stock_classifier_kmeans, should figure out AWS integration later
"""
full_pipeline.py

1) Read your CSV of tickers & risk buckets.
2) Fetch 180 days of EOD closes for each ticker + SPY via Stooq.
3) Compute vol30, mom30, beta60, and 90‑day forward return.
4) Train a RandomForestRegressor on those features → future_return.
5) Predict returns and pick top N in each risk bucket.
6) Write results to CSV.
"""

import time
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# PARAMETERS

INPUT_FILE     = "/content/stock_risk_kmeans_robust.csv"         # CSV output from stock_classifier_kmeans.ipynb
LOOKAHEAD_DAYS = 90                       # days ahead for target return
TOP_N          = 5                        # picks per bucket
SLEEP_SEC      = 1                        # pause between fetches
OUTPUT_FILE    = "top_n_per_category.csv" # final recommendations

# Load tickers and risk labels
df0     = pd.read_csv(INPUT_FILE, index_col="ticker")
tickers = df0.index.tolist()

#  Helper: fetch Close series from Stooq
def fetch_close(symbol):
    try:
        df = pdr.DataReader(symbol, "stooq")
        return df.sort_index()["Close"]
    except Exception as e:
        print(f"Fetch failed for {symbol}: {e}")
        return pd.Series(dtype=float)


# 3) Fetch SPY for beta
spy_series = fetch_close("SPY").pct_change().dropna().rename("SPY")
time.sleep(SLEEP_SEC)

# 4) Fetch each ticker’s history
price_hist = {}
for sym in tickers:
    series = fetch_close(sym)
    if not series.empty:
        price_hist[sym] = series
    else:
        print(f"✗ No data for {sym}, skipping")
    time.sleep(SLEEP_SEC)

# 5) Build feature + target rows
rows = []
for sym, series in price_hist.items():
    if len(series) < LOOKAHEAD_DAYS + 1:
        print(f"{sym}: insufficient history, skipping")
        continue

    # a) 30‑day volatility
    ret = series.pct_change().dropna()
    vol30 = ret.rolling(30).std().iloc[-1] if len(ret) >= 30 else np.nan

    # b) 30‑day momentum
    mom30 = series.pct_change(30).iloc[-1] if len(series) >= 30 else np.nan

    # c) 60‑day beta vs SPY
    comb = pd.concat([ret, spy_series], axis=1, join="inner").dropna()
    if len(comb) < 60:
        beta60 = np.nan
    else:
        cov = comb.iloc[:,0].rolling(60).cov(comb["SPY"])
        var = comb["SPY"].rolling(60).var()
        beta60 = (cov/var).iloc[-1]

    # d) 90‑day forward return
    future_ret = series.iloc[-1] / series.shift(LOOKAHEAD_DAYS).iloc[-1] - 1

    # e) risk label
    risk = df0.loc[sym, "risk_label2"]

    rows.append({
        "ticker":         sym,
        "vol30":          vol30,
        "mom30":          mom30,
        "beta60":         beta60,
        "future_return":  future_ret,
        "risk_label2":    risk
    })

feat_df = pd.DataFrame(rows).set_index("ticker")
if feat_df.empty:
    raise RuntimeError("No valid data to build features/targets!")

#train the model
X_raw = feat_df[["vol30","mom30","beta60"]]
y     = feat_df["future_return"]

# Impute missing feature values
imp    = SimpleImputer(strategy="median")
X_imp  = imp.fit_transform(X_raw)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)


# 7) Train RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

feat_df["pred_return"] = model.predict(X_scaled)

# 9) Select top N in each risk bucket

results = []
for bucket in feat_df["risk_label2"].unique():
    subset = feat_df[feat_df["risk_label2"] == bucket]
    topn   = subset.nlargest(TOP_N, "pred_return").copy()
    topn["bucket"] = bucket
    results.append(topn)

final_df = pd.concat(results)
final_df.to_csv(OUTPUT_FILE)

print(f"Top {TOP_N} picks per bucket written to '{OUTPUT_FILE}'")
print(final_df)

# ——————————————————————————————————————————————
# EXPORT MODEL-3: Return‐Forecast Model

import joblib

# Assume your trained regressor is named `rf`
joblib.dump(rf, 'topreturn_model.joblib')

print("✅ Model-3 artifact saved: topreturn_model.joblib")
