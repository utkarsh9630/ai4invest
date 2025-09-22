#!/usr/bin/env python3
"""
full_pipeline.py

1) Load S&P 500 feature CSV (Model-2 output).
2) Compute log-volatility feature for classification.
3) Use Model-2 pipeline to assign each stock a risk_label2.
4) Fetch last 180 days of adjusted Close prices for each ticker + SPY via yfinance.
5) Compute vol30, mom30, beta60, and 90-day forward return.
6) Train RandomForestRegressor → future_return.
7) Predict returns and select top N stocks per risk bucket.
8) Write recommendations to CSV.
9) Export the trained regressor as a joblib artifact.
"""
import time
import pandas as pd
import numpy as np
import yfinance as yf
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# --- PARAMETERS ---
FEATURE_CSV    = 'sp500_features.csv'        # from Model-2 export
CLASSIFIER_JOB = 'stock_classifier.joblib'   # saved classifier pipeline
LOOKAHEAD_DAYS = 90                          # days ahead for target
TOP_N          = 5                           # picks per bucket
SLEEP_SEC      = 1                           # pause between fetches
OUTPUT_FILE    = 'top_n_per_category.csv'    # recommendations output
MODEL_ARTIFACT = 'topreturn_model.joblib'    # export filename

# 1) Load feature dataframe & classifier pipeline
print(f"▶ Loading features from '{FEATURE_CSV}' and classifier from '{CLASSIFIER_JOB}'...")
feat_df = pd.read_csv(FEATURE_CSV)
clf_pipe = joblib.load(CLASSIFIER_JOB)

# 2) Prepare classifier input: compute log-volatility
print("▶ Computing vol30_log for classification...")
feat_df['vol30_log'] = np.log1p(feat_df['vol30'])
X_clf = feat_df[['vol30_log', 'mom30', 'beta60']]

# 3) Assign risk_label2 via Model-2 pipeline
print("▶ Assigning risk labels to universe...")
feat_df['risk_label2'] = clf_pipe.predict(X_clf)
print(f"  → Risk buckets: {feat_df['risk_label2'].value_counts().to_dict()}\n")

# Prepare for time-series features
df0 = feat_df.set_index('ticker')['risk_label2']
tickers = df0.index.tolist()

# Helper: fetch Close series via yfinance
def fetch_close(symbol):
    try:
        df = yf.download(symbol, period='180d', auto_adjust=True, progress=False)
        return df['Close'].sort_index()
    except Exception as e:
        print(f"❌ Fetch failed for {symbol}: {e}")
        return pd.Series(dtype=float)

# 4) Fetch SPY price series for beta
def get_spy():
    ser = fetch_close('SPY').pct_change().dropna()
    ser.name = 'SPY'
    return ser

print("▶ Fetching SPY series...")
spy_series = get_spy()
time.sleep(SLEEP_SEC)

# 5) Fetch each ticker's history
print("▶ Fetching ticker price history...")
price_hist = {}
for sym in tickers:
    series = fetch_close(sym)
    if not series.empty:
        price_hist[sym] = series
    else:
        print(f"⚠️ No data for {sym}, skipping")
    time.sleep(SLEEP_SEC)
print(f"  → Retrieved data for {len(price_hist)} symbols.\n")

# 6) Build feature & target rows
def build_rows(price_hist, spy_series):
    rows = []
    for sym, series in price_hist.items():
        if len(series) < LOOKAHEAD_DAYS + 1:
            continue
        ret = series.pct_change().dropna()
        vol30 = ret.rolling(30).std().iloc[-1] if len(ret) >= 30 else np.nan
        mom30 = series.pct_change(30).iloc[-1] if len(series) >= 30 else np.nan
        comb = pd.concat([ret, spy_series], axis=1, join='inner').dropna()
        if len(comb) >= 60:
            cov = comb.iloc[:,0].rolling(60).cov(comb['SPY'])
            var = comb['SPY'].rolling(60).var()
            beta60 = (cov/var).iloc[-1]
        else:
            beta60 = np.nan
        future_ret = series.iloc[-1] / series.shift(LOOKAHEAD_DAYS).iloc[-1] - 1
        risk = df0.loc[sym]
        rows.append({
            'ticker': sym,
            'vol30': vol30,
            'mom30': mom30,
            'beta60': beta60,
            'future_return': future_ret,
            'risk_label2': risk
        })
    return rows

print("▶ Building features & target...")
rows = build_rows(price_hist, spy_series)
feat_df2 = pd.DataFrame(rows).set_index('ticker')
# drop missing
mask = feat_df2[['vol30','mom30','beta60','future_return']].isnull().any(axis=1)
if mask.any():
    print(f"⚠️ Dropping {mask.sum()} rows with missing values")
feat_df2 = feat_df2[~mask]
if feat_df2.empty:
    raise RuntimeError("No valid data rows!")
print(f"  → {len(feat_df2)} rows after cleaning.\n")

# 7) Prepare X, y arrays
X_raw = feat_df2[['vol30','mom30','beta60']]
y = feat_df2['future_return'].astype(float).to_numpy()

# 8) Impute missing values
print("▶ Imputing missing features...")
imp = SimpleImputer(strategy='median')
X_imp = imp.fit_transform(X_raw)

# 9) Scale
print("▶ Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# 10) Train regressor
print("▶ Training RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)
feat_df2['pred_return'] = model.predict(X_scaled)

# 11) Select top N per bucket
print(f"▶ Selecting top {TOP_N} per bucket...")
results=[]
for b in feat_df2['risk_label2'].unique():
    sub = feat_df2[feat_df2['risk_label2']==b]
    topn = sub.nlargest(TOP_N,'pred_return').copy()
    topn['bucket']=b
    results.append(topn)
final_df = pd.concat(results)
final_df.to_csv(OUTPUT_FILE)
print(f"✅ Recommendations saved to '{OUTPUT_FILE}'\n")

# 12) Export model
print(f"▶ Exporting model to {MODEL_ARTIFACT}...")
joblib.dump(model, MODEL_ARTIFACT)
print(f"✅ Model-3 saved: {MODEL_ARTIFACT}")
