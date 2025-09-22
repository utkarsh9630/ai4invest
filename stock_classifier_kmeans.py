#!/usr/bin/env python3
import pandas as pd
import numpy as np
import yfinance as yf
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# 1) Get S&P 500 tickers from Wikipedia
wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500    = pd.read_html(wiki_url, header=0)[0]
symbols  = sp500["Symbol"].str.replace(r"\.", "-", regex=True).tolist()

# 2) Top 100 by avg daily volume over last 60 days
vol_data = yf.download(symbols, period="60d", group_by="ticker", auto_adjust=True)
avg_vol  = {
    sym: vol_data[sym]["Volume"].mean()
    for sym in symbols
    if sym in vol_data.columns.get_level_values(0)
}
top100 = sorted(avg_vol, key=avg_vol.get, reverse=True)[:100]
print(f"Using top 100 by volume: {top100}\n")

# 3) Download 1 year of daily price data for these 100 stocks
prices = yf.download(
    tickers=top100,
    period="1y",
    auto_adjust=True,
    group_by="ticker",
    threads=True
)

# 4) Fetch SPY for beta calculation
df_spy = yf.download("SPY", period="1y", auto_adjust=True)
spy    = df_spy["Close"].pct_change().dropna()
spy.name = "SPY"

# 5) Build price‐based features
feat_list = []
for sym in top100:
    df = prices[sym]["Close"].pct_change().to_frame("ret").dropna()
    df["vol30"] = df["ret"].rolling(30).std()
    df["mom30"]  = prices[sym]["Close"].pct_change(30)

    comb = pd.concat([df["ret"], spy], axis=1, join="inner").dropna()
    cov  = comb["ret"].rolling(60).cov(comb["SPY"])
    var  = comb["SPY"].rolling(60).var()
    df["beta60"] = cov.div(var).reindex(df.index)

    last = df.dropna().iloc[-1].to_dict()
    last["ticker"] = sym
    feat_list.append(last)

features_df = pd.DataFrame(feat_list).set_index("ticker")
print("Price-based features:\n", features_df.head(), "\n")

# 6) Build fundamental features
funds = []
for sym in features_df.index:
    info = yf.Ticker(sym).info
    funds.append({
        "ticker":    sym,
        "pe":        info.get("trailingPE",    np.nan),
        "pb":        info.get("priceToBook",   np.nan),
        "div_yield": info.get("dividendYield", np.nan),
    })
fund_df = pd.DataFrame(funds).set_index("ticker")
print("Fundamentals (PE, PB, dividend yield):\n", fund_df.head(), "\n")

# 7) Merge & impute
data = features_df.join(fund_df)
print("Missing values before imputation:\n", data.isna().sum(), "\n")
imp = SimpleImputer(strategy="median")
data_imputed = pd.DataFrame(
    imp.fit_transform(data),
    columns=data.columns,
    index=data.index
)
print("Missing values after imputation:\n", data_imputed.isna().sum(), "\n")

# 8) Robust‐scale & export
data_imputed["vol30_log"] = np.log1p(data_imputed["vol30"])
feat_cols = ["vol30_log", "mom30", "beta60"]

rs   = RobustScaler().fit(data_imputed[feat_cols])
X_rs = rs.transform(data_imputed[feat_cols])

# wrap in DataFrame so we can label and save
X_rs_df = pd.DataFrame(
    X_rs,
    index=data_imputed.index,
    columns=[f"{c}_scaled" for c in feat_cols]
)
X_rs_df.to_csv("stock_level_data.csv")
print("✅ Saved scaled feature matrix to stock_level_data.csv\n")

# --- Evaluation metrics for clusters k=2..6 ---
print("Cluster evaluation metrics:")
for k in range(2, 7):
    km_k     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km_k.fit_predict(X_rs_df)
    sil = silhouette_score(X_rs_df, labels_k)
    ch  = calinski_harabasz_score(X_rs_df, labels_k)
    db  = davies_bouldin_score(X_rs_df, labels_k)
    print(f" k={k}: silhouette={sil:.4f}, CH={ch:.1f}, DB={db:.4f}")
print("\n")

# 9) Fit final KMeans and report metrics
kmeans3     = KMeans(n_clusters=3, random_state=42, n_init=50).fit(X_rs_df)
labels_final = kmeans3.labels_
sil_f = silhouette_score(X_rs_df, labels_final)
ch_f  = calinski_harabasz_score(X_rs_df, labels_final)
db_f  = davies_bouldin_score(X_rs_df, labels_final)
print(f"Final KMeans(k=3): silhouette={sil_f:.4f}, CH={ch_f:.1f}, DB={db_f:.4f}\n")

# 10) Map clusters → risk labels
centroids = pd.DataFrame(
    rs.inverse_transform(kmeans3.cluster_centers_),
    columns=feat_cols
)
order    = centroids["vol30_log"].sort_values().index.tolist()
risk_map = {order[i]: lab for i, lab in enumerate(["Low", "Medium", "High"])}

data_imputed["cluster"]    = labels_final
data_imputed["risk_label"] = data_imputed["cluster"].map(risk_map)
print("Final risk counts:\n", data_imputed["risk_label"].value_counts(), "\n")

# 11) Export artifacts
features_df.reset_index().to_csv("sp500_features.csv", index=False)
stock_clf_pipeline = Pipeline([
    ("robust_scaler", rs),
    ("kmeans",       kmeans3)
])
joblib.dump(stock_clf_pipeline, "stock_classifier.joblib")
print("✅ Artifacts saved: stock_classifier.joblib, sp500_features.csv")
