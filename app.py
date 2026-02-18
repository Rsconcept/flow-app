import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Institutional Options Signals", layout="wide")

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "NVDA", "AMD"]

EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", "")
UW_BEARER = st.secrets.get("UW_BEARER", "")

# ==============================
# HELPERS
# ==============================
def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ------------------------------
# EODHD INTRADAY DATA (FIXED)
# ------------------------------
def get_intraday_data(ticker, interval="5m"):
    try:
        url = f"https://eodhd.com/api/intraday/{ticker}.US"
        params = {
            "api_token": EODHD_API_KEY,
            "interval": interval,
            "fmt": "json"
        }
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        df = pd.DataFrame(data)

        if df.empty:
            return None

        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        return df

    except:
        return None

# ------------------------------
# 10Y YIELD FILTER
# ------------------------------
def get_10y_yield():
    try:
        url = "https://eodhd.com/api/real-time/US10Y.BOND"
        params = {"api_token": EODHD_API_KEY}
        r = requests.get(url, params=params)
        return float(r.json().get("close", 0))
    except:
        return 0

# ------------------------------
# UNUSUAL WHALES OPTIONS VOLUME
# ------------------------------
def get_uw_options_volume(ticker):
    try:
        url = f"https://api.unusualwhales.com/api/stock/{ticker}/options-volume"
        headers = {
            "Authorization": f"Bearer {UW_BEARER}",
            "Accept": "application/json"
        }
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json()["data"][0]
        return data
    except:
        return None

# ==============================
# INDICATORS
# ==============================
def calculate_indicators(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["EMA9"] = close.ewm(span=9).mean()
    df["EMA20"] = close.ewm(span=20).mean()
    df["EMA50"] = close.ewm(span=50).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # VWAP
    df["VWAP"] = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()

    # Volume ratio
    df["Vol_ratio"] = volume / volume.rolling(20).mean()

    return df

# ==============================
# INSTITUTIONAL SCORING
# ==============================
def institutional_score(df, ticker):

    if df is None or len(df) < 50:
        return 50, "WAIT"

    df = calculate_indicators(df)
    last = df.iloc[-1]

    score = 50

    # Trend bias
    if last["EMA9"] > last["EMA20"] > last["EMA50"]:
        score += 10
    if last["EMA9"] < last["EMA20"] < last["EMA50"]:
        score -= 10

    # VWAP bias
    if last["close"] > last["VWAP"]:
        score += 10
    else:
        score -= 10

    # RSI
    if last["RSI"] > 60:
        score += 10
    if last["RSI"] < 40:
        score -= 10

    # MACD
    if last["MACD_hist"] > 0:
        score += 10
    else:
        score -= 10

    # Volume spike
    if last["Vol_ratio"] > 1.5:
        score += 10

    # 10Y yield macro filter
    yield_10y = get_10y_yield()
    if yield_10y > 4.2:
        score -= 10
    if yield_10y < 4.0:
        score += 10

    # Unusual Whales options bias
    uw = get_uw_options_volume(ticker)
    if uw:
        if float(uw["bullish_premium"]) > float(uw["bearish_premium"]):
            score += 10
        else:
            score -= 10

    score = max(0, min(100, score))

    if score >= 75:
        return score, "BUY CALLS"
    elif score <= 25:
        return score, "BUY PUTS"
    else:
        return score, "WAIT"

# ==============================
# UI
# ==============================
st.title("🏛 Institutional Options Signals (5m) — CALLS / PUTS ONLY")

st_autorefresh(interval=60 * 1000, key="refresh")

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY", "QQQ"])
    st.caption("Institutional mode: Signals only above 75 confidence")

st.subheader("Live Signals")
st.write(f"Last update (UTC): {utc_now()}")

results = []

for ticker in tickers:
    df = get_intraday_data(ticker)
    score, signal = institutional_score(df, ticker)

    results.append({
        "Ticker": ticker,
        "Confidence": score,
        "Signal": signal
    })

df_results = pd.DataFrame(results)
st.dataframe(df_results, use_container_width=True)

st.subheader("Institutional Alerts")

alerts = df_results[df_results["Signal"] != "WAIT"]

if alerts.empty:
    st.info("No institutional signals (confidence < 75).")
else:
    for _, row in alerts.iterrows():
        if "CALL" in row["Signal"]:
            st.success(f"{row['Ticker']} → {row['Signal']} | Confidence: {row['Confidence']}")
        else:
            st.error(f"{row['Ticker']} → {row['Signal']} | Confidence: {row['Confidence']}")

