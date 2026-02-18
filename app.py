import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

# =============================
# CONFIG
# =============================

st.set_page_config(page_title="Institutional Options Engine", layout="wide")

EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", "")
UW_TOKEN = st.secrets.get("UW_TOKEN", "")

CST = ZoneInfo("America/Chicago")

DEFAULT_TICKERS = ["SPY", "QQQ", "TSLA", "NVDA"]

# =============================
# TIME
# =============================

def now_cst():
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")

# =============================
# DATA FETCHING
# =============================

def get_intraday_data(ticker):
    url = f"https://eodhd.com/api/intraday/{ticker}.US"
    params = {
        "api_token": EODHD_API_KEY,
        "interval": "5m",
        "fmt": "json",
        "outputsize": 200
    }

    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        if not data:
            return None

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df = df.sort_index()

        return df

    except:
        return None

def get_10y_yield():
    url = "https://eodhd.com/api/real-time/US10Y.BOND"
    params = {"api_token": EODHD_API_KEY, "fmt": "json"}

    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return float(r.json()["close"])
    except:
        return None

def get_uw_options_volume(ticker):
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/options-volume"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UW_TOKEN}"
    }

    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()["data"][0]
        return data
    except:
        return None

# =============================
# INDICATORS
# =============================

def calculate_indicators(df):

    # EMA
    df["EMA9"] = df["close"].ewm(span=9).mean()
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()

    # VWAP
    df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Volume Ratio
    df["Vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # IV proxy (volatility spike using ATR % move)
    df["TR"] = df["high"] - df["low"]
    df["ATR"] = df["TR"].rolling(14).mean()
    df["IV_spike"] = df["ATR"] > df["ATR"].rolling(20).mean() * 1.5

    return df

# =============================
# INSTITUTIONAL SCORE ENGINE
# =============================

def institutional_score(df, ticker):

    last = df.iloc[-1]
    score = 50
    direction = None

    # EMA stack trend
    if last["EMA9"] > last["EMA20"] > last["EMA50"]:
        score += 10
        direction = "CALL"
    elif last["EMA9"] < last["EMA20"] < last["EMA50"]:
        score -= 10
        direction = "PUT"

    # VWAP bias
    if last["close"] > last["VWAP"]:
        score += 10
    else:
        score -= 10

    # RSI
    if last["RSI"] > 55:
        score += 8
    elif last["RSI"] < 45:
        score -= 8

    # MACD
    if last["MACD_hist"] > 0:
        score += 8
    else:
        score -= 8

    # Volume Spike
    if last["Vol_ratio"] > 1.5:
        score += 5

    # IV Spike
    if last["IV_spike"]:
        score += 5

    # 10Y Yield filter
    yield_10y = get_10y_yield()
    if yield_10y:
        if yield_10y > 4.2:
            score -= 8
        elif yield_10y < 4.0:
            score += 8

    # UW Options Bias
    uw = get_uw_options_volume(ticker)
    gamma_bias = "Neutral"

    if uw:
        bull = float(uw["bullish_premium"])
        bear = float(uw["bearish_premium"])

        if bull > bear:
            score += 12
            gamma_bias = "Positive Gamma"
        elif bear > bull:
            score -= 12
            gamma_bias = "Negative Gamma"

    score = max(0, min(100, score))

    if score >= 75:
        signal = "BUY CALLS"
    elif score <= 25:
        signal = "BUY PUTS"
    else:
        signal = "WAIT"

    return score, signal, gamma_bias

# =============================
# UI
# =============================

st.title("🏛 Institutional Options Signals (5m) — CALLS / PUTS ONLY")

st.caption(f"Last update (CST): {now_cst()}")

tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY", "QQQ"])

results = []

for ticker in tickers:
    df = get_intraday_data(ticker)
    if df is None:
        continue

    df = calculate_indicators(df)
    score, signal, gamma_bias = institutional_score(df, ticker)
    last = df.iloc[-1]

    results.append({
        "Ticker": ticker,
        "Confidence": score,
        "Signal": signal,
        "RSI": round(last["RSI"], 1),
        "MACD_hist": round(last["MACD_hist"], 4),
        "VWAP_above": last["close"] > last["VWAP"],
        "Vol_ratio": round(last["Vol_ratio"], 2),
        "IV_spike": bool(last["IV_spike"]),
        "Gamma_bias": gamma_bias
    })

df_final = pd.DataFrame(results)

st.dataframe(df_final, use_container_width=True)

st.subheader("Institutional Alerts (≥75 only)")

alerts = df_final[df_final["Confidence"] >= 75]

if alerts.empty:
    st.info("No institutional signals right now.")
else:
    st.dataframe(alerts, use_container_width=True)

