import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Institutional Options Signals", layout="wide")

CST = ZoneInfo("America/Chicago")

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "NVDA", "AMD"]

UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF"
    "&limit=250&max_dte=3"
    "&min_premium=1000000"
    "&min_volume_oi_ratio=1"
    "&order=premium&order_direction=desc"
)

# ===== Secrets (MUST MATCH THESE NAMES) =====
EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")).strip()
UW_TOKEN = st.secrets.get("UW_TOKEN", os.getenv("UW_TOKEN", "")).strip()

UW_FLOW_ALERTS_URL = "https://api.unusualwhales.com/api/option-trade/flow-alerts"

def now_cst():
    return datetime.now(CST)

def now_cst_str():
    return now_cst().strftime("%Y-%m-%d %H:%M:%S")

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def ok_bad(label, ok: bool, detail: str = ""):
    if ok:
        st.success(f"{label} ✅ {detail}".strip())
    else:
        st.error(f"{label} ❌ {detail}".strip())

# =========================
# EODHD
# =========================
def eodhd_intraday_5m(ticker: str, bars: int = 200):
    url = f"https://eodhd.com/api/intraday/{ticker}.US"
    params = {
        "api_token": EODHD_API_KEY,
        "interval": "5m",
        "fmt": "json",
        "outputsize": bars
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df.set_index("datetime", inplace=True)
    return df

def eodhd_news_safe(ticker: str, lookback_minutes: int = 60, limit: int = 20):
    """
    FIXED: EODHD News API sometimes rejects datetime strings.
    Use DATE-ONLY format (YYYY-MM-DD). This is the most compatible.
    """
    url = "https://eodhd.com/api/news"

    start_date = (now_cst() - timedelta(minutes=int(lookback_minutes))).date().isoformat()
    end_date = now_cst().date().isoformat()

    params = {
        "s": f"{ticker}.US",
        "from": start_date,
        "to": end_date,
        "limit": limit,
        "api_token": EODHD_API_KEY,
        "fmt": "json"
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []

def eodhd_10y_yield_optional():
    """
    OPTIONAL 10Y: Some EODHD plans won't support US10Y.BOND.
    We try, but if it fails we return None (and do NOT break the app).
    """
    try:
        url = "https://eodhd.com/api/real-time/US10Y.BOND"
        params = {"api_token": EODHD_API_KEY, "fmt": "json"}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        val = safe_float(js.get("close"))
        return val if np.isfinite(val) else None
    except:
        return None

# =========================
# UNUSUAL WHALES
# =========================
def uw_headers():
    return {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}"
    }

def uw_options_volume_bias(ticker: str):
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/options-volume"
    r = requests.get(url, headers=uw_headers(), timeout=20)
    r.raise_for_status()
    js = r.json()
    if not js or "data" not in js or not js["data"]:
        return None
    return js["data"][0]

def uw_flow_alerts(limit: int = 200):
    params = {"limit": limit}
    r = requests.get(UW_FLOW_ALERTS_URL, headers=uw_headers(), params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    if isinstance(js, dict) and "data" in js:
        return js["data"]
    if isinstance(js, list):
        return js
    return []

def normalize_flow_alerts(alerts: list):
    rows = []
    for a in alerts:
        ticker = a.get("symbol") or a.get("underlying_symbol") or a.get("ticker") or ""
        premium = safe_float(a.get("premium") or a.get("total_premium") or a.get("notional") or 0)
        option_type = (a.get("option_type") or a.get("type") or "").lower()
        executed_at = a.get("executed_at") or a.get("timestamp") or a.get("created_at") or ""
        expiry = a.get("expiry") or a.get("expiration") or None
        strike = safe_float(a.get("strike") or np.nan)
        underlying_price = safe_float(a.get("underlying_price") or a.get("stock_price") or np.nan)
        tags = a.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]

        rows.append({
            "ticker": ticker,
            "premium": premium,
            "type": option_type,
            "expiry": expiry,
            "strike": strike,
            "underlying_price": underlying_price,
            "executed_at": executed_at,
            "tags": ", ".join(tags[:6]) if tags else ""
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("premium", ascending=False)

# =========================
# INDICATORS
# =========================
def calc_indicators(df: pd.DataFrame):
    df = df.copy()
    df["EMA9"] = df["close"].ewm(span=9).mean()
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()

    df["VWAP"] = (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    df["Vol_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

    tr = (df["high"] - df["low"]).abs()
    df["ATR"] = tr.rolling(14).mean()
    df["IV_spike"] = df["ATR"] > (df["ATR"].rolling(20).mean() * 1.5)

    return df

def score_engine(df: pd.DataFrame, ticker: str, y10: float | None, uw_bias: dict | None):
    last = df.iloc[-1]
    score = 50.0

    # EMA trend
    if last["EMA9"] > last["EMA20"] > last["EMA50"]:
        score += 12
        trend_dir = "bull"
    elif last["EMA9"] < last["EMA20"] < last["EMA50"]:
        score -= 12
        trend_dir = "bear"
    else:
        trend_dir = "neutral"

    # VWAP
    score += 10 if last["close"] > last["VWAP"] else -10

    # RSI
    if last["RSI"] >= 55:
        score += 8
    elif last["RSI"] <= 45:
        score -= 8

    # MACD hist
    score += 8 if last["MACD_hist"] > 0 else -8

    # Volume ratio
    if last["Vol_ratio"] >= 1.5:
        score += 6
    elif last["Vol_ratio"] <= 0.8:
        score -= 3

    # IV spike proxy
    if bool(last["IV_spike"]):
        score += 5

    # 10Y yield (optional)
    if y10 is not None:
        if y10 >= 4.2:
            score -= 6
        elif y10 <= 4.0:
            score += 6

    # UW premium bias
    gamma_bias = "Neutral"
    uw_bias_str = "Neutral"
    if uw_bias:
        bull = safe_float(uw_bias.get("bullish_premium"))
        bear = safe_float(uw_bias.get("bearish_premium"))
        if np.isfinite(bull) and np.isfinite(bear):
            if bull > bear:
                score += 12
                uw_bias_str = "Bullish"
                gamma_bias = "Positive Gamma (proxy)"
            elif bear > bull:
                score -= 12
                uw_bias_str = "Bearish"
                gamma_bias = "Negative Gamma (proxy)"

    score = float(np.clip(score, 0, 100))

    # Calls/Puts only
    if score >= 75:
        signal = "BUY CALLS"
        direction = "CALL"
    elif score <= 25:
        signal = "BUY PUTS"
        direction = "PUT"
    else:
        signal = "WAIT"
        direction = "—"

    return {
        "Ticker": ticker,
        "Confidence": round(score, 1),
        "Direction": direction,
        "Signal": signal,
        "UW Bias": uw_bias_str,
        "Gamma bias": gamma_bias,
        "RSI": round(float(last["RSI"]), 1) if np.isfinite(last["RSI"]) else None,
        "MACD_hist": round(float(last["MACD_hist"]), 4) if np.isfinite(last["MACD_hist"]) else None,
        "VWAP": "Above" if last["close"] > last["VWAP"] else "Below",
        "EMA stack": trend_dir,
        "Vol_ratio": round(float(last["Vol_ratio"]), 2) if np.isfinite(last["Vol_ratio"]) else None,
        "IV spike": bool(last["IV_spike"]),
    }

# =========================
# UI
# =========================
st.title("🏛 Institutional Options Signals (5m) — CALLS / PUTS ONLY")
st.caption(f"Last update (CST): {now_cst_str()}")

refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 10, 300, 30, step=5)
st_autorefresh(interval=refresh_seconds * 1000, key="auto_refresh")

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY", "TSLA"])
    news_lookback = st.number_input("News lookback (minutes)", min_value=5, max_value=240, value=60, step=5)

    st.divider()
    st.subheader("Keys status (green/red)")
    ok_bad("EODHD_API_KEY", bool(EODHD_API_KEY))
    ok_bad("UW_TOKEN (Bearer)", bool(UW_TOKEN), "Fix Secrets name to UW_TOKEN" if not UW_TOKEN else "")

# Endpoint checks
with st.sidebar:
    st.subheader("Endpoints status")

    intraday_ok, intraday_err = False, ""
    if EODHD_API_KEY:
        try:
            _df = eodhd_intraday_5m("SPY", bars=20)
            intraday_ok = _df is not None and len(_df) > 10
        except Exception as e:
            intraday_err = str(e)[:90]
    ok_bad("EODHD intraday", intraday_ok, intraday_err)

    news_ok, news_err = False, ""
    if EODHD_API_KEY:
        try:
            _n = eodhd_news_safe("SPY", lookback_minutes=60, limit=3)
            news_ok = isinstance(_n, list)
        except Exception as e:
            news_err = str(e)[:90]
    ok_bad("EODHD news", news_ok, news_err)

    y10_val = eodhd_10y_yield_optional() if EODHD_API_KEY else None
    ok_bad("10Y yield (optional)", y10_val is not None, f"{y10_val:.2f}" if y10_val is not None else "Not available (ok)")

    uw_vol_ok, uw_vol_err = False, ""
    if UW_TOKEN:
        try:
            _v = uw_options_volume_bias("SPY")
            uw_vol_ok = _v is not None
        except Exception as e:
            uw_vol_err = str(e)[:90]
    ok_bad("UW options-volume", uw_vol_ok, uw_vol_err)

    uw_flow_ok, uw_flow_err = False, ""
    if UW_TOKEN:
        try:
            _a = uw_flow_alerts(limit=5)
            uw_flow_ok = isinstance(_a, list)
        except Exception as e:
            uw_flow_err = str(e)[:90]
    ok_bad("UW flow-alerts", uw_flow_ok, uw_flow_err)

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.components.v1.iframe(UW_SCREENER_URL, height=860, scrolling=True)

with right:
    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW flow)")
    if not tickers:
        st.info("Pick at least 1 ticker in the sidebar.")
        st.stop()

    rows = []
    for t in tickers:
        df = eodhd_intraday_5m(t, bars=200) if EODHD_API_KEY else None
        if df is None or len(df) < 60:
            continue
        df = calc_indicators(df)
        uw_bias = uw_options_volume_bias(t) if UW_TOKEN else None
        rows.append(score_engine(df, t, y10_val, uw_bias))

    score_df = pd.DataFrame(rows)
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.subheader("Alerts (institutional)")
    inst = score_df[score_df["Confidence"] >= 75] if not score_df.empty else score_df
    if inst.empty:
        st.info("No institutional signals (confidence < 75).")
    else:
        for _, r in inst.sort_values("Confidence", ascending=False).iterrows():
            st.success(f"{r['Ticker']}: {r['Signal']} | Conf={r['Confidence']} | UW Bias={r['UW Bias']} | IV_spike={r['IV spike']}")

    st.subheader("Unusual Flow Alerts (UW API)")
    if not UW_TOKEN:
        st.warning("UW_TOKEN missing in Secrets (fix name).")
    else:
        alerts_df = normalize_flow_alerts(uw_flow_alerts(limit=200))
        if alerts_df.empty:
            st.info("No flow alerts returned right now.")
        else:
            alerts_df = alerts_df[alerts_df["premium"] >= 1_000_000]
            alerts_df = alerts_df[alerts_df["ticker"].isin(tickers)]
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)

    st.subheader(f"News — last {int(news_lookback)} minutes (EODHD)")
    if not EODHD_API_KEY:
        st.warning("EODHD_API_KEY missing.")
    else:
        news_rows = []
        for t in tickers:
            items = eodhd_news_safe(t, lookback_minutes=int(news_lookback), limit=20)
            for it in items:
                news_rows.append({
                    "Ticker": t,
                    "published": it.get("date") or it.get("published") or "",
                    "Source": it.get("source") or "",
                    "Title": it.get("title") or "",
                    "URL": it.get("link") or it.get("url") or ""
                })
        if not news_rows:
            st.info("No news returned in this window.")
        else:
            st.dataframe(pd.DataFrame(news_rows), use_container_width=True, hide_index=True)
