import os
import math
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Institutional Options Signals (5m) — Calls/Puts Only", layout="wide")

CST = ZoneInfo("America/Chicago")

DEFAULT_QUICK = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "NVDA", "AMD"]

UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF"
    "&limit=250&max_dte=3"
    "&min_premium=1000000"
    "&min_volume_oi_ratio=1"
    "&order=premium&order_direction=desc"
)

# Secrets (Streamlit Cloud) or local env
EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")).strip()
UW_TOKEN      = st.secrets.get("UW_TOKEN", os.getenv("UW_TOKEN", "")).strip()

# MUST be dash flow-alerts, not underscore
UW_FLOW_ALERTS_URL = st.secrets.get(
    "UW_FLOW_ALERTS_URL",
    os.getenv("UW_FLOW_ALERTS_URL", "https://api.unusualwhales.com/api/option-trade/flow-alerts")
).strip()

# Force-correct common mistake (underscore)
UW_FLOW_ALERTS_URL = UW_FLOW_ALERTS_URL.replace("flow_alerts", "flow-alerts")


# =========================================================
# SMALL UTILS
# =========================================================
def now_cst() -> datetime:
    return datetime.now(tz=CST)

def fmt_cst(dt: datetime) -> str:
    return dt.astimezone(CST).strftime("%Y-%m-%d %H:%M:%S CST")

def safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def http_get_json(url, params=None, headers=None, timeout=20):
    r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def http_get_text(url, params=None, headers=None, timeout=20):
    r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    return r.text


# =========================================================
# TECH INDICATORS (5m bars)
# =========================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def macd_hist(series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return hist

def vwap(df: pd.DataFrame) -> pd.Series:
    # Typical price VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    return pv.cumsum() / df["volume"].cumsum().replace(0, pd.NA)

def calc_volume_ratio(df: pd.DataFrame, lookback: int = 20):
    # current volume / avg volume
    if df.empty:
        return None
    v = df["volume"]
    if len(v) < 5:
        return None
    avg = v.rolling(lookback).mean().iloc[-1]
    cur = v.iloc[-1]
    if avg is None or pd.isna(avg) or avg == 0:
        return None
    return float(cur / avg)


# =========================================================
# EODHD (intraday + news)
# NOTE: EODHD intraday wants symbol like AAPL.US
# =========================================================
@st.cache_data(ttl=30)
def eodhd_intraday(symbol_us: str, interval="5m", lookback_minutes=240):
    """
    Returns DataFrame with columns: datetime, open, high, low, close, volume
    """
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"

    url = "https://eodhd.com/api/intraday/" + symbol_us
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "interval": interval,
    }

    try:
        data = http_get_json(url, params=params)
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame(), "ticker_data_empty"
        df = pd.DataFrame(data)

        # EODHD uses 'datetime'
        if "datetime" not in df.columns:
            return pd.DataFrame(), "bad_schema"

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(CST)
        df = df.dropna(subset=["datetime"]).sort_values("datetime")

        # Cut to lookback window
        cutoff = now_cst() - timedelta(minutes=int(lookback_minutes))
        df = df[df["datetime"] >= cutoff].copy()

        # Ensure numeric
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"]).copy()

        if df.empty:
            return pd.DataFrame(), "insufficient_bars"

        return df, "ok"

    except requests.HTTPError as e:
        return pd.DataFrame(), f"http_{e.response.status_code}"
    except Exception:
        return pd.DataFrame(), "error"


@st.cache_data(ttl=60)
def eodhd_news(symbol_us: str, lookback_minutes=60, limit=30):
    """
    EODHD News endpoint: https://eodhd.com/financial-apis/stock-market-news-api/
    We'll use: /api/news?s=AAPL.US&offset=0&limit=...
    """
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"

    url = "https://eodhd.com/api/news"
    params = {
        "api_token": EODHD_API_KEY,
        "s": symbol_us,
        "limit": int(limit),
        "offset": 0,
        "fmt": "json",
    }

    try:
        data = http_get_json(url, params=params)
        if not isinstance(data, list):
            return pd.DataFrame(), "bad_schema"

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(), "no_headlines"

        # try to normalize fields
        # common fields: date, title, source, link
        if "date" in df.columns:
            df["published"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(CST)
        elif "published_at" in df.columns:
            df["published"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.tz_convert(CST)
        else:
            df["published"] = pd.NaT

        cutoff = now_cst() - timedelta(minutes=int(lookback_minutes))
        df = df[df["published"].isna() | (df["published"] >= cutoff)].copy()

        # columns
        df["Title"] = df.get("title", "")
        df["Source"] = df.get("source", "")
        df["URL"] = df.get("link", df.get("url", ""))

        df = df[["published", "Source", "Title", "URL"]].copy()
        df = df.sort_values("published", ascending=False)

        return df, "ok"
    except requests.HTTPError as e:
        return pd.DataFrame(), f"http_{e.response.status_code}"
    except Exception:
        return pd.DataFrame(), "error"


def naive_news_sentiment(title: str) -> float:
    """
    Cheap keyword sentiment: returns -1..+1
    (You can replace later with a real model.)
    """
    if not isinstance(title, str):
        return 0.0
    t = title.lower()
    pos = ["beats", "beat", "surge", "rally", "up", "growth", "record", "strong", "bull", "upgrade", "raises"]
    neg = ["miss", "misses", "drop", "down", "lawsuit", "fraud", "weak", "bear", "downgrade", "cuts", "warn", "loss"]
    score = 0
    for w in pos:
        if w in t:
            score += 1
    for w in neg:
        if w in t:
            score -= 1
    return clamp(score / 5.0, -1.0, 1.0)


# =========================================================
# 10Y Yield (optional)
# We'll attempt: EODHD intraday "US10Y.INDX" (common on some feeds)
# If it fails, we mark N/A (and we DO NOT break your app).
# =========================================================
@st.cache_data(ttl=120)
def fetch_10y_yield_proxy():
    if not EODHD_API_KEY:
        return None, "missing_key"
    symbol = "US10Y.INDX"
    df, status = eodhd_intraday(symbol, interval="5m", lookback_minutes=1440)
    if status != "ok" or df.empty:
        return None, f"unavailable_{status}"
    val = safe_float(df["close"].iloc[-1], None)
    return val, "ok"


# =========================================================
# UNUSUAL WHALES (Options Volume Bias + Flow Alerts)
# =========================================================
@st.cache_data(ttl=60)
def uw_options_volume_bias(ticker: str):
    """
    Uses endpoint you posted:
    GET https://api.unusualwhales.com/api/stock/{ticker}/options-volume
    """
    if not UW_TOKEN:
        return None, "missing_token"

    url = f"https://api.unusualwhales.com/api/stock/{ticker}/options-volume"
    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}",
    }
    try:
        data = http_get_json(url, headers=headers)
        rows = data.get("data", [])
        if not rows:
            return None, "no_data"
        row = rows[0]

        bullish_prem = safe_float(row.get("bullish_premium"), 0.0)
        bearish_prem = safe_float(row.get("bearish_premium"), 0.0)
        call_vol = safe_float(row.get("call_volume"), None)
        put_vol = safe_float(row.get("put_volume"), None)

        # Bias score -1..+1
        prem_total = max(bullish_prem + bearish_prem, 1.0)
        prem_bias = (bullish_prem - bearish_prem) / prem_total  # -1..+1 approx

        # Put/Call vol ratio
        pc = None
        if call_vol and call_vol > 0 and put_vol is not None:
            pc = float(put_vol / call_vol)

        return {
            "prem_bias": float(prem_bias),
            "put_call_vol": pc,
            "bullish_premium": bullish_prem,
            "bearish_premium": bearish_prem,
        }, "ok"

    except requests.HTTPError as e:
        return None, f"http_{e.response.status_code}"
    except Exception:
        return None, "error"


@st.cache_data(ttl=30)
def uw_flow_alerts(limit: int = 300):
    """
    Correct endpoint (dash):
    GET https://api.unusualwhales.com/api/option-trade/flow-alerts

    If this is still 404, it means:
      - your plan doesn't include this endpoint OR
      - UW changed the route OR
      - token scope mismatch

    We show status clearly and keep app running.
    """
    if not UW_TOKEN:
        return [], "missing_token"

    url = UW_FLOW_ALERTS_URL
    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}",
    }

    # We keep params minimal to avoid “unknown param” failures.
    params = {
        "limit": int(limit),
    }

    try:
        data = http_get_json(url, params=params, headers=headers)
        rows = data.get("data", data.get("results", data if isinstance(data, list) else []))
        if not isinstance(rows, list):
            return [], "bad_schema"
        return rows, "ok"

    except requests.HTTPError as e:
        return [], f"http_{e.response.status_code}"
    except Exception:
        return [], "error"


def filter_flow_alerts_client_side(rows, *, min_premium=1_000_000, max_dte=3, exclude_itm=True, vol_gt_oi=True):
    """
    You asked for:
      - premium >= $1M
      - DTE <= 3
      - exclude ITM
      - volume > OI
      - stocks + ETFs only (best done in screener; API schema varies)
    We'll apply what we can safely from returned fields.
    """
    out = []
    for r in rows:
        prem = safe_float(r.get("premium"), None)
        if prem is not None and prem < min_premium:
            continue

        # DTE: try direct, else compute from expiry
        dte = safe_float(r.get("dte"), None)
        if dte is None:
            exp = r.get("expiry")
            try:
                exp_dt = pd.to_datetime(exp).to_pydatetime()
                dte = (exp_dt.date() - now_cst().date()).days
            except Exception:
                dte = None
        if dte is not None and dte > max_dte:
            continue

        if exclude_itm:
            itm = r.get("itm")
            # some schemas: 'is_itm' or 'itm'
            if itm is None:
                itm = r.get("is_itm")
            if isinstance(itm, bool) and itm:
                continue

        if vol_gt_oi:
            vol = safe_float(r.get("volume"), None)
            oi = safe_float(r.get("open_interest"), None)
            if vol is not None and oi is not None and not (vol > oi):
                continue

        out.append(r)
    return out


def flow_features_for_ticker(rows, ticker: str, lookback_minutes: int = 60):
    """
    Extract:
      - unusual flag (any alerts)
      - call/put bias from option_type tags
      - IV spike proxy
      - Gamma bias proxy
    """
    cutoff = now_cst() - timedelta(minutes=int(lookback_minutes))

    f = []
    for r in rows:
        sym = (r.get("underlying_symbol") or r.get("ticker") or "").upper()
        if sym != ticker.upper():
            continue

        # executed_at
        t = r.get("executed_at") or r.get("created_at") or r.get("timestamp")
        dt = None
        try:
            dt = pd.to_datetime(t, utc=True).tz_convert(CST)
        except Exception:
            dt = None

        if dt is not None and dt < cutoff:
            continue

        f.append((dt, r))

    if not f:
        return {
            "unusual": False,
            "flow_bias": 0.0,       # -1 puts, +1 calls
            "iv_spike": False,
            "iv_now": None,
            "gamma_bias": "N/A",
            "gamma_proxy": None,
            "recent_count": 0,
        }

    # sort newest first
    f.sort(key=lambda x: (x[0] is not None, x[0]), reverse=True)
    recent = [r for _, r in f[:200]]

    # call/put bias
    calls = 0
    puts = 0
    for r in recent:
        ot = (r.get("option_type") or r.get("type") or "").lower()
        if "call" in ot:
            calls += 1
        elif "put" in ot:
            puts += 1

    denom = max(calls + puts, 1)
    flow_bias = (calls - puts) / denom  # -1..+1

    # IV spike proxy
    ivs = [safe_float(r.get("implied_volatility"), None) for r in recent]
    ivs = [x for x in ivs if x is not None and x > 0]
    iv_spike = False
    iv_now = None
    if len(ivs) >= 10:
        iv_now = float(pd.Series(ivs[:10]).mean())
        iv_base = float(pd.Series(ivs).mean())
        if iv_base > 0 and iv_now > iv_base * 1.25 and iv_now > 0.35:
            iv_spike = True

    # Gamma bias proxy (net gamma from recent trades)
    # (Not true market-wide GEX, but useful directional hint)
    gsum = 0.0
    gcount = 0
    for r in recent:
        g = safe_float(r.get("gamma"), None)
        if g is None:
            continue
        ot = (r.get("option_type") or "").lower()
        sign = 1.0 if "call" in ot else (-1.0 if "put" in ot else 0.0)
        if sign == 0.0:
            continue
        size = safe_float(r.get("size"), 1.0) or 1.0
        gsum += sign * g * size
        gcount += 1

    gamma_proxy = gsum if gcount > 0 else None
    if gamma_proxy is None:
        gamma_bias = "N/A"
    else:
        gamma_bias = "Positive Gamma (proxy)" if gamma_proxy > 0 else "Negative Gamma (proxy)"

    return {
        "unusual": True,
        "flow_bias": float(flow_bias),
        "iv_spike": bool(iv_spike),
        "iv_now": iv_now,
        "gamma_bias": gamma_bias,
        "gamma_proxy": gamma_proxy,
        "recent_count": len(recent),
    }


# =========================================================
# SCORING (Calls/Puts only)
# =========================================================
def score_ticker(df_5m: pd.DataFrame, *, uw_vol=None, flow_feat=None, news_df=None, ten_y=None, weights=None):
    """
    Returns:
      confidence (0..100),
      direction: "BULLISH"/"BEARISH"/"—",
      signal: "BUY CALLS"/"BUY PUTS"/"WAIT",
      plus feature columns.
    """
    if df_5m is None or df_5m.empty or len(df_5m) < 30:
        return {
            "confidence": 50,
            "direction": "—",
            "signal": "WAIT",
            "rsi": None,
            "macd_hist": None,
            "vwap_above": None,
            "ema_stack": None,
            "vol_ratio": None,
            "uw_bias": None,
            "uw_unusual": "NO",
            "iv_spike": None,
            "gamma_bias": "N/A",
            "ten_y": ten_y if ten_y is not None else None,
        }

    close = df_5m["close"]
    df = df_5m.copy()

    df["EMA9"] = ema(close, 9)
    df["EMA20"] = ema(close, 20)
    df["EMA50"] = ema(close, 50)
    df["RSI"] = rsi(close, 14)
    df["MACD_H"] = macd_hist(close)
    df["VWAP"] = vwap(df)

    last = df.iloc[-1]

    rsi_v = float(last["RSI"])
    macd_h = float(last["MACD_H"])
    vwap_above = bool(last["close"] > last["VWAP"])
    ema_stack = bool(last["EMA9"] > last["EMA20"] > last["EMA50"]) or bool(last["EMA9"] < last["EMA20"] < last["EMA50"])
    ema_bull = bool(last["EMA9"] > last["EMA20"] > last["EMA50"])
    ema_bear = bool(last["EMA9"] < last["EMA20"] < last["EMA50"])

    vol_ratio = calc_volume_ratio(df, 20)

    # News sentiment (simple)
    ns = 0.0
    if news_df is not None and not news_df.empty:
        top = news_df.head(10)
        scores = [naive_news_sentiment(t) for t in top["Title"].tolist()]
        if scores:
            ns = float(pd.Series(scores).mean())

    # UW options volume bias
    uw_bias = 0.0
    put_call_vol = None
    if uw_vol and isinstance(uw_vol, dict):
        uw_bias = float(uw_vol.get("prem_bias", 0.0))
        put_call_vol = uw_vol.get("put_call_vol", None)

    # Flow features
    flow_bias = 0.0
    unusual = "NO"
    iv_spike = None
    gamma_bias = "N/A"
    if flow_feat and isinstance(flow_feat, dict):
        unusual = "YES" if flow_feat.get("unusual") else "NO"
        flow_bias = float(flow_feat.get("flow_bias", 0.0))
        iv_spike = flow_feat.get("iv_spike", None)
        gamma_bias = flow_feat.get("gamma_bias", "N/A")

    # 10Y filter (optional)
    # If yield is rising hard, it can pressure high-beta. We only nudge score, never hard-block.
    ten_y_nudge = 0.0
    if ten_y is not None:
        # Just a mild penalty if yield is above a typical intraday “risk-off” zone.
        # (You can tune this later.)
        if ten_y > 4.8:
            ten_y_nudge = -0.10
        elif ten_y < 4.0:
            ten_y_nudge = +0.05

    # Weights (sum doesn't need to be 1)
    w = weights or {
        "vwap": 0.18,
        "ema": 0.18,
        "rsi": 0.15,
        "macd": 0.15,
        "vol": 0.12,
        "uw_vol": 0.20,
        "flow": 0.20,
        "news": 0.05,
        "teny": 0.05,
    }

    # Convert indicators to -1..+1 components
    # RSI: 50 neutral; above bullish; below bearish
    rsi_comp = clamp((rsi_v - 50.0) / 25.0, -1.0, 1.0)

    # MACD histogram: scale by recent std
    mh = df["MACD_H"].dropna()
    mh_std = float(mh.tail(100).std()) if len(mh) > 20 else 0.0
    macd_comp = 0.0 if mh_std == 0 else clamp(macd_h / (2.0 * mh_std), -1.0, 1.0)

    # VWAP component
    vwap_comp = 1.0 if vwap_above else -1.0

    # EMA stack component
    if ema_bull:
        ema_comp = 1.0
    elif ema_bear:
        ema_comp = -1.0
    else:
        ema_comp = 0.0

    # Volume ratio component
    vol_comp = 0.0
    if vol_ratio is not None:
        # >1 = more momentum; cap
        vol_comp = clamp((vol_ratio - 1.0) / 1.5, -1.0, 1.0)

    # UW vol bias component already -1..+1 (premium bias)
    uw_comp = clamp(uw_bias, -1.0, 1.0)

    # Flow bias component -1..+1 (calls vs puts count)
    flow_comp = clamp(flow_bias, -1.0, 1.0)

    # News comp -1..+1
    news_comp = clamp(ns, -1.0, 1.0)

    # 10Y nudge already small
    teny_comp = clamp(ten_y_nudge, -1.0, 1.0)

    raw = (
        w["vwap"] * vwap_comp +
        w["ema"]  * ema_comp +
        w["rsi"]  * rsi_comp +
        w["macd"] * macd_comp +
        w["vol"]  * vol_comp +
        w["uw_vol"] * uw_comp +
        w["flow"] * flow_comp +
        w["news"] * news_comp +
        w["teny"] * teny_comp
    )

    # Convert raw (-sum..+sum) to confidence 0..100
    max_abs = sum(abs(x) for x in w.values())
    norm = 0.0 if max_abs == 0 else clamp(raw / max_abs, -1.0, 1.0)

    # Confidence is strength (absolute), direction is sign
    confidence = int(round(50 + 50 * abs(norm)))
    direction = "BULLISH" if norm > 0.05 else ("BEARISH" if norm < -0.05 else "—")

    # Calls/Puts ONLY decision
    if direction == "BULLISH" and confidence >= 60:
        signal = "BUY CALLS"
    elif direction == "BEARISH" and confidence >= 60:
        signal = "BUY PUTS"
    else:
        signal = "WAIT"

    return {
        "confidence": confidence,
        "direction": direction,
        "signal": signal,
        "rsi": round(rsi_v, 1),
        "macd_hist": round(macd_h, 4),
        "vwap_above": "Above" if vwap_above else "Below",
        "ema_stack": "Bull" if ema_bull else ("Bear" if ema_bear else "Neutral"),
        "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        "uw_bias": round(uw_comp, 2),
        "uw_unusual": unusual,
        "iv_spike": "YES" if iv_spike else "NO",
        "gamma_bias": gamma_bias,
        "ten_y": ten_y,
        "put_call_vol": put_call_vol,
    }


# =========================================================
# SIDEBAR UI (FREE TICKER INPUT)
# =========================================================
with st.sidebar:
    st.header("Settings")

    st.caption("Type any tickers (comma-separated). Example: SPY,TSLA,NVDA")
    raw = st.text_input("Tickers", value="SPY,TSLA")

    # optional quick picker
    quick = st.multiselect("Quick pick (optional)", DEFAULT_QUICK, default=[])

    # merge + clean
    typed = [t.strip().upper() for t in raw.split(",") if t.strip()]
    tickers = []
    for t in typed + [q.upper() for q in quick]:
        if t and t not in tickers:
            tickers.append(t)

    st.divider()

    news_lookback = st.number_input("News lookback (minutes)", min_value=1, max_value=240, value=60, step=1)
    price_lookback = st.number_input("Price lookback (minutes)", min_value=60, max_value=1440, value=240, step=30)

    st.divider()
    st.subheader("Refresh")
    refresh_seconds = st.slider("Auto-refresh (seconds)", min_value=10, max_value=300, value=30, step=5)

    st.divider()
    st.subheader("Institutional mode")
    inst_min = st.slider("Signals only if confidence ≥", min_value=50, max_value=95, value=75, step=1)

    st.divider()
    st.subheader("Weights (sum doesn't have to be 1)")
    w_vwap = st.slider("VWAP weight", 0.0, 0.40, 0.18, 0.01)
    w_ema  = st.slider("EMA stack (9/20/50) weight", 0.0, 0.40, 0.18, 0.01)
    w_rsi  = st.slider("RSI weight", 0.0, 0.40, 0.15, 0.01)
    w_macd = st.slider("MACD hist weight", 0.0, 0.40, 0.15, 0.01)
    w_vol  = st.slider("Volume ratio weight", 0.0, 0.40, 0.12, 0.01)
    w_uw   = st.slider("UW options volume weight", 0.0, 0.60, 0.20, 0.01)
    w_flow = st.slider("UW flow alerts weight", 0.0, 0.60, 0.20, 0.01)
    w_news = st.slider("News sentiment weight", 0.0, 0.20, 0.05, 0.01)
    w_teny = st.slider("10Y yield (optional) weight", 0.0, 0.20, 0.05, 0.01)

    weights = {
        "vwap": w_vwap, "ema": w_ema, "rsi": w_rsi, "macd": w_macd,
        "vol": w_vol, "uw_vol": w_uw, "flow": w_flow, "news": w_news, "teny": w_teny,
    }

    st.divider()
    st.subheader("Keys status (green/red)")
    def pill(label, ok: bool, detail=""):
        if ok:
            st.success(f"{label} ✅")
        else:
            st.error(f"{label} ❌ {detail}".strip())

    pill("EODHD_API_KEY", bool(EODHD_API_KEY), "(missing)")
    pill("UW_TOKEN (Bearer)", bool(UW_TOKEN), "(missing)")
    pill("UW_FLOW_ALERTS_URL", bool(UW_FLOW_ALERTS_URL), "(missing)")

    st.caption("Tip: If UW flow-alerts still shows 404, it’s almost always plan/endpoint access — not your code.")


# =========================================================
# AUTOREFRESH
# =========================================================
if HAS_AUTOREFRESH:
    st_autorefresh(interval=int(refresh_seconds * 1000), key="autorefresh")
else:
    # simple manual hint
    st.caption("Auto-refresh module not installed; refresh the page to update.")


# =========================================================
# MAIN UI
# =========================================================
st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")
st.write(f"Last update (CST): **{fmt_cst(now_cst())}**")

if not tickers:
    st.warning("Type at least one ticker in the sidebar (example: SPY,TSLA).")
    st.stop()

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("This is embedded. True filtering (DTE/ITM/premium rules) is best done inside the screener itself.")
    st.components.v1.iframe(UW_SCREENER_URL, height=850, scrolling=True)

with right:
    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW flow)")

    # 10Y yield (optional)
    ten_y_val, ten_y_status = fetch_10y_yield_proxy()
    ten_y_display = ten_y_val

    # UW flow alerts (global pull; then filter per ticker)
    flow_rows, flow_status = uw_flow_alerts(limit=400)
    flow_rows = filter_flow_alerts_client_side(
        flow_rows,
        min_premium=1_000_000,
        max_dte=3,
        exclude_itm=True,
        vol_gt_oi=True
    )

    # Show endpoint status block
    st.markdown("### Endpoints status")
    c1, c2, c3 = st.columns(3)

    with c1:
        ok = True if ten_y_status == "ok" else False
        st.success("10Y yield ✅" if ok else "10Y yield (optional) ⚠️")
        if not ok:
            st.caption("Not available (ok).")

    with c2:
        if flow_status == "ok":
            st.success("UW flow-alerts ✅")
        else:
            st.error(f"UW flow-alerts ❌ ({flow_status})")
            st.caption("If you see http_404 here: endpoint access / plan / route. Code is fine.")

    with c3:
        if EODHD_API_KEY:
            st.success("EODHD ✅")
        else:
            st.error("EODHD ❌ (missing key)")

    # Build table rows
    rows = []
    news_frames = []

    for t in tickers:
        symbol_us = f"{t}.US"

        # price bars
        bars, bars_status = eodhd_intraday(symbol_us, interval="5m", lookback_minutes=int(price_lookback))

        # news
        ndf, news_status = eodhd_news(symbol_us, lookback_minutes=int(news_lookback), limit=50)
        if news_status == "ok" and not ndf.empty:
            ndf2 = ndf.copy()
            ndf2.insert(0, "Ticker", t)
            news_frames.append(ndf2)

        # uw options volume
        uwv, uwv_status = uw_options_volume_bias(t)

        # flow features
        flow_feat = flow_features_for_ticker(flow_rows, t, lookback_minutes=int(news_lookback))

        # score
        res = score_ticker(
            bars,
            uw_vol=uwv if uwv_status == "ok" else None,
            flow_feat=flow_feat,
            news_df=ndf if news_status == "ok" else None,
            ten_y=ten_y_display if ten_y_status == "ok" else None,
            weights=weights
        )

        rows.append({
            "Ticker": t,
            "Confidence": res["confidence"],
            "Direction": res["direction"],
            "Signal": res["signal"],
            "UW Unusual": res["uw_unusual"],
            "UW Bias": res["uw_bias"],
            "Gamma bias": res["gamma_bias"],
            "IV spike": res["iv_spike"],
            "RSI": res["rsi"],
            "MACD_hist": res["macd_hist"],
            "VWAP": res["vwap_above"],
            "EMA stack": res["ema_stack"],
            "Vol_ratio": res["vol_ratio"],
            "Put/Call vol": res.get("put_call_vol", None),
            "10Y": res["ten_y"] if res["ten_y"] is not None else "N/A",
            "EODHD bars": bars_status,
            "EODHD news": news_status,
            "UW opt-vol": uwv_status,
            "UW flow": flow_status,
        })

    df_out = pd.DataFrame(rows)

    # show main table
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    # institutional alerts only
    st.markdown("### Institutional Alerts (≥75 only)")
    inst = df_out[df_out["Confidence"] >= int(inst_min)].copy()
    inst = inst[inst["Signal"].isin(["BUY CALLS", "BUY PUTS"])]
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        for _, r in inst.sort_values("Confidence", ascending=False).iterrows():
            st.success(f"{r['Ticker']}: {r['Signal']} | {r['Direction']} | Confidence={r['Confidence']} | UW Unusual={r['UW Unusual']} | IV spike={r['IV spike']}")

    # UW Flow Alerts box (just status + counts)
    st.markdown("### Unusual Flow Alerts (UW API)")
    st.caption("Rules applied: premium ≥ $1M, DTE ≤ 3, Volume > OI, exclude ITM. (Best effort based on fields available.)")

    if flow_status != "ok":
        st.error(f"UW flow fetch failed: {UW_FLOW_ALERTS_URL} → {flow_status}")
        st.caption("If it’s http_404: endpoint access/plan or route changed. Verify in UW docs for your plan.")
    else:
        st.success(f"UW flow OK — {len(flow_rows)} alerts matched your filters in the fetched batch.")
        # show last few alerts (light normalization)
        simple = []
        for r in flow_rows[:30]:
            simple.append({
                "executed_at": r.get("executed_at"),
                "ticker": r.get("underlying_symbol") or r.get("ticker"),
                "type": r.get("option_type"),
                "expiry": r.get("expiry"),
                "strike": r.get("strike"),
                "premium": r.get("premium"),
                "volume": r.get("volume"),
                "open_interest": r.get("open_interest"),
                "iv": r.get("implied_volatility"),
            })
        st.dataframe(pd.DataFrame(simple), use_container_width=True, hide_index=True)

    # News table
    st.markdown(f"### News — last {int(news_lookback)} minutes (EODHD)")
    if news_frames:
        news_all = pd.concat(news_frames, ignore_index=True)
        # show CST timestamps only
        news_all["published_cst"] = news_all["published"].dt.strftime("%Y-%m-%d %H:%M:%S CST")
        show = news_all[["Ticker", "published_cst", "Source", "Title", "URL"]].copy()
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("Tip: Click URL column links (or copy/paste).")
    else:
        st.info("No news in this lookback window (or EODHD returned none).")

    # Debug helper
    with st.expander("Debug (why something might show N/A)"):
        st.write("- Indicators need enough 5m bars (≈30+) from EODHD intraday.")
        st.write("- If EODHD intraday says `ticker_data_empty`, try another ticker or confirm it exists as `TICKER.US` on EODHD.")
        st.write("- If UW flow shows `http_404`, it’s an endpoint/plan access issue (not your code).")
        st.write("- 10Y uses `US10Y.INDX` as a proxy; if your feed doesn’t include it, it will show N/A.")
