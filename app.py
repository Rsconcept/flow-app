import os
import math
import time
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st

from datetime import datetime, timedelta, date, timezone
try:
    from zoneinfo import ZoneInfo
    CST = ZoneInfo("America/Chicago")
except Exception:
    CST = None  # fallback handled below


# =========================
# App config
# =========================
st.set_page_config(page_title="Institutional Options Signals", layout="wide")
st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")

# =========================
# Helpers
# =========================
def now_cst():
    if CST:
        return datetime.now(CST)
    # fallback: approximate CST as UTC-6 (ignores DST)
    return datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=-6)

def to_cst_str(dt: datetime) -> str:
    if not isinstance(dt, datetime):
        return "N/A"
    try:
        if CST:
            return dt.astimezone(CST).strftime("%Y-%m-%d %H:%M:%S CST")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "N/A"

def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("none", "nan", "null"):
            return None
        return float(s)
    except Exception:
        return None

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def http_get(url, headers=None, params=None, timeout=20):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        return r
    except Exception as e:
        return e

def try_json(resp):
    """
    Returns (ok_bool, data_or_text, error)
    Handles HTML/text errors gracefully.
    """
    if isinstance(resp, Exception):
        return False, None, str(resp)

    ct = (resp.headers.get("Content-Type") or "").lower()
    txt = resp.text or ""

    if resp.status_code >= 400:
        # Return raw text to show user
        return False, txt[:500], f"http_{resp.status_code}"

    # Sometimes APIs return text/html even with 200
    if "application/json" in ct or txt.strip().startswith("{") or txt.strip().startswith("["):
        try:
            return True, resp.json(), None
        except Exception as e:
            return False, txt[:500], f"parse_error: {e}"

    return False, txt[:500], "parse_error: non-json response"

def badge(label, status="ok", detail=None):
    """
    status: ok / warn / err
    """
    if status == "ok":
        st.success(f"{label}" + (f" — {detail}" if detail else ""))
    elif status == "warn":
        st.warning(f"{label}" + (f" — {detail}" if detail else ""))
    else:
        st.error(f"{label}" + (f" — {detail}" if detail else ""))

def normalize_iv_to_percent(iv_raw):
    """
    EODHD options endpoint may return decimal (0.31) or percent (31) depending on field/source.
    We normalize to percent 0..500 range.
    """
    iv = safe_float(iv_raw)
    if iv is None:
        return None

    # If it looks like a normal decimal IV
    if 0 < iv <= 5:
        return iv * 100.0

    # If it looks like it's already percent
    if 5 < iv <= 500:
        return iv

    # If it’s crazy huge (like 3719), likely basis points or multiplied twice
    # Convert bps -> percent
    if iv > 500:
        return iv / 100.0

    return iv

def compute_rsi(close, period=14):
    close = pd.Series(close).dropna()
    if len(close) < period + 2:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None

def compute_ema(series, span):
    s = pd.Series(series).dropna()
    if len(s) < span + 2:
        return None
    return float(s.ewm(span=span, adjust=False).mean().iloc[-1])

def compute_macd_hist(close, fast=12, slow=26, signal=9):
    c = pd.Series(close).dropna()
    if len(c) < slow + signal + 5:
        return None
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return float(hist.iloc[-1])

def compute_vwap(df):
    # needs close + volume
    if df is None or df.empty:
        return None
    if not {"close", "volume"}.issubset(df.columns):
        return None
    vol = df["volume"].astype(float)
    if vol.sum() <= 0:
        return None
    vwap = (df["close"].astype(float) * vol).sum() / vol.sum()
    return float(vwap)

def vol_ratio(df, lookback=30):
    if df is None or df.empty or "volume" not in df.columns:
        return None
    v = df["volume"].astype(float)
    if len(v) < lookback + 2:
        return None
    last = v.iloc[-1]
    avg = v.iloc[-lookback-1:-1].mean()
    if avg <= 0:
        return None
    return float(last / avg)

def parse_datetime_any(x):
    # handles "2026-02-18 15:12:52" or ISO
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt.astimezone(CST) if CST else dt
        except Exception:
            pass
    try:
        dt = pd.to_datetime(s, utc=True)
        if pd.isna(dt):
            return None
        dt = dt.to_pydatetime()
        return dt.astimezone(CST) if (CST and dt.tzinfo) else dt
    except Exception:
        return None


# =========================
# Secrets
# =========================
UW_TOKEN = st.secrets.get("UW_TOKEN", "").strip()
UW_FLOW_ALERTS_URL = st.secrets.get("UW_FLOW_ALERTS_URL", "").strip()
EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", "").strip()
FINVIZ_AUTH = st.secrets.get("FINVIZ_AUTH", "").strip()
FRED_API_KEY = st.secrets.get("FRED_API_KEY", "").strip()

# Defaults if user left URL blank
if not UW_FLOW_ALERTS_URL:
    UW_FLOW_ALERTS_URL = "https://api.unusualwhales.com/api/option-trades/flow-alerts"

# =========================
# Sidebar UI
# =========================
st.sidebar.header("Settings")

tickers_raw = st.sidebar.text_input(
    "Type any tickers (comma-separated).",
    value="SPY,TSLA,NVDA",
    help="Example: SPY,TSLA,NVDA or AAPL,MSFT,QQQ"
)

quick = st.sidebar.multiselect(
    "Quick pick (optional)",
    options=["SPY", "QQQ", "IWM", "DIA", "TSLA", "NVDA", "AMD", "AAPL", "MSFT", "META"],
    default=[]
)

# Merge + normalize
tickers = []
for part in (tickers_raw or "").split(","):
    t = part.strip().upper()
    if t:
        tickers.append(t)
tickers = list(dict.fromkeys(tickers + [q.upper() for q in quick]))

news_lookback_min = st.sidebar.number_input("News lookback (minutes)", min_value=5, max_value=240, value=60, step=5)
price_lookback_min = st.sidebar.number_input("Price lookback (minutes)", min_value=60, max_value=2000, value=240, step=30)
refresh_sec = st.sidebar.slider("Auto-refresh (seconds)", min_value=10, max_value=120, value=30, step=5)

st.sidebar.divider()
st.sidebar.subheader("Institutional mode")
inst_threshold = st.sidebar.slider("Signals only if confidence ≥", min_value=50, max_value=95, value=75, step=1)

st.sidebar.divider()
st.sidebar.subheader("Weights (sum doesn't have to be 1)")
w_rsi = st.sidebar.slider("RSI weight", 0.0, 0.3, 0.15, 0.01)
w_macd = st.sidebar.slider("MACD weight", 0.0, 0.3, 0.15, 0.01)
w_vwap = st.sidebar.slider("VWAP weight", 0.0, 0.3, 0.15, 0.01)
w_ema = st.sidebar.slider("EMA stack (9/20/50) weight", 0.0, 0.3, 0.18, 0.01)
w_volr = st.sidebar.slider("Volume ratio weight", 0.0, 0.3, 0.12, 0.01)
w_uw = st.sidebar.slider("UW flow weight", 0.0, 0.4, 0.20, 0.01)
w_news = st.sidebar.slider("News weight (placeholder)", 0.0, 0.2, 0.05, 0.01)
w_10y = st.sidebar.slider("10Y yield (optional) weight", 0.0, 0.2, 0.05, 0.01)


# =========================
# Status panel (keys)
# =========================
st.sidebar.divider()
st.sidebar.subheader("Keys status (green/red)")

def key_present(name, val):
    if val:
        st.sidebar.success(name)
    else:
        st.sidebar.error(f"{name} (missing)")

key_present("UW_TOKEN", UW_TOKEN)
key_present("EODHD_API_KEY", EODHD_API_KEY)
key_present("FRED_API_KEY (10Y live)", FRED_API_KEY)
if FINVIZ_AUTH:
    st.sidebar.info("FINVIZ_AUTH present (not used in this build)")
else:
    st.sidebar.info("FINVIZ_AUTH not set (ok)")

# =========================
# Data fetchers
# =========================
@st.cache_data(ttl=20, show_spinner=False)
def eodhd_intraday_bars(ticker: str, minutes_back: int):
    """
    EODHD intraday requires UNIX seconds for from/to:
    /intraday/{ticker}.US?interval=5m&from=...&to=...&api_token=...&fmt=json
    """
    if not EODHD_API_KEY:
        return None, "missing_key", "EODHD_API_KEY missing"

    end_dt = now_cst()
    start_dt = end_dt - timedelta(minutes=int(minutes_back))

    # EODHD expects UTC timestamps seconds
    # Convert CST dt -> UTC epoch
    end_utc = end_dt.astimezone(timezone.utc) if end_dt.tzinfo else end_dt.replace(tzinfo=timezone.utc)
    start_utc = start_dt.astimezone(timezone.utc) if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)

    to_ts = int(end_utc.timestamp())
    from_ts = int(start_utc.timestamp())

    url = f"https://eodhd.com/api/intraday/{ticker}.US"
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "interval": "5m",
        "from": from_ts,
        "to": to_ts
    }
    r = http_get(url, params=params)
    ok, data, err = try_json(r)
    if not ok:
        return None, err or "error", str(data)

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(), "ticker_data_empty", "No intraday bars returned for this window."

    df = pd.DataFrame(data)
    # Typical columns: datetime, open, high, low, close, volume
    # Normalize
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    return df, "ok", "ok"

@st.cache_data(ttl=60, show_spinner=False)
def eodhd_news(ticker: str, minutes_back: int):
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key", "EODHD_API_KEY missing"

    url = "https://eodhd.com/api/news"
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "s": f"{ticker}.US",
        "limit": 50,
        "offset": 0
    }
    r = http_get(url, params=params)
    ok, data, err = try_json(r)
    if not ok:
        # If EODHD returns HTML/text, don't crash
        return pd.DataFrame(), err or "error", str(data)

    if not isinstance(data, list):
        return pd.DataFrame(), "parse_error", "News response is not a list"

    df = pd.DataFrame(data)
    if df.empty:
        return df, "ok", "no headlines"

    # Normalize column names (EODHD uses "date" or "publishedAt" depending)
    if "date" in df.columns:
        df["published"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    elif "publishedAt" in df.columns:
        df["published"] = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")
    else:
        df["published"] = pd.NaT

    cutoff = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(minutes=int(minutes_back))
    df = df[df["published"].notna() & (df["published"] >= cutoff)].copy()

    # Keep safe columns
    out = pd.DataFrame()
    out["ticker"] = ticker
    out["published_cst"] = df["published"].dt.tz_convert(CST).dt.strftime("%Y-%m-%d %H:%M:%S CST") if CST else df["published"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["source"] = df.get("source", "")
    out["title"] = df.get("title", "")
    out["url"] = df.get("link", df.get("url", ""))
    return out.head(50), "ok", "ok"

@st.cache_data(ttl=30, show_spinner=False)
def uw_flow_alerts(limit=150):
    if not UW_TOKEN:
        return pd.DataFrame(), "missing_key", "UW_TOKEN missing"

    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}"
    }
    params = {"limit": int(limit)}
    r = http_get(UW_FLOW_ALERTS_URL, headers=headers, params=params)
    ok, data, err = try_json(r)
    if not ok:
        return pd.DataFrame(), err or "error", str(data)

    # UW often wraps in {"data":[...]}
    rows = None
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        rows = data["data"]
    elif isinstance(data, list):
        rows = data
    else:
        return pd.DataFrame(), "parse_error", "Unexpected UW response shape"

    df = pd.DataFrame(rows)
    if df.empty:
        return df, "ok", "no flow alerts"

    # Normalize some likely columns
    # created_at might be executed_at
    if "executed_at" in df.columns and "created_at" not in df.columns:
        df["created_at"] = df["executed_at"]
    if "created_at" in df.columns:
        df["created_at_dt"] = df["created_at"].apply(parse_datetime_any)

    # make numeric columns safe
    for c in ["premium", "volume", "open_interest", "underlying_price", "strike", "implied_volatility", "gamma"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, "ok", "ok"

@st.cache_data(ttl=120, show_spinner=False)
def uw_options_volume_bias(ticker: str):
    """
    Uses the endpoint you shared:
    GET /stock/{ticker}/options-volume
    """
    if not UW_TOKEN:
        return None, "missing_key", "UW_TOKEN missing"

    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}"
    }
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/options-volume"
    r = http_get(url, headers=headers)
    ok, data, err = try_json(r)
    if not ok:
        return None, err or "error", str(data)

    rows = None
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
        rows = data["data"][0]
    else:
        return None, "parse_error", "Unexpected options-volume response"

    # Compute a simple bias from bullish/bearish premium if present
    bull = safe_float(rows.get("bullish_premium"))
    bear = safe_float(rows.get("bearish_premium"))
    call_vol = safe_float(rows.get("call_volume"))
    put_vol = safe_float(rows.get("put_volume"))

    bias = "N/A"
    put_call = None

    if call_vol is not None and put_vol is not None and call_vol > 0:
        put_call = put_vol / call_vol

    if bull is not None and bear is not None:
        if bull > bear * 1.1:
            bias = "Bullish"
        elif bear > bull * 1.1:
            bias = "Bearish"
        else:
            bias = "Neutral"

    return {"bias": bias, "put_call": put_call}, "ok", "ok"

@st.cache_data(ttl=600, show_spinner=False)
def fred_10y_latest():
    if not FRED_API_KEY:
        return None, "missing_key", "FRED_API_KEY missing"
    # DGS10 latest
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "series_id": "DGS10",
        "sort_order": "desc",
        "limit": 3
    }
    r = http_get(url, params=params)
    ok, data, err = try_json(r)
    if not ok:
        return None, err or "error", str(data)

    obs = data.get("observations", []) if isinstance(data, dict) else []
    vals = []
    for o in obs:
        v = safe_float(o.get("value"))
        if v is not None:
            vals.append(v)
    if not vals:
        return None, "parse_error", "No numeric yields returned"
    latest = vals[0]
    prior = vals[1] if len(vals) > 1 else None
    return {"latest": latest, "prior": prior}, "ok", "ok"

@st.cache_data(ttl=180, show_spinner=False)
def eodhd_options_chain_iv(ticker: str):
    """
    EODHD options endpoint:
    /options/{ticker}.US?fmt=json&api_token=...
    We extract a representative 'current IV' by taking the nearest expiration ATM-ish contract.
    """
    if not EODHD_API_KEY:
        return None, "missing_key", "EODHD_API_KEY missing"

    url = f"https://eodhd.com/api/options/{ticker}.US"
    params = {"api_token": EODHD_API_KEY, "fmt": "json"}
    r = http_get(url, params=params)
    ok, data, err = try_json(r)
    if not ok:
        return None, err or "error", str(data)

    # Data shape varies by plan; attempt to find contracts
    # Typical: {"data":[{expirationDate:"...", options:{CALL:[...],PUT:[...]}}]}
    contracts = []

    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        for exp_block in data["data"]:
            exp = exp_block.get("expirationDate") or exp_block.get("expiration")
            opts = exp_block.get("options") or {}
            for side_key in ["CALL", "PUT", "call", "put"]:
                lst = opts.get(side_key)
                if isinstance(lst, list):
                    for c in lst:
                        c2 = dict(c)
                        c2["_exp"] = exp
                        contracts.append(c2)
    elif isinstance(data, list):
        # sometimes it's already a list of contracts
        contracts = data
    else:
        return None, "parse_error", "Unexpected options response"

    if not contracts:
        return None, "ok", "no options chain"

    df = pd.DataFrame(contracts)

    # Try common field names
    iv_field = None
    for cand in ["impliedVolatility", "implied_volatility", "iv", "IV"]:
        if cand in df.columns:
            iv_field = cand
            break
    if iv_field is None:
        return None, "parse_error", "No IV field found"

    df[iv_field] = df[iv_field].apply(normalize_iv_to_percent)

    # Choose representative IV: median of top liquid (highest volume) contracts
    vol_field = None
    for cand in ["volume", "Volume"]:
        if cand in df.columns:
            vol_field = cand
            break
    if vol_field:
        df[vol_field] = pd.to_numeric(df[vol_field], errors="coerce").fillna(0)
        df = df.sort_values(vol_field, ascending=False)

    iv_vals = df[iv_field].dropna().tolist()
    if not iv_vals:
        return None, "ok", "no iv values"
    iv_now = float(np.median(iv_vals[:30]))  # robust

    return {"iv_now": iv_now}, "ok", "ok"


# =========================
# Compute ticker score
# =========================
def score_ticker(ticker: str, bars_df: pd.DataFrame, uw_bias_obj, flow_df: pd.DataFrame, teny_obj, iv_obj):
    reason = []
    out = {
        "Ticker": ticker,
        "Confidence": 50,
        "Direction": "—",
        "Signal": "WAIT",
        "Institutional": "NO",
        "RSI": None,
        "MACD_hist": None,
        "VWAP_above": None,
        "EMA_stack": None,
        "Vol_ratio": None,
        "UW_unusual": "NO",
        "UW_bias": "N/A",
        "Put/Call vol": None,
        "IV_now": None,
        "IV_spike": None,
        "Gamma_bias": "N/A",
        "10Y": None,
        "Bars": 0,
        "Last bar (CST)": None,
        "Bars_status": None,
        "News_status": None,
        "UW_flow_status": None,
        "UW_tickerflow_status": None,
        "Reason": ""
    }

    # ---- bars-based indicators
    if bars_df is None:
        out["Bars_status"] = "missing"
        reason.append("No bars (missing key or request failed).")
    else:
        out["Bars"] = int(len(bars_df))
        if len(bars_df) == 0:
            out["Bars_status"] = "ticker_data_empty"
            reason.append("No intraday bars returned for this lookback.")
        else:
            out["Bars_status"] = "ok"
            last_dt = bars_df["datetime"].iloc[-1] if "datetime" in bars_df.columns else None
            try:
                last_dt = last_dt.to_pydatetime()
            except Exception:
                pass
            out["Last bar (CST)"] = to_cst_str(last_dt) if last_dt else "N/A"

            close = bars_df["close"].astype(float).values
            rsi = compute_rsi(close, 14)
            macd_hist = compute_macd_hist(close)
            vwap_val = compute_vwap(bars_df)
            ema9 = compute_ema(close, 9)
            ema20 = compute_ema(close, 20)
            ema50 = compute_ema(close, 50)
            vr = vol_ratio(bars_df, lookback=30)

            out["RSI"] = None if rsi is None else round(rsi, 2)
            out["MACD_hist"] = None if macd_hist is None else round(macd_hist, 4)
            out["Vol_ratio"] = None if vr is None else round(vr, 2)

            last_close = float(close[-1])
            if vwap_val is not None:
                out["VWAP_above"] = "Above" if last_close >= vwap_val else "Below"
            else:
                out["VWAP_above"] = None

            # EMA stack: bullish if 9>20>50, bearish if 9<20<50
            if None not in (ema9, ema20, ema50):
                if ema9 > ema20 > ema50:
                    out["EMA_stack"] = "Bullish"
                elif ema9 < ema20 < ema50:
                    out["EMA_stack"] = "Bearish"
                else:
                    out["EMA_stack"] = "Neutral"
            else:
                out["EMA_stack"] = None

    # ---- UW options volume bias
    if isinstance(uw_bias_obj, dict):
        out["UW_bias"] = uw_bias_obj.get("bias", "N/A")
        out["Put/Call vol"] = None if uw_bias_obj.get("put_call") is None else round(float(uw_bias_obj["put_call"]), 2)

    # ---- IV
    if isinstance(iv_obj, dict) and iv_obj.get("iv_now") is not None:
        out["IV_now"] = round(float(iv_obj["iv_now"]), 2)

    # IV spike detection (simple): if IV_now >= 90th percentile of "typical" range (we don't have history),
    # so we use a heuristic threshold.
    if out["IV_now"] is not None:
        out["IV_spike"] = "YES" if out["IV_now"] >= 60 else "NO"  # tweak later if you want

    # ---- 10Y
    teny_latest = None
    teny_prior = None
    if isinstance(teny_obj, dict):
        teny_latest = teny_obj.get("latest")
        teny_prior = teny_obj.get("prior")
        out["10Y"] = None if teny_latest is None else round(float(teny_latest), 2)

    # ---- UW flow alerts -> "unusual" + directional pressure
    flow_score = 0.0
    if flow_df is not None and not flow_df.empty:
        # Filter to this ticker if possible
        tcol = None
        for c in ["underlying_symbol", "ticker", "symbol"]:
            if c in flow_df.columns:
                tcol = c
                break

        df_t = flow_df
        if tcol:
            df_t = flow_df[flow_df[tcol].astype(str).str.upper() == ticker].copy()

        # Apply your strict rules:
        # premium >= 1,000,000 ; DTE <= 3 ; volume > OI ; exclude ITM
        if not df_t.empty:
            out["UW_unusual"] = "YES"
            # normalize premium numeric
            if "premium" in df_t.columns:
                df_t["premium"] = pd.to_numeric(df_t["premium"], errors="coerce")
            if "volume" in df_t.columns:
                df_t["volume"] = pd.to_numeric(df_t["volume"], errors="coerce")
            if "open_interest" in df_t.columns:
                df_t["open_interest"] = pd.to_numeric(df_t["open_interest"], errors="coerce")
            if "strike" in df_t.columns:
                df_t["strike"] = pd.to_numeric(df_t["strike"], errors="coerce")
            if "underlying_price" in df_t.columns:
                df_t["underlying_price"] = pd.to_numeric(df_t["underlying_price"], errors="coerce")

            # DTE
            if "expiry" in df_t.columns:
                df_t["expiry_dt"] = pd.to_datetime(df_t["expiry"], errors="coerce")
                df_t["dte"] = (df_t["expiry_dt"].dt.date - date.today()).apply(lambda x: x.days if pd.notna(x) else None)
            else:
                df_t["dte"] = None

            # option_type
            ot = None
            for c in ["option_type", "type", "put_call"]:
                if c in df_t.columns:
                    ot = c
                    break

            # ITM filter
            def is_itm(row):
                try:
                    opt = str(row.get(ot, "")).lower()
                    strike = safe_float(row.get("strike"))
                    und = safe_float(row.get("underlying_price"))
                    if strike is None or und is None:
                        return False
                    if "call" in opt:
                        return strike < und
                    if "put" in opt:
                        return strike > und
                    return False
                except Exception:
                    return False

            df_t["itm"] = df_t.apply(is_itm, axis=1)

            # Apply filters
            df_f = df_t.copy()
            if "premium" in df_f.columns:
                df_f = df_f[df_f["premium"].fillna(0) >= 1_000_000]
            if "dte" in df_f.columns:
                df_f = df_f[df_f["dte"].fillna(999) <= 3]
            if "volume" in df_f.columns and "open_interest" in df_f.columns:
                df_f = df_f[df_f["volume"].fillna(0) > df_f["open_interest"].fillna(0)]
            df_f = df_f[df_f["itm"] == False]

            # directional pressure (calls vs puts)
            if ot and not df_f.empty:
                calls = df_f[df_f[ot].astype(str).str.lower().str.contains("call")]
                puts = df_f[df_f[ot].astype(str).str.lower().str.contains("put")]

                call_prem = calls["premium"].sum() if "premium" in calls.columns else 0
                put_prem = puts["premium"].sum() if "premium" in puts.columns else 0

                # flow_score in [-1, +1]
                denom = (call_prem + put_prem)
                if denom > 0:
                    flow_score = (call_prem - put_prem) / denom

                # Gamma bias proxy if gamma exists
                if "gamma" in df_f.columns:
                    # crude: weighted gamma by premium for calls vs puts
                    df_f["gamma"] = pd.to_numeric(df_f["gamma"], errors="coerce")
                    df_f["gxp"] = df_f["gamma"].fillna(0) * df_f["premium"].fillna(0)
                    cg = calls["gxp"].sum() if "gxp" in calls.columns else 0
                    pg = puts["gxp"].sum() if "gxp" in puts.columns else 0
                    if cg > pg * 1.1:
                        out["Gamma_bias"] = "Positive (proxy)"
                    elif pg > cg * 1.1:
                        out["Gamma_bias"] = "Negative (proxy)"
                    else:
                        out["Gamma_bias"] = "Neutral"

    # =========================
    # Score composition -> confidence 0..100
    # =========================
    score = 0.0
    weight_sum = 0.0

    def add_component(val, w):
        nonlocal score, weight_sum
        if val is None:
            return
        score += (val * w)
        weight_sum += w

    # RSI: bullish if >50, bearish if <50 (simple)
    rsi = out["RSI"]
    if rsi is not None:
        rsi_val = (rsi - 50) / 50  # -1..+1
        add_component(clamp(rsi_val, -1, 1), w_rsi)

    # MACD hist: >0 bullish, <0 bearish
    mh = out["MACD_hist"]
    if mh is not None:
        add_component(clamp(mh / 0.5, -1, 1), w_macd)  # scale

    # VWAP above/below
    if out["VWAP_above"] == "Above":
        add_component(+1, w_vwap)
    elif out["VWAP_above"] == "Below":
        add_component(-1, w_vwap)

    # EMA stack
    if out["EMA_stack"] == "Bullish":
        add_component(+1, w_ema)
    elif out["EMA_stack"] == "Bearish":
        add_component(-1, w_ema)
    elif out["EMA_stack"] == "Neutral":
        add_component(0, w_ema)

    # Volume ratio: >1 bullish momentum, <1 bearish (weak)
    vr = out["Vol_ratio"]
    if vr is not None:
        add_component(clamp((vr - 1.0), -1, 1), w_volr)

    # UW bias (premium bias)
    if out["UW_bias"] == "Bullish":
        add_component(+0.6, w_uw)
    elif out["UW_bias"] == "Bearish":
        add_component(-0.6, w_uw)
    elif out["UW_bias"] == "Neutral":
        add_component(0, w_uw)

    # UW flow score from alerts
    if flow_score != 0:
        add_component(clamp(flow_score, -1, 1), w_uw)

    # 10Y filter: rising yields = slight headwind to calls; falling yields = slight tailwind
    if teny_latest is not None and teny_prior is not None:
        delta = teny_latest - teny_prior
        # delta > 0 -> bearish for calls
        add_component(clamp(-delta / 0.20, -1, 1), w_10y)

    # News placeholder (kept small) — you can replace later with real sentiment
    add_component(0, w_news)

    if weight_sum <= 0:
        conf = 50
    else:
        # score is in roughly [-weight_sum, +weight_sum]
        norm = score / weight_sum  # -1..+1
        conf = int(round(50 + 50 * clamp(norm, -1, 1)))

    out["Confidence"] = clamp(conf, 0, 100)

    # Direction + CALLS/PUTS only output
    if out["Confidence"] >= 55:
        if score >= 0:
            out["Direction"] = "BULLISH"
            out["Signal"] = "BUY CALLS"
        else:
            out["Direction"] = "BEARISH"
            out["Signal"] = "BUY PUTS"
    else:
        out["Direction"] = "NEUTRAL"
        out["Signal"] = "WAIT"

    # Institutional mode
    if out["Confidence"] >= inst_threshold and out["Signal"] in ("BUY CALLS", "BUY PUTS"):
        out["Institutional"] = "YES"
    else:
        out["Institutional"] = "NO"

    out["Reason"] = "; ".join(reason) if reason else ""
    return out


# =========================
# Layout
# =========================
left, right = st.columns([1.2, 1.0], gap="large")

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("This is embedded. True filtering (DTE/ITM/premium rules) is best done inside the screener itself.")
    st.components.v1.iframe(
        "https://unusualwhales.com/options-screener",
        height=680,
        scrolling=True
    )

with right:
    st.subheader("Endpoints status")

    # Endpoint checks
    # EODHD intraday check on first ticker only (if exists)
    eod_intr_status = ("warn", "No tickers")
    eod_intr_detail = ""
    if tickers:
        df_bars, s, d = eodhd_intraday_bars(tickers[0], price_lookback_min)
        if s == "ok":
            eod_intr_status = ("ok", "EODHD intraday bars (ok)")
            eod_intr_detail = ""
        elif s == "ticker_data_empty":
            eod_intr_status = ("warn", "EODHD intraday bars (empty)")
            eod_intr_detail = d
        else:
            eod_intr_status = ("err", f"EODHD intraday bars ({s})")
            eod_intr_detail = d

    if eod_intr_status[0] == "ok":
        badge(eod_intr_status[1], "ok")
    elif eod_intr_status[0] == "warn":
        badge(eod_intr_status[1], "warn", eod_intr_detail)
    else:
        badge(eod_intr_status[1], "err", eod_intr_detail)

    flow_df, uw_flow_s, uw_flow_d = uw_flow_alerts(limit=200)
    if uw_flow_s == "ok":
        badge("UW flow-alerts (ok)", "ok")
    else:
        badge(f"UW flow-alerts ({uw_flow_s})", "err", uw_flow_d)

    if tickers:
        news_df, ns, nd = eodhd_news(tickers[0], news_lookback_min)
        if ns == "ok":
            badge("EODHD news (ok)", "ok", ("no headlines" if nd == "no headlines" else None))
        else:
            badge(f"EODHD news ({ns})", "err", nd)
    else:
        badge("EODHD news", "warn", "No tickers")

    teny_obj, teny_s, teny_d = fred_10y_latest()
    if teny_s == "ok":
        badge("FRED 10Y yield (ok)", "ok")
    else:
        badge(f"FRED 10Y yield ({teny_s})", "warn", teny_d)

    st.divider()

    st.subheader("Live Score / Signals (EODHD intraday + EODHD headlines + UW flow)")

    # Build table
    rows = []
    uw_bias_status = {}
    iv_status = {}

    teny = teny_obj if teny_s == "ok" else None

    for t in tickers:
        bars_df, bars_s, bars_d = eodhd_intraday_bars(t, price_lookback_min)
        uwb, uwb_s, uwb_d = uw_options_volume_bias(t)
        uw_bias_status[t] = (uwb_s, uwb_d)
        iv_obj, iv_s, iv_d = eodhd_options_chain_iv(t)
        iv_status[t] = (iv_s, iv_d)

        out = score_ticker(
            ticker=t,
            bars_df=bars_df if bars_s in ("ok", "ticker_data_empty") else None,
            uw_bias_obj=uwb if uwb_s == "ok" else None,
            flow_df=flow_df if uw_flow_s == "ok" else pd.DataFrame(),
            teny_obj=teny,
            iv_obj=iv_obj if iv_s == "ok" else None
        )

        # Carry statuses
        out["Bars_status"] = bars_s
        out["UW_flow_status"] = uw_flow_s
        out["News_status"] = "ok"  # per-ticker news displayed below
        # UW ticker options-flow removed (was 404 for you); keep a status slot
        out["UW_tickerflow_status"] = "disabled"
        rows.append(out)

    live_df = pd.DataFrame(rows)
    show_cols = [
        "Ticker","Confidence","Direction","Signal","Institutional",
        "RSI","MACD_hist","VWAP_above","EMA_stack","Vol_ratio",
        "UW_unusual","UW_bias","Put/Call vol",
        "IV_now","IV_spike","Gamma_bias","10Y",
        "Bars","Last bar (CST)","Bars_status","News_status","UW_flow_status","Reason"
    ]
    st.dataframe(live_df[show_cols], use_container_width=True, height=220)

    st.subheader(f"Institutional Alerts (≥ {inst_threshold} only)")
    inst = live_df[(live_df["Institutional"] == "YES")].copy()
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        st.dataframe(inst[["Ticker","Confidence","Signal","UW_bias","Put/Call vol","IV_spike","10Y","Reason"]], use_container_width=True, height=180)

    st.subheader("Unusual Flow Alerts (UW API) — filtered")
    st.caption("Rules applied: premium ≥ $1M, DTE ≤ 3, volume > OI, exclude ITM (best-effort based on fields available).")

    if uw_flow_s == "ok" and not flow_df.empty:
        df_show = flow_df.copy()

        # Filter by tickers in list (if underlying_symbol exists)
        if "underlying_symbol" in df_show.columns:
            df_show = df_show[df_show["underlying_symbol"].astype(str).str.upper().isin(tickers)]

        # premium filter
        if "premium" in df_show.columns:
            df_show["premium"] = pd.to_numeric(df_show["premium"], errors="coerce").fillna(0)
            df_show = df_show[df_show["premium"] >= 1_000_000]

        # DTE
        if "expiry" in df_show.columns:
            df_show["expiry_dt"] = pd.to_datetime(df_show["expiry"], errors="coerce")
            df_show["dte"] = (df_show["expiry_dt"].dt.date - date.today()).apply(lambda x: x.days if pd.notna(x) else 999)
            df_show = df_show[df_show["dte"] <= 3]

        # volume > OI
        if "volume" in df_show.columns and "open_interest" in df_show.columns:
            df_show["volume"] = pd.to_numeric(df_show["volume"], errors="coerce").fillna(0)
            df_show["open_interest"] = pd.to_numeric(df_show["open_interest"], errors="coerce").fillna(0)
            df_show = df_show[df_show["volume"] > df_show["open_interest"]]

        # exclude ITM
        if {"option_type","strike","underlying_price"}.issubset(df_show.columns):
            df_show["strike"] = pd.to_numeric(df_show["strike"], errors="coerce")
            df_show["underlying_price"] = pd.to_numeric(df_show["underlying_price"], errors="coerce")

            def itm_row(r):
                opt = str(r.get("option_type","")).lower()
                strike = safe_float(r.get("strike"))
                und = safe_float(r.get("underlying_price"))
                if strike is None or und is None:
                    return False
                if "call" in opt:
                    return strike < und
                if "put" in opt:
                    return strike > und
                return False

            df_show["itm"] = df_show.apply(itm_row, axis=1)
            df_show = df_show[df_show["itm"] == False]

        cols = []
        for c in ["created_at","executed_at","underlying_symbol","option_chain_id","option_type","strike","expiry","premium","volume","open_interest","underlying_price","gamma","implied_volatility"]:
            if c in df_show.columns:
                cols.append(c)

        if not cols:
            st.info("Flow alerts returned, but expected columns were not present.")
        else:
            st.dataframe(df_show[cols].head(200), use_container_width=True, height=220)
    else:
        st.warning(f"UW flow alerts not available: {uw_flow_s} — {uw_flow_d}")

    st.subheader(f"News — last {int(news_lookback_min)} minutes (EODHD)")
    all_news = []
    for t in tickers[:6]:  # keep it light
        ndf, ns, nd = eodhd_news(t, news_lookback_min)
        if ns == "ok" and not ndf.empty:
            all_news.append(ndf)
    if all_news:
        news_all = pd.concat(all_news, ignore_index=True)
        # Ensure safe columns even if provider changes
        for col in ["ticker","published_cst","source","title","url"]:
            if col not in news_all.columns:
                news_all[col] = ""
        st.dataframe(news_all[["ticker","published_cst","source","title","url"]].head(80), use_container_width=True, height=220)
        st.caption("Tip: Click URL column links (or copy/paste).")
    else:
        st.info("No news in this lookback window (or EODHD returned none).")

    with st.expander("What None/N/A means (plain English)"):
        st.write(
            "- **Bars_status = ticker_data_empty**: EODHD returned no candles for that time window.\n"
            "- **RSI/MACD/VWAP/EMA = None**: Not enough candles yet to compute (or bars empty).\n"
            "- **UW_bias = N/A**: UW options-volume endpoint didn’t return premiums/volumes.\n"
            "- **IV_now**: From EODHD options chain; normalized to percent. If it looks insane, it’s usually scaling.\n"
            "- **UW flow filters** are strict (premium ≥ $1M, DTE ≤ 3, vol>OI, exclude ITM) — it’s normal to see fewer rows."
        )

# Auto-refresh
time.sleep(0.1)
st.caption(f"Last update (CST): {to_cst_str(now_cst())}")
st.experimental_set_query_params(t=str(int(time.time())))
