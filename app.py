import os
import math
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Institutional Options Signals (5m) — CALLS/PUTS ONLY", layout="wide")

TZ = ZoneInfo("America/Chicago")  # CST/CDT auto-handled
NOW_CST = lambda: datetime.now(tz=TZ)

DEFAULT_QUICK = ["SPY", "QQQ", "IWM", "DIA", "TSLA", "NVDA", "AMD"]

# Unusual Whales screener URL (web view) — your rules:
# $1M premium min, DTE<=3, stocks+ETF only, volume>OI, exclude ITM
UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF"
    "&min_premium=1000000"
    "&max_dte=3"
    "&min_volume_oi_ratio=1"
    "&order=premium&order_direction=desc"
)

# ============================================================
# SECRETS
# ============================================================
def get_secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, default)
    except Exception:
        v = default
    if v is None:
        v = default
    return str(v).strip()

EODHD_API_KEY = get_secret("EODHD_API_KEY")
UW_TOKEN = get_secret("UW_TOKEN")  # Bearer token
UW_FLOW_ALERTS_URL = get_secret("UW_FLOW_ALERTS_URL")  # must be FULL URL
FINVIZ_AUTH = get_secret("FINVIZ_AUTH")  # optional (not used here)


# ============================================================
# SMALL HELPERS
# ============================================================
def safe_upper_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    # allow letters, numbers, dot, dash (for some tickers)
    cleaned = "".join(ch for ch in t if ch.isalnum() or ch in [".", "-"])
    return cleaned

def as_eodhd_symbol(ticker: str) -> str:
    """
    EODHD uses e.g. AAPL.US
    If user already typed a suffix (like .US), keep it.
    """
    t = safe_upper_ticker(ticker)
    if not t:
        return ""
    if "." in t:
        return t
    return f"{t}.US"

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def http_get(url, params=None, headers=None, timeout=20):
    return requests.get(url, params=params, headers=headers, timeout=timeout)

def fmt_num(x, digits=2):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "N/A"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


# ============================================================
# TECH INDICATORS (pandas)
# ============================================================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, pd.NA)
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill")

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return hist

def vwap(df: pd.DataFrame) -> pd.Series:
    # VWAP = cumulative(sum(price*volume)) / cumulative(sum(volume))
    # use typical price (H+L+C)/3 if available, else close
    if all(c in df.columns for c in ["high", "low", "close", "volume"]):
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].replace(0, pd.NA)
        cum_pv = (tp * vol).cumsum()
        cum_v = vol.cumsum()
        out = cum_pv / cum_v
        return out.fillna(method="bfill")
    elif "close" in df.columns and "volume" in df.columns:
        vol = df["volume"].replace(0, pd.NA)
        cum_pv = (df["close"] * vol).cumsum()
        cum_v = vol.cumsum()
        out = cum_pv / cum_v
        return out.fillna(method="bfill")
    else:
        return pd.Series([pd.NA] * len(df), index=df.index)


# ============================================================
# DATA SOURCES
# ============================================================
@st.cache_data(ttl=60)
def eodhd_intraday(symbol: str, interval: str = "5m", lookback_minutes: int = 240):
    """
    Returns dataframe with columns: datetime, open, high, low, close, volume
    """
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"

    url = f"https://eodhd.com/api/intraday/{symbol}"
    # request a window slightly bigger than needed
    params = {
        "api_token": EODHD_API_KEY,
        "interval": interval,
        "fmt": "json",
    }

    try:
        r = http_get(url, params=params, timeout=20)
        if r.status_code == 429:
            return pd.DataFrame(), "rate_limited"
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame(), "empty"

        df = pd.DataFrame(data)
        # EODHD returns "datetime" like "2026-02-18 12:55:00"
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"]).sort_values("datetime")
        else:
            return pd.DataFrame(), "bad_format"

        # keep only last N minutes if possible
        if lookback_minutes and "datetime" in df.columns:
            cutoff = pd.Timestamp(NOW_CST() - timedelta(minutes=int(lookback_minutes))).tz_localize(None)
            df = df[df["datetime"] >= cutoff]

        # normalize numeric columns
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["close"])
        return df, "ok"
    except Exception as e:
        return pd.DataFrame(), f"error: {type(e).__name__}"

@st.cache_data(ttl=120)
def eodhd_news(symbol: str, lookback_minutes: int = 60, limit: int = 20):
    """
    EODHD News endpoint:
    https://eodhd.com/api/news?s=AAPL.US&from=YYYY-MM-DD&to=YYYY-MM-DD&api_token=...&fmt=json
    """
    if not EODHD_API_KEY:
        return [], "missing_key"

    url = "https://eodhd.com/api/news"
    now = NOW_CST()
    start = now - timedelta(minutes=int(lookback_minutes))
    params = {
        "s": symbol,
        "from": start.strftime("%Y-%m-%d"),
        "to": now.strftime("%Y-%m-%d"),
        "limit": int(limit),
        "api_token": EODHD_API_KEY,
        "fmt": "json",
    }
    try:
        r = http_get(url, params=params, timeout=20)
        if r.status_code == 429:
            return [], "rate_limited"
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            return [], "bad_format"

        # Filter to last lookback_minutes using published datetime if present
        items = []
        for it in data:
            published = it.get("date") or it.get("datetime") or it.get("published")
            # EODHD uses "date": "2026-02-18 11:20:00"
            dt = None
            if published:
                dt = pd.to_datetime(published, errors="coerce")
            if dt is not None and not pd.isna(dt):
                # treat as CST naive
                if dt.to_pydatetime() >= (now.replace(tzinfo=None) - timedelta(minutes=int(lookback_minutes))):
                    items.append(it)
            else:
                items.append(it)

        return items, "ok"
    except Exception as e:
        return [], f"error: {type(e).__name__}"

def simple_news_sentiment(headline: str) -> int:
    """
    Very simple sentiment: +1 / 0 / -1
    (You can later replace with a better model—this is stable + fast.)
    """
    if not headline:
        return 0
    h = headline.lower()
    pos = ["beats", "surge", "rally", "up", "wins", "strong", "record", "bull", "growth", "upgrade"]
    neg = ["miss", "plunge", "down", "weak", "fraud", "lawsuit", "bear", "cut", "downgrade", "probe"]
    score = 0
    for w in pos:
        if w in h:
            score += 1
    for w in neg:
        if w in h:
            score -= 1
    return 1 if score > 0 else (-1 if score < 0 else 0)

@st.cache_data(ttl=120)
def uw_options_volume_bias(ticker: str):
    """
    Unusual Whales:
    GET https://api.unusualwhales.com/api/stock/{ticker}/options-volume
    Header: Authorization: Bearer <token>
    """
    if not UW_TOKEN:
        return None, "missing_key"

    url = f"https://api.unusualwhales.com/api/stock/{safe_upper_ticker(ticker)}/options-volume"
    headers = {"Accept": "application/json, text/plain", "Authorization": f"Bearer {UW_TOKEN}"}
    try:
        r = http_get(url, headers=headers, timeout=20)
        if r.status_code == 429:
            return None, "rate_limited"
        r.raise_for_status()
        js = r.json()
        data = js.get("data", [])
        if not data:
            return None, "empty"
        latest = data[0]  # API usually returns a list, newest first
        # bias
        bullish_prem = float(latest.get("bullish_premium", 0) or 0)
        bearish_prem = float(latest.get("bearish_premium", 0) or 0)
        call_vol = float(latest.get("call_volume", 0) or 0)
        put_vol = float(latest.get("put_volume", 0) or 0)
        call_oi = float(latest.get("call_open_interest", 0) or 0)
        put_oi = float(latest.get("put_open_interest", 0) or 0)

        # "Gamma bias" proxy (not true GEX): OI imbalance
        gamma_proxy = (call_oi - put_oi)
        gamma_bias = "Neutral"
        if gamma_proxy > 0:
            gamma_bias = "Positive Gamma (proxy)"
        elif gamma_proxy < 0:
            gamma_bias = "Negative Gamma (proxy)"

        uw_bias = "Neutral"
        if bullish_prem > bearish_prem and call_vol >= put_vol:
            uw_bias = "Bullish"
        elif bearish_prem > bullish_prem and put_vol >= call_vol:
            uw_bias = "Bearish"

        return {
            "uw_bias": uw_bias,
            "bullish_premium": bullish_prem,
            "bearish_premium": bearish_prem,
            "call_vol": call_vol,
            "put_vol": put_vol,
            "gamma_bias": gamma_bias
        }, "ok"
    except requests.HTTPError:
        return None, f"http_{r.status_code}"
    except Exception as e:
        return None, f"error: {type(e).__name__}"

@st.cache_data(ttl=30)
def uw_flow_alerts(ticker: str, limit: int = 50):
    """
    Uses YOUR EXACT UW_FLOW_ALERTS_URL from Secrets.
    If that endpoint is wrong / plan-restricted, we show it as RED but we do not crash.
    """
    if not UW_TOKEN:
        return [], "missing_key"

    if not UW_FLOW_ALERTS_URL:
        return [], "missing_url"

    url = UW_FLOW_ALERTS_URL.strip()  # do NOT modify (no underscore/dash changes)
    headers = {"Accept": "application/json, text/plain", "Authorization": f"Bearer {UW_TOKEN}"}

    # params are best-effort; if UW ignores them, fine.
    params = {
        "limit": int(limit),
        "ticker": safe_upper_ticker(ticker),
        "min_premium": 1000000,
        "max_dte": 3,
        "exclude_itm": True,
        "volume_gt_oi": True,
        "order": "desc",
    }

    try:
        r = http_get(url, params=params, headers=headers, timeout=20)
        if r.status_code == 429:
            return [], "rate_limited"
        if r.status_code == 404:
            # Plan/route issue is common here
            return [], "http_404"
        r.raise_for_status()
        js = r.json()

        # UW sometimes returns {"data":[...]} or a raw list. Handle both.
        if isinstance(js, dict) and "data" in js:
            items = js.get("data") or []
        elif isinstance(js, list):
            items = js
        else:
            items = []

        # ticker-filter again just in case
        out = []
        for it in items:
            sym = (it.get("underlying_symbol") or it.get("ticker") or "").upper()
            if sym == safe_upper_ticker(ticker):
                out.append(it)

        return out, "ok"
    except requests.HTTPError:
        return [], f"http_{getattr(r, 'status_code', 'ERR')}"
    except Exception as e:
        return [], f"error: {type(e).__name__}"

@st.cache_data(ttl=300)
def ten_year_yield_optional():
    """
    Optional 10Y source (no key): Stooq CSV.
    If it fails, return None (still stable).
    """
    try:
        # Stooq US10Y symbol varies; this works often:
        # daily: https://stooq.com/q/d/l/?s=us10y&i=d
        url = "https://stooq.com/q/d/l/?s=us10y&i=d"
        r = http_get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(r.text))  # may fail on some envs
        if df.empty:
            return None, "empty"
        # last two closes
        df = df.dropna()
        if len(df) < 2:
            return None, "too_short"
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        chg = last - prev
        return {"last": last, "chg": chg}, "ok"
    except Exception:
        # keep it quiet and stable
        return None, "not_available"


# ============================================================
# SCORING
# ============================================================
def score_ticker(df_bars: pd.DataFrame, news_items: list, uw_bias_obj: dict | None, ten_y: dict | None,
                 w_rsi: float, w_macd: float, w_vwap: float, w_ema: float, w_vol: float, w_uw: float, w_news: float, w_10y: float):
    """
    Returns:
      score_0_100, direction, signal (BUY CALLS/BUY PUTS/WAIT), components dict
    """
    # Default neutral
    comp = {
        "RSI": None,
        "MACD_hist": None,
        "VWAP": None,
        "EMA_stack": None,
        "Vol_ratio": None,
        "IV_spike": None,     # optional (requires alerts with IV)
        "Gamma_bias": "N/A",
        "UW_bias": "Neutral",
        "UW_unusual": "NO",
        "10Y": "N/A",
        "News_sent": 0
    }

    # ---- News sentiment
    if news_items:
        s = 0
        for it in news_items[:20]:
            title = it.get("title") or it.get("Title") or ""
            s += simple_news_sentiment(title)
        # clamp to -3..+3
        comp["News_sent"] = int(clamp(s, -3, 3))
    else:
        comp["News_sent"] = 0

    # ---- UW bias + Gamma proxy
    if uw_bias_obj:
        comp["UW_bias"] = uw_bias_obj.get("uw_bias", "Neutral")
        comp["Gamma_bias"] = uw_bias_obj.get("gamma_bias", "N/A")

    # ---- Indicators (if we have bars)
    if df_bars is not None and not df_bars.empty and "close" in df_bars.columns:
        df = df_bars.copy()
        close = df["close"].astype(float)

        # RSI
        rsi14 = rsi(close, 14)
        comp["RSI"] = float(rsi14.iloc[-1])

        # MACD hist
        mh = macd_hist(close)
        comp["MACD_hist"] = float(mh.iloc[-1])

        # EMA stack (9/20/50)
        ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        # bullish if 9>20>50; bearish if 9<20<50
        if ema9 > ema20 > ema50:
            comp["EMA_stack"] = "Bullish"
        elif ema9 < ema20 < ema50:
            comp["EMA_stack"] = "Bearish"
        else:
            comp["EMA_stack"] = "Neutral"

        # VWAP
        vw = vwap(df)
        vwap_last = vw.iloc[-1] if len(vw) else pd.NA
        comp["VWAP"] = float(vwap_last) if pd.notna(vwap_last) else None

        # Volume ratio = last volume / avg volume (last 30 bars)
        if "volume" in df.columns and df["volume"].notna().any():
            vol = df["volume"].fillna(0).astype(float)
            base = vol.tail(30).mean() if len(vol) >= 10 else vol.mean()
            comp["Vol_ratio"] = float(vol.iloc[-1] / base) if base and base > 0 else None
        else:
            comp["Vol_ratio"] = None
    else:
        # keep N/A
        pass

    # ---- 10Y
    if ten_y and isinstance(ten_y, dict):
        comp["10Y"] = f"{fmt_num(ten_y.get('last'), 2)} ({fmt_num(ten_y.get('chg'), 2)})"

    # ========================================================
    # Convert components to normalized signals in [-1, +1]
    # ========================================================
    sigs = []

    # RSI: bullish if rising from oversold-ish, bearish if overbought-ish
    if comp["RSI"] is not None:
        r = comp["RSI"]
        if r <= 35:
            sigs.append(("rsi", +0.6))
        elif r >= 65:
            sigs.append(("rsi", -0.6))
        else:
            sigs.append(("rsi", 0.0))

    # MACD hist: positive bullish, negative bearish (scaled)
    if comp["MACD_hist"] is not None:
        m = comp["MACD_hist"]
        sigs.append(("macd", clamp(m * 10.0, -1.0, 1.0)))  # scale small numbers

    # VWAP: price above vwap bullish, below bearish
    if df_bars is not None and not df_bars.empty and comp["VWAP"] is not None:
        last_px = float(df_bars["close"].iloc[-1])
        vw = float(comp["VWAP"])
        sigs.append(("vwap", +0.7 if last_px > vw else (-0.7 if last_px < vw else 0.0)))

    # EMA stack
    if comp["EMA_stack"] == "Bullish":
        sigs.append(("ema", +0.7))
    elif comp["EMA_stack"] == "Bearish":
        sigs.append(("ema", -0.7))
    else:
        sigs.append(("ema", 0.0))

    # Volume ratio: spike favors continuation (direction decided by trend proxies)
    if comp["Vol_ratio"] is not None:
        vr = comp["Vol_ratio"]
        # cap: if >2 = strong
        sigs.append(("vol", clamp((vr - 1.0) / 1.5, -1.0, 1.0)))

    # UW bias
    if comp["UW_bias"] == "Bullish":
        sigs.append(("uw", +0.8))
    elif comp["UW_bias"] == "Bearish":
        sigs.append(("uw", -0.8))
    else:
        sigs.append(("uw", 0.0))

    # News sentiment
    ns = comp["News_sent"]
    sigs.append(("news", clamp(ns / 3.0, -1.0, 1.0)))

    # 10Y: rising yields = mild bearish; falling yields = mild bullish
    if ten_y and isinstance(ten_y, dict) and ten_y.get("chg") is not None:
        chg = float(ten_y["chg"])
        sigs.append(("10y", clamp(-chg * 0.5, -0.5, 0.5)))  # mild

    # ========================================================
    # Weighted sum
    # ========================================================
    weights = {
        "rsi": w_rsi,
        "macd": w_macd,
        "vwap": w_vwap,
        "ema": w_ema,
        "vol": w_vol,
        "uw": w_uw,
        "news": w_news,
        "10y": w_10y,
    }
    total_w = sum(max(0.0, v) for v in weights.values()) or 1.0

    agg = 0.0
    for name, val in sigs:
        agg += (weights.get(name, 0.0) * float(val))
    agg_norm = agg / total_w
    agg_norm = clamp(agg_norm, -1.0, 1.0)

    confidence = int(round(50 + 50 * abs(agg_norm)))
    direction = "BULLISH" if agg_norm > 0.12 else ("BEARISH" if agg_norm < -0.12 else "NEUTRAL")

    # CALLS/PUTS only
    if direction == "BULLISH":
        signal = "BUY CALLS" if confidence >= 60 else "WAIT"
    elif direction == "BEARISH":
        signal = "BUY PUTS" if confidence >= 60 else "WAIT"
    else:
        signal = "WAIT"

    return confidence, direction, signal, comp


# ============================================================
# UI — SIDEBAR
# ============================================================
st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")
st.caption(f"Last update (CST): {NOW_CST().strftime('%Y-%m-%d %H:%M:%S %Z')}")

# refresh
with st.sidebar:
    st.header("Settings")

    # 1) Let user type ANY ticker
    tickers_text = st.text_input(
        "Type tickers (comma-separated). Example: SPY,TSLA,NVDA",
        value="SPY,TSLA",
        help="You can type ANY ticker here. We auto-uppercase it."
    )

    # 2) Optional quick pick list (just convenience)
    quick_pick = st.multiselect("Quick pick (optional)", DEFAULT_QUICK, default=[])

    # Combine & dedupe
    typed = [safe_upper_ticker(x) for x in tickers_text.split(",")]
    typed = [x for x in typed if x]
    combined = []
    for t in typed + quick_pick:
        if t and t not in combined:
            combined.append(t)
    tickers = combined[:25]  # safety cap

    st.divider()
    news_lookback = st.number_input("News lookback (minutes)", 1, 360, 60, 1)
    price_lookback = st.number_input("Price lookback (minutes)", 30, 780, 240, 5)

    st.divider()
    st.subheader("Refresh")
    refresh_sec = st.slider("Auto-refresh (seconds)", 10, 300, 30, 5)
    st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh")

    st.divider()
    st.subheader("Institutional mode")
    institutional_min = st.slider("Signals only if confidence ≥", 50, 95, 75, 1)

    st.divider()
    st.subheader("Weights (sum doesn't have to be 1)")

    w_rsi = st.slider("RSI weight", 0.0, 0.40, 0.15, 0.01)
    w_macd = st.slider("MACD weight", 0.0, 0.40, 0.15, 0.01)
    w_vwap = st.slider("VWAP weight", 0.0, 0.40, 0.15, 0.01)
    w_ema = st.slider("EMA stack (9/20/50) weight", 0.0, 0.40, 0.18, 0.01)
    w_vol = st.slider("Volume ratio weight", 0.0, 0.40, 0.12, 0.01)
    w_uw = st.slider("UW flow/bias weight", 0.0, 0.60, 0.20, 0.01)
    w_news = st.slider("News weight", 0.0, 0.30, 0.05, 0.01)
    w_10y = st.slider("10Y yield weight (optional)", 0.0, 0.20, 0.05, 0.01)

    st.divider()
    st.subheader("Keys status (green/red)")
    st.success("EODHD_API_KEY ✅" if EODHD_API_KEY else "EODHD_API_KEY ❌")
    st.success("UW_TOKEN (Bearer) ✅" if UW_TOKEN else "UW_TOKEN ❌")
    st.success("UW_FLOW_ALERTS_URL ✅" if UW_FLOW_ALERTS_URL else "UW_FLOW_ALERTS_URL ❌")
    if FINVIZ_AUTH:
        st.info("FINVIZ_AUTH present (not used in this build)")

# ============================================================
# LAYOUT
# ============================================================
left, right = st.columns([1.35, 1.0])

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("This is embedded. True filtering (DTE/ITM/premium rules) is best done inside the screener itself.")
    st.components.v1.iframe(UW_SCREENER_URL, height=820, scrolling=True)

# ============================================================
# RIGHT PANEL: DATA + SIGNALS
# ============================================================
with right:
    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW bias)")
    st.caption(f"Last update (CST): {NOW_CST().strftime('%Y-%m-%d %H:%M:%S %Z')}")

    if not tickers:
        st.warning("Type at least 1 ticker in the sidebar (comma-separated).")
        st.stop()

    # 10Y optional (does not break app if unavailable)
    teny, teny_status = ten_year_yield_optional()

    rows = []
    endpoint_notes = {
        "eodhd_intraday": [],
        "eodhd_news": [],
        "uw_options_volume": [],
        "uw_flow_alerts": []
    }

    # Build signals
    for t in tickers:
        sym = as_eodhd_symbol(t)

        bars, s_bars = eodhd_intraday(sym, interval="5m", lookback_minutes=int(price_lookback))
        news, s_news = eodhd_news(sym, lookback_minutes=int(news_lookback), limit=30)
        uw_bias_obj, s_uw = uw_options_volume_bias(t)

        # Flow alerts are OPTIONAL; we try but do not crash if broken
        alerts, s_alerts = uw_flow_alerts(t, limit=50)

        # If alerts present, set UW_unusual quickly (premium >= 1M and put/call)
        uw_unusual = "NO"
        uw_dir = "Neutral"
        if alerts and isinstance(alerts, list):
            for it in alerts[:50]:
                prem = float(it.get("premium") or 0)
                otype = (it.get("option_type") or it.get("type") or "").lower()
                tags = it.get("tags") or []
                if prem >= 1_000_000:
                    uw_unusual = "YES"
                    # direction from tags or option_type
                    if isinstance(tags, list) and any("bearish" in str(x).lower() for x in tags):
                        uw_dir = "Bearish"
                    elif isinstance(tags, list) and any("bullish" in str(x).lower() for x in tags):
                        uw_dir = "Bullish"
                    elif otype == "put":
                        uw_dir = "Bearish"
                    elif otype == "call":
                        uw_dir = "Bullish"
                    break

        confidence, direction, signal, comp = score_ticker(
            bars, news, uw_bias_obj, teny,
            w_rsi, w_macd, w_vwap, w_ema, w_vol, w_uw, w_news, w_10y
        )
        comp["UW_unusual"] = uw_unusual
        comp["UW_bias"] = uw_dir if uw_unusual == "YES" and uw_dir != "Neutral" else comp["UW_bias"]

        # Institutional filter
        if confidence < institutional_min:
            inst_signal = "WAIT"
        else:
            inst_signal = signal

        rows.append({
            "Ticker": t,
            "Confidence": confidence,
            "Direction": direction,
            "Signal": inst_signal,     # filtered output
            "UW_Unusual": comp["UW_unusual"],
            "UW_Bias": comp["UW_bias"],
            "Gamma_bias": comp["Gamma_bias"],
            "RSI": fmt_num(comp["RSI"], 1),
            "MACD_hist": fmt_num(comp["MACD_hist"], 4),
            "VWAP": fmt_num(comp["VWAP"], 2),
            "EMA_stack": comp["EMA_stack"] if comp["EMA_stack"] else "N/A",
            "Vol_ratio": fmt_num(comp["Vol_ratio"], 2),
            "News_sent": comp["News_sent"],
            "10Y": comp["10Y"],
        })

        endpoint_notes["eodhd_intraday"].append((t, s_bars))
        endpoint_notes["eodhd_news"].append((t, s_news))
        endpoint_notes["uw_options_volume"].append((t, s_uw))
        endpoint_notes["uw_flow_alerts"].append((t, s_alerts))

    df_out = pd.DataFrame(rows)

    st.dataframe(df_out, use_container_width=True, hide_index=True)

    # Alerts section
    st.divider()
    st.subheader(f"Institutional Alerts (≥{institutional_min} only)")
    inst = df_out[df_out["Confidence"] >= institutional_min]
    inst = inst[inst["Signal"].isin(["BUY CALLS", "BUY PUTS"])]
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        for _, r in inst.iterrows():
            st.success(f"{r['Ticker']}: {r['Signal']} — {r['Direction']} — Confidence {r['Confidence']} (UW unusual: {r['UW_Unusual']}, UW bias: {r['UW_Bias']})")

    # UW flow alerts panel
    st.divider()
    st.subheader("Unusual Flow Alerts (UW API)")
    if not UW_TOKEN:
        st.warning("UW_TOKEN missing in Secrets.")
    elif not UW_FLOW_ALERTS_URL:
        st.warning("UW_FLOW_ALERTS_URL missing in Secrets.")
    else:
        # show endpoint health summary
        bad = [x for x in endpoint_notes["uw_flow_alerts"] if x[1] not in ["ok", "empty"]]
        if bad:
            # Common failure = 404 due to plan/route mismatch
            st.error(
                f"UW flow alerts failing for some tickers (example: {bad[0][0]} => {bad[0][1]}).\n\n"
                f"URL being used (from Secrets): {UW_FLOW_ALERTS_URL}\n\n"
                "If you see http_404: this is almost always an API route/plan access issue (not your code)."
            )
        else:
            st.success("UW flow-alerts endpoint responded (ok/empty).")

        st.caption("This panel won’t crash your app. It will show RED if UW blocks the endpoint.")
        st.code(
            "Tip: The working URL MUST exactly match the 'curl --url ...' line in UW docs.\n"
            "If UW returns 404, your token may not have access to that endpoint, or the URL is wrong.\n"
            "Your code uses UW_FLOW_ALERTS_URL EXACTLY as typed in Secrets."
        )

    # EODHD News panel (clickable)
    st.divider()
    st.subheader(f"News — last {int(news_lookback)} minutes (EODHD)")
    any_news = False
    for t in tickers:
        sym = as_eodhd_symbol(t)
        items, st_news = eodhd_news(sym, lookback_minutes=int(news_lookback), limit=15)
        if items:
            any_news = True
            for it in items[:8]:
                title = it.get("title", "(no title)")
                url = it.get("link") or it.get("url") or ""
                dt = it.get("date") or ""
                if url:
                    st.markdown(f"- **{t}** — {dt} — [{title}]({url})")
                else:
                    st.markdown(f"- **{t}** — {dt} — {title}")
    if not any_news:
        st.info("No news in this lookback window (or EODHD returned none).")

    # Debug / endpoint status
    st.divider()
    with st.expander("Debug (endpoint status per ticker)"):
        st.write("EODHD intraday:", endpoint_notes["eodhd_intraday"])
        st.write("EODHD news:", endpoint_notes["eodhd_news"])
        st.write("UW options-volume:", endpoint_notes["uw_options_volume"])
        st.write("UW flow-alerts:", endpoint_notes["uw_flow_alerts"])
        st.write("10Y yield source:", teny_status)
