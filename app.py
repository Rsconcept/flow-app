# app.py
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ============================================================
# CONFIG / SECRETS
# ============================================================
st.set_page_config(page_title="Institutional Options Signals (5m)", layout="wide")

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "AMD", "NVDA"]

EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")).strip()
UW_BEARER_TOKEN = st.secrets.get("UW_BEARER_TOKEN", os.getenv("UW_BEARER_TOKEN", "")).strip()

# If you *really* want to embed the UnusualWhales web page, set this.
# If blank, we just show API-driven signals.
UW_SCREENER_URL = st.secrets.get("UW_SCREENER_URL", os.getenv("UW_SCREENER_URL", "")).strip()

UTC = timezone.utc


# ============================================================
# SMALL UTILS
# ============================================================
def utc_now():
    return datetime.now(UTC)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def zscore_last(series: pd.Series, window: int = 50) -> float:
    s = series.dropna().tail(window)
    if len(s) < 10:
        return 0.0
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((s.iloc[-1] - mu) / sd)


# ============================================================
# INDICATORS (pure pandas)
# ============================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    # Session VWAP approximation over the loaded bars
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    v = df["volume"].replace(0, np.nan).fillna(0.0)
    cum_pv = (tp * v).cumsum()
    cum_v = v.cumsum().replace(0, np.nan)
    return (cum_pv / cum_v).fillna(method="bfill")


def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0):
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = mid + mult * sd
    lower = mid - mult * sd
    return mid, upper, lower


def supertrend(df: pd.DataFrame, length: int = 10, mult: float = 3.0) -> pd.Series:
    # Simple Supertrend implementation
    hl2 = (df["high"] + df["low"]) / 2.0
    _atr = atr(df, length).fillna(method="bfill")
    upperband = hl2 + mult * _atr
    lowerband = hl2 - mult * _atr

    st_dir = pd.Series(index=df.index, dtype=float)
    st_line = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if i == 0:
            st_dir.iat[i] = 1
            st_line.iat[i] = lowerband.iat[i]
            continue

        prev_line = st_line.iat[i - 1]
        prev_dir = st_dir.iat[i - 1]
        c = df["close"].iat[i]

        # Adjust bands
        if upperband.iat[i] < upperband.iat[i - 1] or df["close"].iat[i - 1] > upperband.iat[i - 1]:
            ub = upperband.iat[i]
        else:
            ub = upperband.iat[i - 1]

        if lowerband.iat[i] > lowerband.iat[i - 1] or df["close"].iat[i - 1] < lowerband.iat[i - 1]:
            lb = lowerband.iat[i]
        else:
            lb = lowerband.iat[i - 1]

        # Direction flip?
        if prev_dir == 1:
            if c < lb:
                st_dir.iat[i] = -1
                st_line.iat[i] = ub
            else:
                st_dir.iat[i] = 1
                st_line.iat[i] = lb
        else:
            if c > ub:
                st_dir.iat[i] = 1
                st_line.iat[i] = lb
            else:
                st_dir.iat[i] = -1
                st_line.iat[i] = ub

        upperband.iat[i] = ub
        lowerband.iat[i] = lb

    return st_dir.fillna(method="bfill").fillna(1)


# ============================================================
# DATA SOURCES
# ============================================================
@st.cache_data(ttl=20)
def eodhd_intraday(symbol: str, interval: str, bars: int = 300):
    """
    EODHD intraday:
    https://eodhd.com/api/intraday/{symbol}?api_token=...&interval=5m&fmt=json
    """
    if not EODHD_API_KEY:
        raise RuntimeError("Missing EODHD_API_KEY in Streamlit secrets.")

    url = f"https://eodhd.com/api/intraday/{symbol}"
    params = {"api_token": EODHD_API_KEY, "interval": interval, "fmt": "json"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # Typical EODHD fields: datetime, open, high, low, close, volume
    # Normalize:
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
        df = df.set_index("datetime")
    else:
        # fallback:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"]).tail(bars)
    return df


@st.cache_data(ttl=60)
def eodhd_news(symbol: str, minutes: int, limit: int = 30):
    """
    EODHD news:
    https://eodhd.com/api/news?s={symbol}&api_token=...&from=YYYY-MM-DD&to=YYYY-MM-DD&limit=...
    """
    if not EODHD_API_KEY:
        return []

    now = utc_now().date()
    frm = (utc_now() - timedelta(days=3)).date()  # buffer
    url = "https://eodhd.com/api/news"
    params = {
        "s": symbol,
        "api_token": EODHD_API_KEY,
        "from": frm.isoformat(),
        "to": now.isoformat(),
        "limit": limit,
        "fmt": "json",
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code == 429:
        return []
    r.raise_for_status()
    items = r.json()
    if not isinstance(items, list):
        return []

    cutoff = utc_now() - timedelta(minutes=int(minutes))
    out = []
    for it in items:
        dt = it.get("date") or it.get("datetime") or it.get("published_at")
        dtp = pd.to_datetime(dt, utc=True, errors="coerce")
        if pd.isna(dtp):
            continue
        if dtp >= cutoff:
            out.append(it)
    return out


@st.cache_data(ttl=20)
def uw_options_volume(ticker: str):
    """
    Unusual Whales:
    GET https://api.unusualwhales.com/api/stock/{ticker}/options-volume
    Authorization: Bearer <token>
    """
    if not UW_BEARER_TOKEN:
        raise RuntimeError("Missing UW_BEARER_TOKEN in Streamlit secrets.")

    url = f"https://api.unusualwhales.com/api/stock/{ticker}/options-volume"
    headers = {"Accept": "application/json, text/plain", "Authorization": f"Bearer {UW_BEARER_TOKEN}"}
    r = requests.get(url, headers=headers, timeout=20)

    if r.status_code == 429:
        # UW throttling – return empty; app will stay stable
        return None

    r.raise_for_status()
    j = r.json()
    data = j.get("data", [])
    if not data:
        return None
    # most recent is typically first, but we’ll pick max date if present
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=False)
    return df.iloc[0].to_dict()


# ============================================================
# NEWS SENTIMENT (simple keyword model; stable / no extra deps)
# ============================================================
POS_WORDS = {
    "beat", "beats", "surge", "soar", "soars", "record", "upgrade", "upgrades", "bullish",
    "strong", "growth", "profit", "profits", "raises", "raise", "guidance up", "buyback",
    "partnership", "wins", "win", "approval",
}
NEG_WORDS = {
    "miss", "misses", "plunge", "plunges", "downgrade", "downgrades", "bearish",
    "weak", "lawsuit", "investigation", "fraud", "cut", "cuts", "guidance down",
    "halt", "halts", "recall", "default", "bankrupt", "bankruptcy",
}


def headline_sentiment_score(items: list) -> float:
    """
    Returns [-1, +1] sentiment from headlines.
    """
    if not items:
        return 0.0
    score = 0.0
    n = 0
    for it in items:
        title = (it.get("title") or it.get("headline") or "").lower()
        if not title:
            continue
        n += 1
        pos = sum(1 for w in POS_WORDS if w in title)
        neg = sum(1 for w in NEG_WORDS if w in title)
        score += (pos - neg)
    if n == 0:
        return 0.0
    # squash
    raw = score / max(3.0, n)
    return float(clamp(raw, -1.0, 1.0))


# ============================================================
# SCORING MODEL (CALLS / PUTS ONLY)
# ============================================================
def compute_features(df: pd.DataFrame):
    """
    df index is datetime utc; columns: open/high/low/close/volume
    """
    out = {}
    if df.empty or len(df) < 60:
        return None

    close = df["close"]
    vol = df["volume"].fillna(0)

    # Core indicators
    out["vwap"] = vwap(df)
    out["ema9"] = ema(close, 9)
    out["ema20"] = ema(close, 20)
    out["ema50"] = ema(close, 50)
    out["rsi"] = rsi(close, 14)
    _, _, hist = macd(close, 12, 26, 9)
    out["macd_hist"] = hist
    mid, upper, lower = bollinger(close, 20, 2.0)
    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["atr"] = atr(df, 14)
    out["supertrend_dir"] = supertrend(df, 10, 3.0)

    # Derived “signals”
    last_close = float(close.iloc[-1])
    last_vwap = float(out["vwap"].iloc[-1])
    last_rsi = float(out["rsi"].iloc[-1])
    last_macd_hist = float(out["macd_hist"].iloc[-1])

    # VWAP bias
    out["above_vwap"] = 1.0 if last_close > last_vwap else 0.0

    # EMA stack bias
    e9 = float(out["ema9"].iloc[-1])
    e20 = float(out["ema20"].iloc[-1])
    e50 = float(out["ema50"].iloc[-1])
    out["ema_bull_stack"] = 1.0 if (e9 > e20 > e50) else 0.0
    out["ema_bear_stack"] = 1.0 if (e9 < e20 < e50) else 0.0

    # RSI normalizations
    out["rsi_bull"] = clamp((last_rsi - 50.0) / 25.0, 0.0, 1.0)   # 50->0, 75->1
    out["rsi_bear"] = clamp((50.0 - last_rsi) / 25.0, 0.0, 1.0)   # 50->0, 25->1

    # MACD hist normalization (use zscore)
    z = zscore_last(out["macd_hist"], 60)
    out["macd_bull"] = clamp((z / 2.0), 0.0, 1.0)
    out["macd_bear"] = clamp((-z / 2.0), 0.0, 1.0)

    # Bollinger position (mean reversion + trend)
    bb_u = float(out["bb_upper"].iloc[-1])
    bb_l = float(out["bb_lower"].iloc[-1])
    if bb_u > bb_l:
        pos = (last_close - bb_l) / (bb_u - bb_l)
    else:
        pos = 0.5
    # When near upper: bullish continuation; near lower: bearish continuation
    out["bb_bull"] = clamp((pos - 0.5) / 0.5, 0.0, 1.0)
    out["bb_bear"] = clamp((0.5 - pos) / 0.5, 0.0, 1.0)

    # Volume spike
    vol_z = zscore_last(vol, 80)
    out["vol_spike"] = clamp(vol_z / 3.0, 0.0, 1.0)  # only positive spikes matter

    # Supertrend dir
    st_dir = float(out["supertrend_dir"].iloc[-1])
    out["st_bull"] = 1.0 if st_dir > 0 else 0.0
    out["st_bear"] = 1.0 if st_dir < 0 else 0.0

    return out


def score_calls_puts(features: dict, news_sent: float, uw: dict, weights: dict):
    """
    Returns:
    calls_score (0..100), puts_score (0..100), unusual_flag(bool), debug dict
    """
    if features is None:
        return 0, 0, False, {}

    # ----- TECH COMPONENTS (0..1)
    bull = 0.0
    bear = 0.0

    # VWAP
    bull += weights["w_vwap"] * features["above_vwap"]
    bear += weights["w_vwap"] * (1.0 - features["above_vwap"])

    # EMA stack
    bull += weights["w_ema"] * features["ema_bull_stack"]
    bear += weights["w_ema"] * features["ema_bear_stack"]

    # RSI
    bull += weights["w_rsi"] * features["rsi_bull"]
    bear += weights["w_rsi"] * features["rsi_bear"]

    # MACD
    bull += weights["w_macd"] * features["macd_bull"]
    bear += weights["w_macd"] * features["macd_bear"]

    # Bollinger
    bull += weights["w_bb"] * features["bb_bull"]
    bear += weights["w_bb"] * features["bb_bear"]

    # Volume confirmation
    bull += weights["w_vol"] * features["vol_spike"]
    bear += weights["w_vol"] * features["vol_spike"]

    # Supertrend
    bull += weights["w_st"] * features["st_bull"]
    bear += weights["w_st"] * features["st_bear"]

    # ----- NEWS COMPONENT (-1..+1 -> bull/bear)
    # map sentiment to bull/bear boosts
    news_bull = clamp((news_sent + 1.0) / 2.0, 0.0, 1.0)
    news_bear = clamp((1.0 - (news_sent + 1.0) / 2.0), 0.0, 1.0)
    bull += weights["w_news"] * news_bull
    bear += weights["w_news"] * news_bear

    # ----- UNUSUAL WHALES OPTIONS-VOLUME COMPONENT
    unusual_flag = False
    uw_bull = 0.5
    uw_bear = 0.5

    if uw:
        call_vol = safe_float(uw.get("call_volume"), 0.0)
        put_vol = safe_float(uw.get("put_volume"), 0.0)
        call_prem = safe_float(uw.get("call_premium"), 0.0)
        put_prem = safe_float(uw.get("put_premium"), 0.0)

        avg3_call = safe_float(uw.get("avg_3_day_call_volume"), np.nan)
        avg3_put = safe_float(uw.get("avg_3_day_put_volume"), np.nan)

        # “unusual” if current vol significantly above short-term avg
        call_spike = 0.0
        put_spike = 0.0
        if not np.isnan(avg3_call) and avg3_call > 0:
            call_spike = clamp((call_vol / avg3_call - 1.0) / 1.5, 0.0, 1.0)  # 1x->0, 2.5x->1
        if not np.isnan(avg3_put) and avg3_put > 0:
            put_spike = clamp((put_vol / avg3_put - 1.0) / 1.5, 0.0, 1.0)

        # Bias by premium imbalance (more robust than raw volume)
        prem_total = max(1.0, call_prem + put_prem)
        call_share = call_prem / prem_total
        put_share = put_prem / prem_total

        uw_bull = clamp(call_share, 0.0, 1.0)
        uw_bear = clamp(put_share, 0.0, 1.0)

        # Unusual flag if either side spikes hard
        unusual_flag = (call_spike >= 0.65) or (put_spike >= 0.65)

        # Add to bull/bear
        bull += weights["w_uw"] * uw_bull
        bear += weights["w_uw"] * uw_bear
    else:
        # If UW unavailable, keep stable (no crash)
        bull += weights["w_uw"] * 0.5
        bear += weights["w_uw"] * 0.5

    # Normalize to 0..100 confidence
    total = max(1e-9, bull + bear)
    bull_conf = bull / total
    bear_conf = bear / total

    calls_score = int(round(bull_conf * 100))
    puts_score = int(round(bear_conf * 100))

    debug = {
        "bull_raw": bull,
        "bear_raw": bear,
        "news_sent": news_sent,
        "uw_bull_share": uw_bull,
        "uw_bear_share": uw_bear,
    }
    return calls_score, puts_score, unusual_flag, debug


# ============================================================
# UI
# ============================================================
st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")

with st.sidebar:
    st.header("Settings")

    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY", "QQQ", "DIA", "IWM"])
    price_lookback_minutes = st.number_input("Price lookback (minutes)", min_value=60, max_value=720, value=240, step=30)
    news_lookback_minutes = st.number_input("News lookback (minutes)", min_value=5, max_value=180, value=60, step=5)
    refresh_seconds = st.slider("Auto-refresh (seconds)", min_value=10, max_value=120, value=30, step=5)

    st.divider()
    st.subheader("Institutional mode")
    inst_threshold = st.slider("Signals only if confidence ≥", min_value=50, max_value=95, value=75, step=1)

    st.divider()
    st.subheader("Keys status")
    st.write("EODHD:", "✅" if EODHD_API_KEY else "❌ (missing)")
    st.write("UW Bearer:", "✅" if UW_BEARER_TOKEN else "❌ (missing)")

    st.divider()
    st.subheader("Weights (total can be anything)")
    w_vwap = st.slider("VWAP", 0.0, 0.50, 0.15, 0.01)
    w_ema = st.slider("EMA stack (9/20/50)", 0.0, 0.50, 0.18, 0.01)
    w_rsi = st.slider("RSI", 0.0, 0.50, 0.12, 0.01)
    w_macd = st.slider("MACD hist", 0.0, 0.50, 0.12, 0.01)
    w_bb = st.slider("Bollinger", 0.0, 0.50, 0.08, 0.01)
    w_vol = st.slider("Volume spike", 0.0, 0.50, 0.10, 0.01)
    w_st = st.slider("Supertrend", 0.0, 0.50, 0.10, 0.01)
    w_news = st.slider("News sentiment", 0.0, 0.50, 0.05, 0.01)
    w_uw = st.slider("UnusualWhales options-volume", 0.0, 0.80, 0.25, 0.01)

    weights = {
        "w_vwap": w_vwap,
        "w_ema": w_ema,
        "w_rsi": w_rsi,
        "w_macd": w_macd,
        "w_bb": w_bb,
        "w_vol": w_vol,
        "w_st": w_st,
        "w_news": w_news,
        "w_uw": w_uw,
    }

# Auto-refresh
st_autorefresh(interval=int(refresh_seconds * 1000), key="auto_refresh")

col1, col2 = st.columns([1.2, 1])

# LEFT: UW web embed (optional)
with col1:
    st.subheader("Unusual Whales Screener (web view)")
    if UW_SCREENER_URL:
        st.caption("This is embedded. For true alerts, use UnusualWhales native alerts too.")
        st.components.v1.iframe(UW_SCREENER_URL, height=850, scrolling=True)
    else:
        st.info("No UW_SCREENER_URL set (optional). This app uses UW API for options-volume anyway.")

# RIGHT: Live score/signals
with col2:
    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW options-volume)")
    st.caption(f"Last update (UTC): {utc_now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not tickers:
        st.warning("Pick at least 1 ticker.")
        st.stop()

    # Build results table
    rows = []
    alerts = []

    for t in tickers:
        try:
            # 5m bars; request enough bars
            bars = int(max(120, price_lookback_minutes / 5 + 60))
            df = eodhd_intraday(t, interval="5m", bars=bars)

            feats = compute_features(df)
            news_items = eodhd_news(t, minutes=int(news_lookback_minutes), limit=30)
            news_sent = headline_sentiment_score(news_items)

            uw = None
            uw_err = None
            try:
                uw = uw_options_volume(t)
            except Exception as e:
                uw_err = str(e)

            calls_score, puts_score, unusual_flag, dbg = score_calls_puts(
                feats, news_sent=news_sent, uw=uw, weights=weights
            )

            # Institutional decision: CALLS or PUTS only, else WAIT
            if calls_score >= inst_threshold and calls_score > puts_score:
                signal = "BUY CALLS"
                bias = "Bullish"
                conf = calls_score
            elif puts_score >= inst_threshold and puts_score > calls_score:
                signal = "BUY PUTS"
                bias = "Bearish"
                conf = puts_score
            else:
                signal = "WAIT"
                bias = "Neutral"
                conf = max(calls_score, puts_score)

            # For quick monitoring
            last_rsi = float(feats["rsi"].iloc[-1]) if feats else np.nan
            last_macd_hist = float(feats["macd_hist"].iloc[-1]) if feats else np.nan
            vol_ratio = np.nan
            if uw:
                call_vol = safe_float(uw.get("call_volume"), 0.0)
                put_vol = safe_float(uw.get("put_volume"), 0.0)
                vol_ratio = (call_vol / max(1.0, put_vol)) if put_vol else np.nan

            rows.append(
                {
                    "Ticker": t,
                    "CALLS_score": calls_score,
                    "PUTS_score": puts_score,
                    "Bias": bias,
                    "Signal": signal,
                    "Confidence": conf,
                    "Unusual": "YES" if unusual_flag else "NO",
                    "RSI": round(last_rsi, 2) if not np.isnan(last_rsi) else None,
                    "MACD_hist": round(last_macd_hist, 4) if not np.isnan(last_macd_hist) else None,
                    "Call/Put_vol_ratio": round(float(vol_ratio), 2) if not np.isnan(vol_ratio) else None,
                    "News_sent": round(float(news_sent), 2),
                }
            )

            if signal != "WAIT" or unusual_flag:
                alerts.append(f"**{t}** — {signal} | Conf={conf} | Unusual={('YES' if unusual_flag else 'NO')}")

            if uw_err:
                alerts.append(f"⚠️ **{t}** UW error: {uw_err}")

        except Exception as e:
            rows.append(
                {
                    "Ticker": t,
                    "CALLS_score": None,
                    "PUTS_score": None,
                    "Bias": "Error",
                    "Signal": "ERROR",
                    "Confidence": None,
                    "Unusual": None,
                    "RSI": None,
                    "MACD_hist": None,
                    "Call/Put_vol_ratio": None,
                    "News_sent": None,
                }
            )
            alerts.append(f"❌ **{t}** error: {e}")

    out_df = pd.DataFrame(rows)

    st.dataframe(out_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Alerts (institutional)")
    if alerts:
        for a in alerts[:20]:
            st.success(a)
    else:
        st.info("No signals above threshold right now.")

    st.divider()
    st.subheader(f"News (last {news_lookback_minutes} minutes — EODHD)")
    # show compact news list for the first ticker selected
    t0 = tickers[0]
    items0 = eodhd_news(t0, minutes=int(news_lookback_minutes), limit=30)
    if not items0:
        st.info("No news in window (or rate-limited / no headlines).")
    else:
        # show top 10
        for it in items0[:10]:
            title = it.get("title") or it.get("headline") or "(no title)"
            url = it.get("link") or it.get("url") or ""
            dt = it.get("date") or it.get("datetime") or ""
            if url:
                st.markdown(f"- **{t0}** [{title}]({url})  \n  _{dt}_")
            else:
                st.write(f"- **{t0}** {title} ({dt})")

