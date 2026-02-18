import os
import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Institutional Options Signals (5m)", layout="wide")

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "AMD", "NVDA"]

UW_SCREENER_URL = st.secrets.get(
    "UW_SCREENER_URL",
    "https://unusualwhales.com/options-screener"
)

EODHD_API_KEY = (st.secrets.get("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")) or "").strip()
UW_BEARER = (st.secrets.get("UW_BEARER", os.getenv("UW_BEARER", "")) or "").strip()
UW_FLOW_ALERTS_URL = (st.secrets.get("UW_FLOW_ALERTS_URL", os.getenv("UW_FLOW_ALERTS_URL", "")) or "").strip()

INSTITUTIONAL_MIN_CONF = 75

# UW filters (hard rules)
MIN_PREMIUM = 1_000_000
MAX_DTE_DAYS = 3
REQUIRE_VOL_GT_OI = True
EXCLUDE_ITM = True


# -----------------------------
# UTIL
# -----------------------------
def utc_now():
    return datetime.now(timezone.utc)

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def symbol_to_eodhd(sym: str) -> str:
    return f"{sym}.US"


# -----------------------------
# TECHNICALS
# -----------------------------
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss.replace(0, pd.NA))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50)

def macd_hist(series: pd.Series) -> pd.Series:
    macd_line = ema(series, 12) - ema(series, 26)
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return hist.fillna(0)

def vwap(df: pd.DataFrame) -> pd.Series:
    if not {"high", "low", "close", "volume"}.issubset(df.columns):
        return pd.Series([pd.NA] * len(df), index=df.index)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"].fillna(0)
    cum_pv = pv.cumsum()
    cum_v = df["volume"].fillna(0).cumsum().replace(0, pd.NA)
    out = (cum_pv / cum_v).fillna(method="bfill")
    return out


# -----------------------------
# EODHD: INTRADAY + QUOTE + NEWS
# -----------------------------
@st.cache_data(ttl=20)
def eodhd_intraday(ticker: str, lookback_minutes: int, api_key: str) -> pd.DataFrame:
    """
    Tries 5m bars first; if empty, tries 1m bars.
    Handles different EODHD field names safely.
    """
    if not api_key:
        return pd.DataFrame()

    sym = symbol_to_eodhd(ticker)
    end_ts = utc_now()
    start_ts = end_ts - timedelta(minutes=lookback_minutes)

    def _fetch(interval: str) -> pd.DataFrame:
        url = f"https://eodhd.com/api/intraday/{sym}"
        params = {
            "api_token": api_key,
            "interval": interval,
            "fmt": "json",
            "from": int(start_ts.timestamp()),
            "to": int(end_ts.timestamp()),
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data)

        # timestamp normalization
        # possible fields: datetime, timestamp, date
        dt_col = None
        for c in ["datetime", "timestamp", "date"]:
            if c in df.columns:
                dt_col = c
                break

        if dt_col is None:
            return pd.DataFrame()

        # If timestamp is numeric, convert from seconds
        if dt_col == "timestamp":
            df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
        else:
            df["dt"] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")

        df = df.dropna(subset=["dt"]).sort_values("dt").set_index("dt")

        # normalize numeric fields
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["close"])
        return df

    df5 = _fetch("5m")
    if len(df5) >= 30:
        return df5

    df1 = _fetch("1m")
    return df1


@st.cache_data(ttl=10)
def eodhd_quote(ticker: str, api_key: str) -> dict:
    """
    Real-time-ish quote fallback when intraday bars are missing after-hours.
    """
    if not api_key:
        return {}
    sym = symbol_to_eodhd(ticker)
    url = f"https://eodhd.com/api/real-time/{sym}"
    params = {"api_token": api_key, "fmt": "json"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    return js if isinstance(js, dict) else {}


@st.cache_data(ttl=30)
def eodhd_news(ticker: str, lookback_minutes: int, api_key: str) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()

    sym = symbol_to_eodhd(ticker)
    now = utc_now()
    from_dt = (now - timedelta(days=3)).date().isoformat()
    to_dt = now.date().isoformat()

    url = "https://eodhd.com/api/news"
    params = {
        "api_token": api_key,
        "s": sym,
        "from": from_dt,
        "to": to_dt,
        "limit": 50,
        "fmt": "json",
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["published_utc"] = pd.to_datetime(df.get("date", pd.NaT), utc=True, errors="coerce")

    cutoff = now - timedelta(minutes=lookback_minutes)
    df = df[df["published_utc"].notna() & (df["published_utc"] >= cutoff)].copy()

    df["Ticker"] = ticker
    df["Title"] = df.get("title", "")
    df["Source"] = df.get("source", "")
    df["URL"] = df.get("link", "")

    return df[["Ticker", "published_utc", "Source", "Title", "URL"]].sort_values("published_utc", ascending=False)


def quick_news_sentiment_score(news_df: pd.DataFrame) -> float:
    if news_df is None or news_df.empty:
        return 0.0
    pos_words = ["beats", "surge", "upgrade", "record", "strong", "growth", "rally", "profit"]
    neg_words = ["miss", "plunge", "downgrade", "weak", "lawsuit", "loss", "cut", "crash"]
    score = 0
    for t in news_df["Title"].fillna("").astype(str).tolist():
        lt = t.lower()
        score += sum(1 for w in pos_words if w in lt)
        score -= sum(1 for w in neg_words if w in lt)
    return clamp(score / 8.0, -1.0, 1.0)


# -----------------------------
# UNUSUAL WHALES FLOW ALERTS
# -----------------------------
@st.cache_data(ttl=15)
def uw_flow_alerts_raw(api_url: str, bearer: str, limit: int = 300) -> list[dict]:
    if not api_url or not bearer:
        return []
    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {bearer}",
    }
    params = {"limit": limit}
    r = requests.get(api_url, headers=headers, params=params, timeout=20)
    if r.status_code == 429:
        return []
    r.raise_for_status()
    js = r.json()
    if isinstance(js, dict) and isinstance(js.get("data"), list):
        return js["data"]
    if isinstance(js, list):
        return js
    return []

def dte_days(expiry_str: str) -> int | None:
    try:
        exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        return (exp - utc_now().date()).days
    except Exception:
        return None

def is_itm(option_type: str, strike: float, underlying: float) -> bool:
    ot = (option_type or "").lower()
    if ot == "call":
        return underlying > strike
    if ot == "put":
        return underlying < strike
    return False

def filter_uw_alerts(alerts: list[dict], ticker: str) -> list[dict]:
    out = []
    for a in alerts or []:
        sym = a.get("underlying_symbol") or a.get("ticker") or a.get("symbol")
        if sym != ticker:
            continue

        prem = safe_float(a.get("premium"), 0.0)
        if prem < MIN_PREMIUM:
            continue

        exp = a.get("expiry") or a.get("expiration") or a.get("exp")
        dte = dte_days(exp) if exp else None
        if dte is not None and dte > MAX_DTE_DAYS:
            continue

        if EXCLUDE_ITM:
            ot = a.get("option_type") or a.get("type")
            strike = safe_float(a.get("strike"), 0.0)
            und = safe_float(a.get("underlying_price"), 0.0)
            if und > 0 and strike > 0 and is_itm(ot, strike, und):
                continue

        if REQUIRE_VOL_GT_OI:
            vol = safe_float(a.get("volume"), 0.0)
            oi = safe_float(a.get("open_interest"), 0.0)
            if not (vol > oi):
                continue

        out.append(a)

    return out

def uw_bias_from_alerts(filtered_alerts: list[dict]) -> tuple[str, bool]:
    if not filtered_alerts:
        return ("Neutral", False)

    call_p = 0.0
    put_p = 0.0
    for a in filtered_alerts:
        ot = (a.get("option_type") or "").lower()
        prem = safe_float(a.get("premium"), 0.0)
        if ot == "call":
            call_p += prem
        elif ot == "put":
            put_p += prem

    total = call_p + put_p
    if total <= 0:
        return ("Neutral", True)

    if call_p / total >= 0.60:
        return ("Bullish", True)
    if put_p / total >= 0.60:
        return ("Bearish", True)
    return ("Neutral", True)


# -----------------------------
# SCORING (CALLS/PUTS ONLY)
# -----------------------------
def build_score(df: pd.DataFrame, news_sent: float) -> dict:
    if df is None or df.empty or len(df) < 30:
        return {
            "bull": 50.0,
            "bear": 50.0,
            "rsi": None,
            "macd_hist": None,
            "vwap_bias": None,
            "ema_bias": None,
            "vol_ratio": None,
            "bars": 0,
            "last_bar_utc": None,
            "news_sent": news_sent,
        }

    close = df["close"].astype(float)
    vol = df.get("volume", pd.Series([0]*len(df), index=df.index)).fillna(0).astype(float)

    rsi14 = rsi(close, 14).iloc[-1]
    mh = macd_hist(close).iloc[-1]

    ema9 = ema(close, 9).iloc[-1]
    ema20 = ema(close, 20).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]

    vw = vwap(df).iloc[-1]
    last = close.iloc[-1]

    vol_mean = vol.rolling(20).mean().iloc[-1]
    vol_ratio = (vol.iloc[-1] / vol_mean) if (vol_mean and vol_mean > 0) else 1.0

    # Component scoring → bullishness 0..100
    rsi_bull = 70 if rsi14 <= 30 else (30 if rsi14 >= 70 else 50)
    macd_bull = clamp(50 + (mh * 800), 0, 100)
    vwap_bull = 75 if (pd.notna(vw) and last > vw) else 25

    ema_bull = 50
    if ema9 > ema20 > ema50:
        ema_bull = 75
    elif ema9 < ema20 < ema50:
        ema_bull = 25

    vol_bull = clamp(50 + (vol_ratio - 1.0) * 25, 0, 100)
    news_bull = clamp(50 + news_sent * 35, 0, 100)

    # Institutional weights (fixed default for now)
    bull = (
        0.15 * vwap_bull +
        0.18 * ema_bull +
        0.15 * rsi_bull +
        0.15 * macd_bull +
        0.22 * vol_bull +
        0.15 * news_bull
    )
    bull = clamp(bull, 0, 100)
    bear = 100 - bull

    return {
        "bull": round(bull, 1),
        "bear": round(bear, 1),
        "rsi": round(float(rsi14), 2),
        "macd_hist": round(float(mh), 4),
        "vwap_bias": "Above" if (pd.notna(vw) and last > vw) else "Below",
        "ema_bias": "Bull" if ema_bull >= 60 else ("Bear" if ema_bull <= 40 else "Neutral"),
        "vol_ratio": round(float(vol_ratio), 2),
        "bars": int(len(df)),
        "last_bar_utc": df.index.max().strftime("%Y-%m-%d %H:%M:%S"),
        "news_sent": round(float(news_sent), 2),
    }


# -----------------------------
# UI
# -----------------------------
st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY", "QQQ", "NVDA", "TSLA", "AMD"])
    news_lookback = st.number_input("News lookback (minutes)", min_value=5, max_value=240, value=60, step=5)
    price_lookback = st.number_input("Price lookback (minutes)", min_value=120, max_value=1200, value=420, step=30)

    st.divider()
    st.subheader("Refresh")
    refresh_seconds = st.slider("Auto-refresh (seconds)", 15, 300, 30, 5)

    st.divider()
    st.subheader("Keys status")
    st.write("EODHD:", "✅" if EODHD_API_KEY else "❌")
    st.write("UW Bearer:", "✅" if UW_BEARER else "❌")
    st.write("UW Alerts URL:", "✅" if UW_FLOW_ALERTS_URL else "❌")

st_autorefresh(interval=int(refresh_seconds * 1000), key="autorefresh")

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("After-hours: flow alerts may be quiet. Signals still run off price + headlines.")
    st.components.v1.iframe(UW_SCREENER_URL, height=850, scrolling=True)

with right:
    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW flow)")
    st.write(f"Last update (UTC): **{utc_now().strftime('%Y-%m-%d %H:%M:%S')}**")

    if not tickers:
        st.info("Pick at least 1 ticker.")
        st.stop()

    # UW alerts (may be empty after-hours)
    uw_raw = []
    try:
        uw_raw = uw_flow_alerts_raw(UW_FLOW_ALERTS_URL, UW_BEARER, limit=400)
    except Exception as e:
        st.warning(f"UW flow fetch failed: {e}")

    rows = []
    alerts_out = []
    debug_rows = []

    for t in tickers:
        # PRICE
        df = pd.DataFrame()
        mode = "intraday"
        try:
            df = eodhd_intraday(t, int(price_lookback), EODHD_API_KEY)
        except Exception as e:
            mode = "error"
            df = pd.DataFrame()

        if df.empty:
            # Quote fallback (after-hours safe)
            mode = "quote_fallback"
            q = {}
            try:
                q = eodhd_quote(t, EODHD_API_KEY)
            except Exception:
                q = {}

            # If quote exists, build a tiny fake df so we don't show None everywhere
            last = safe_float(q.get("close") or q.get("price") or q.get("last"), 0.0)
            if last > 0:
                now = utc_now()
                df = pd.DataFrame(
                    {
                        "open": [last]*35,
                        "high": [last]*35,
                        "low":  [last]*35,
                        "close":[last]*35,
                        "volume":[1]*35,
                    },
                    index=pd.date_range(end=now, periods=35, freq="1min", tz="UTC")
                )

        # NEWS
        news_df = pd.DataFrame()
        try:
            news_df = eodhd_news(t, int(news_lookback), EODHD_API_KEY)
        except Exception:
            news_df = pd.DataFrame()

        ns = quick_news_sentiment_score(news_df)

        # TECH SCORE
        s = build_score(df, ns)

        # UW FILTERED
        uw_filtered = filter_uw_alerts(uw_raw, t)
        uw_bias, uw_unusual = uw_bias_from_alerts(uw_filtered)

        bull_adj = s["bull"]
        bear_adj = s["bear"]

        if uw_unusual and uw_bias == "Bullish":
            bull_adj = clamp(bull_adj + 10, 0, 100)
            bear_adj = 100 - bull_adj
        elif uw_unusual and uw_bias == "Bearish":
            bear_adj = clamp(bear_adj + 10, 0, 100)
            bull_adj = 100 - bear_adj

        if bull_adj >= bear_adj:
            direction = "CALLS"
            conf = bull_adj
        else:
            direction = "PUTS"
            conf = bear_adj

        signal = f"BUY {direction}" if conf >= INSTITUTIONAL_MIN_CONF else "WAIT"

        rows.append({
            "Ticker": t,
            "Confidence": round(conf, 1),
            "Direction": direction if signal != "WAIT" else "—",
            "Signal": signal,
            "UW Unusual": "YES" if uw_unusual else "NO",
            "UW Bias": uw_bias,
            "RSI": s["rsi"],
            "MACD_hist": s["macd_hist"],
            "VWAP": s["vwap_bias"],
            "EMA": s["ema_bias"],
            "Vol_ratio": s["vol_ratio"],
        })

        debug_rows.append({
            "Ticker": t,
            "Data mode": mode,
            "Bars": s["bars"],
            "Last bar (UTC)": s["last_bar_utc"],
        })

        if signal != "WAIT":
            alerts_out.append(f"**{t}** → {signal} | Conf={round(conf,1)} | UW={('YES' if uw_unusual else 'NO')} ({uw_bias})")

    score_df = pd.DataFrame(rows).sort_values("Confidence", ascending=False)
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Alerts (institutional)")
    if alerts_out:
        for a in alerts_out:
            st.success(a)
    else:
        st.info("No institutional signals (confidence < 75 for all tickers).")

    st.divider()
    st.subheader("Debug (why indicators might be None)")
    st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(f"News — last {int(news_lookback)} minutes (EODHD)")
    all_news = []
    for t in tickers:
        try:
            ndf = eodhd_news(t, int(news_lookback), EODHD_API_KEY)
            if not ndf.empty:
                all_news.append(ndf)
        except Exception:
            pass

    if all_news:
        nd = pd.concat(all_news, ignore_index=True).sort_values("published_utc", ascending=False)
        st.dataframe(nd, use_container_width=True, hide_index=True)
    else:
        st.info("No news in the last window (or EODHD returned none).")

