import os
import math
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Flow + News + Live Score", layout="wide")

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "AMD", "NVDA"]

# Your embedded Unusual Whales screener link
UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF&limit=250&max_dte=7"
    "&min_diff=-0.1&min_oi_change_perc=-10&min_premium=500000"
    "&min_volume_oi_ratio=1&min_volume_ticker_vol_ratio=0.03"
    "&order=premium&order_direction=desc&watchlist_name=GPT%20Filter%20"
)

# Secrets (Streamlit -> Manage app -> Settings -> Secrets)
EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")).strip()
UW_BEARER = st.secrets.get("UW_BEARER", os.getenv("UW_BEARER", "")).strip()
UW_FLOW_ALERTS_URL = st.secrets.get("UW_FLOW_ALERTS_URL", os.getenv("UW_FLOW_ALERTS_URL", "")).strip()

# Defaults if user leaves it blank
if not UW_FLOW_ALERTS_URL:
    UW_FLOW_ALERTS_URL = "https://api.unusualwhales.com/api/flow-alerts"

# =========================================================
# SMALL UTILS
# =========================================================
def utc_now():
    return datetime.now(timezone.utc)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# =========================================================
# TECHNICAL INDICATORS (pandas)
# =========================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss.replace(0, math.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    sig = ema(macd_line, signal)
    hist = macd_line - sig
    return hist.fillna(0)

# =========================================================
# EODHD (PRICE + NEWS)
# =========================================================
@st.cache_data(ttl=20)
def eodhd_intraday(symbol: str, minutes: int, api_key: str) -> pd.DataFrame:
    """
    Pull intraday bars from EODHD. For ETFs/stocks, EODHD uses exchanges.
    This endpoint works well for live-ish scoring if you keep it modest.
    """
    if not api_key:
        return pd.DataFrame()

    # 1-minute bars (you can change interval if needed)
    url = "https://eodhd.com/api/intraday/{symbol}.US"
    params = {
        "api_token": api_key,
        "fmt": "json",
        "interval": "1m",
    }

    try:
        r = requests.get(url.format(symbol=symbol), params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        if df.empty:
            return df

        # standardize columns
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime")
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def eodhd_news(symbol: str, lookback_minutes: int, api_key: str, limit: int = 20) -> pd.DataFrame:
    """
    EODHD News endpoint. Stable alternative to Polygon.
    """
    if not api_key:
        return pd.DataFrame()

    # EODHD news endpoint
    url = "https://eodhd.com/api/news"
    since = (utc_now() - timedelta(minutes=lookback_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "api_token": api_key,
        "fmt": "json",
        "s": f"{symbol}.US",
        "from": since,
        "limit": limit,
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        if df.empty:
            return df

        # normalize
        # common fields: date, title, source, link, content
        if "date" in df.columns:
            df["published_utc"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        else:
            df["published_utc"] = pd.NaT

        df["title"] = df.get("title", "")
        df["source"] = df.get("source", "")
        df["url"] = df.get("link", "")
        df["ticker"] = symbol
        return df[["ticker", "published_utc", "title", "source", "url"]].sort_values("published_utc", ascending=False)
    except Exception:
        return pd.DataFrame()

# =========================================================
# VERY SIMPLE NEWS SENTIMENT (no extra libs = stable)
# =========================================================
POS_WORDS = {
    "beats","beat","surge","surges","rally","rallies","upgrade","upgrades","strong",
    "record","bull","bullish","growth","positive","outperform","profit","profits",
    "win","wins","raises","raise","guidance up","accelerates","higher"
}
NEG_WORDS = {
    "miss","misses","plunge","plunges","drop","drops","downgrade","downgrades","weak",
    "warning","bear","bearish","lawsuit","fraud","negative","underperform","loss","losses",
    "cuts","cut","guidance down","lower","halt","investigation"
}

def headline_sentiment_score(title: str) -> float:
    t = (title or "").lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    raw = pos - neg  # could be negative
    return raw

def aggregate_news_sentiment(news_df: pd.DataFrame) -> float:
    """
    Returns a normalized sentiment in [-1, +1]
    """
    if news_df is None or news_df.empty:
        return 0.0
    scores = [headline_sentiment_score(x) for x in news_df["title"].fillna("")]
    if not scores:
        return 0.0
    s = sum(scores)
    # squash
    return clamp(s / 5.0, -1.0, 1.0)

# =========================================================
# UNUSUAL WHALES FLOW ALERTS (LIST ENDPOINT)
# =========================================================
@st.cache_data(ttl=10)
def uw_recent_flow_alerts(symbols: list[str], minutes: int, bearer: str, list_url: str) -> pd.DataFrame:
    """
    Pull recent flow alerts from a LIST endpoint.
    The exact list endpoint can vary by UW plan/version.
    You set it in secrets as UW_FLOW_ALERTS_URL.

    We filter locally for your tickers + time window.
    """
    if not bearer or not list_url:
        return pd.DataFrame()

    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {bearer}",
    }

    # Try some common params. If UW ignores them, we still filter locally.
    params = {
        "limit": 200,
    }

    try:
        r = requests.get(list_url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        # Common shapes: {"data":[...]} or just [...]
        rows = data.get("data", data) if isinstance(data, dict) else data
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Try to find timestamp field
        # often "executed_at" or "created_at"
        ts_col = None
        for c in ["executed_at", "created_at", "timestamp", "time"]:
            if c in df.columns:
                ts_col = c
                break

        if ts_col:
            df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        else:
            df["ts"] = pd.NaT

        # Try to find underlying symbol field
        sym_col = None
        for c in ["underlying_symbol", "symbol", "ticker", "underlying"]:
            if c in df.columns:
                sym_col = c
                break

        if not sym_col:
            # can't filter without a symbol column
            return pd.DataFrame()

        df["sym"] = df[sym_col].astype(str).str.upper()

        cutoff = utc_now() - timedelta(minutes=minutes)
        df = df[df["sym"].isin([s.upper() for s in symbols])]
        if df["ts"].notna().any():
            df = df[df["ts"] >= cutoff]

        # Columns we care about (keep flexible if missing)
        keep = []
        for c in ["sym", "ts", "option_type", "premium", "size", "volume", "tags", "id", "option_chain_id"]:
            if c in df.columns:
                keep.append(c)

        if not keep:
            return pd.DataFrame()

        df = df[keep].sort_values("ts", ascending=False, na_position="last")
        return df
    except Exception:
        return pd.DataFrame()

# =========================================================
# SCORING (0–100)
# =========================================================
def compute_score(symbol: str, bars: pd.DataFrame, news_sent: float, flow_df: pd.DataFrame,
                  w_rsi: float, w_macd: float, w_vol: float, w_news: float, w_flow: float):
    """
    Returns:
      score_0_100, bias_text, signal_text, unusual_alert_bool, details_dict
    """
    if bars is None or bars.empty or "close" not in bars.columns:
        return 0, "Neutral", "WAIT", False, {"reason": "No price data"}

    close = bars["close"].astype(float)
    vol = bars.get("volume", pd.Series([0]*len(bars))).astype(float)

    rsi_val = float(rsi(close, 14).iloc[-1])
    macd_val = float(macd_hist(close).iloc[-1])

    # Volume spike: current vol vs last 30 avg
    if len(vol) >= 35:
        vol_avg = float(vol.iloc[-31:-1].mean())
    else:
        vol_avg = float(vol.mean()) if len(vol) else 0.0
    vol_now = float(vol.iloc[-1]) if len(vol) else 0.0
    vol_ratio = (vol_now / vol_avg) if vol_avg > 0 else 1.0
    vol_spike = clamp((vol_ratio - 1.0), -1.0, 3.0)  # -1..3

    # Normalize components into [-1, +1]
    # RSI: 50 neutral; >50 bullish; <50 bearish
    rsi_norm = clamp((rsi_val - 50.0) / 25.0, -1.0, 1.0)

    # MACD hist: scale with tanh-like squash
    macd_norm = clamp(macd_val / (abs(macd_val) + 0.02), -1.0, 1.0) if macd_val != 0 else 0.0

    # Volume spike: map roughly to [-1..+1]
    vol_norm = clamp(vol_spike / 2.0, -1.0, 1.0)

    # News already [-1..+1]
    news_norm = clamp(news_sent, -1.0, 1.0)

    # Flow: if we have any flow alerts for symbol in window -> bullish/bearish from tags if present
    flow_norm = 0.0
    unusual_alert = False
    if flow_df is not None and not flow_df.empty:
        sym_rows = flow_df[flow_df.get("sym", "") == symbol.upper()]
        if not sym_rows.empty:
            unusual_alert = True
            # infer direction from tags if available
            if "tags" in sym_rows.columns:
                tags = str(sym_rows.iloc[0]["tags"]).lower()
                if "bull" in tags:
                    flow_norm = 1.0
                elif "bear" in tags:
                    flow_norm = -1.0
                else:
                    flow_norm = 0.3  # activity but unclear
            else:
                flow_norm = 0.3

    # Weighted blend -> [-1..+1]
    w_sum = max(0.0001, (w_rsi + w_macd + w_vol + w_news + w_flow))
    blend = (
        rsi_norm * w_rsi +
        macd_norm * w_macd +
        vol_norm * w_vol +
        news_norm * w_news +
        flow_norm * w_flow
    ) / w_sum

    # Convert to 0..100
    score = int(round((blend + 1.0) * 50.0))
    score = clamp(score, 0, 100)

    # Bias
    if score >= 60:
        bias = "Bullish"
    elif score <= 40:
        bias = "Bearish"
    else:
        bias = "Neutral"

    # Signal logic
    # (simple but stable): buy if bullish and RSI rising and MACD positive, sell if bearish and RSI falling and MACD negative
    rsi_prev = float(rsi(close, 14).iloc[-2]) if len(close) >= 2 else rsi_val
    rsi_up = (rsi_val > rsi_prev)

    signal = "WAIT"
    if score >= 70 and rsi_up and macd_val > 0:
        signal = "BUY"
    elif score <= 30 and (not rsi_up) and macd_val < 0:
        signal = "SELL"

    details = {
        "RSI": round(rsi_val, 2),
        "MACD_hist": round(macd_val, 4),
        "Vol_ratio": round(vol_ratio, 2),
        "News_sent": round(news_norm, 2),
        "Flow": "YES" if unusual_alert else "NO"
    }
    return score, bias, signal, unusual_alert, details

# =========================================================
# UI
# =========================================================
st.title("📈 Option Flow (Unusual Whales) + 🗞️ News (EODHD) + Live Score/Signals")

with st.sidebar:
    st.header("Settings")

    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY", "QQQ", "DIA", "IWM"])
    news_minutes = st.number_input("News lookback (minutes)", min_value=1, max_value=240, value=60, step=1)
    price_minutes = st.number_input("Price window (minutes)", min_value=30, max_value=600, value=240, step=10)

    st.divider()
    st.subheader("Refresh")
    refresh_seconds = st.slider("Auto-refresh (seconds)", 5, 120, 30)

    st.divider()
    st.subheader("Keys status")
    st.write("EODHD:", "✅" if EODHD_API_KEY else "❌ (missing EODHD_API_KEY)")
    st.write("UW Bearer:", "✅" if UW_BEARER else "❌ (missing UW_BEARER)")
    st.write("UW Alerts URL:", "✅" if UW_FLOW_ALERTS_URL else "❌ (missing UW_FLOW_ALERTS_URL)")

    st.divider()
    st.subheader("Scoring weights (total doesn’t have to = 1)")
    w_rsi = st.slider("RSI weight", 0.0, 1.0, 0.25, 0.01)
    w_macd = st.slider("MACD weight", 0.0, 1.0, 0.20, 0.01)
    w_vol = st.slider("Volume spike weight", 0.0, 1.0, 0.20, 0.01)
    w_news = st.slider("News sentiment weight", 0.0, 1.0, 0.20, 0.01)
    w_flow = st.slider("Unusual flow weight", 0.0, 1.0, 0.15, 0.01)

# Auto refresh
st_autorefresh(interval=int(refresh_seconds * 1000), key="refresh")

if not tickers:
    st.info("Pick at least 1 ticker in the sidebar.")
    st.stop()

# Layout
col1, col2 = st.columns([1.35, 1])

with col1:
    st.subheader("Unusual Whales — your screener")
    st.caption("This is embedded. For true flow alerts, we also pull UW API (right panel).")
    st.components.v1.iframe(UW_SCREENER_URL, height=900, scrolling=True)

with col2:
    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW flow)")
    st.write(f"Last update (UTC): **{utc_now().strftime('%Y-%m-%d %H:%M:%S')}**")

    # 1) Pull UW flow alerts once per refresh
    flow_df = uw_recent_flow_alerts(tickers, minutes=15, bearer=UW_BEARER, list_url=UW_FLOW_ALERTS_URL)

    # 2) Build scores per ticker
    rows = []
    alerts = []

    for t in tickers:
        bars = eodhd_intraday(t, minutes=int(price_minutes), api_key=EODHD_API_KEY)
        news_df = eodhd_news(t, lookback_minutes=int(news_minutes), api_key=EODHD_API_KEY, limit=25)
        sent = aggregate_news_sentiment(news_df)

        score, bias, signal, unusual_alert, details = compute_score(
            t, bars, sent, flow_df,
            w_rsi=w_rsi, w_macd=w_macd, w_vol=w_vol, w_news=w_news, w_flow=w_flow
        )

        rows.append({
            "Ticker": t,
            "Score": score,
            "Bias": bias,
            "Signal": signal,
            "Unusual Alert": "YES" if unusual_alert else "NO",
            "RSI": details.get("RSI"),
            "MACD_hist": details.get("MACD_hist"),
            "Vol_ratio": details.get("Vol_ratio"),
            "News_sent": details.get("News_sent"),
        })

        if signal in ("BUY", "SELL") or unusual_alert:
            alerts.append(f"{t}: {signal} | Unusual={('YES' if unusual_alert else 'NO')} | Score={score}")

    score_df = pd.DataFrame(rows).sort_values(["Score"], ascending=False)

    st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Alerts (simple)")
    if alerts:
        for a in alerts[:20]:
            st.warning(a)
    else:
        st.info("No BUY/SELL or unusual activity spikes right now.")

    st.divider()
    st.subheader(f"News — last {news_minutes} minutes (EODHD)")
    combined_news = []
    for t in tickers:
        nd = eodhd_news(t, lookback_minutes=int(news_minutes), api_key=EODHD_API_KEY, limit=10)
        if nd is not None and not nd.empty:
            combined_news.append(nd)

    if not combined_news:
        st.info("No news in the last window (or EODHD key missing).")
    else:
        news_all = pd.concat(combined_news, ignore_index=True).sort_values("published_utc", ascending=False)
        st.dataframe(news_all, use_container_width=True, hide_index=True)

        st.caption("Clickable links:")
        for _, r in news_all.head(25).iterrows():
            st.markdown(f"- **{r['ticker']}** — [{r['title']}]({r['url']})")
