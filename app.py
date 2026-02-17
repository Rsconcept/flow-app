import os
import math
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Flow + News + Live Score", layout="wide")

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "AMD", "NVDA"]

UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF&limit=250&max_dte=7"
    "&min_diff=-0.1&min_oi_change_perc=-10&min_premium=500000"
    "&min_volume_oi_ratio=1&min_volume_ticker_vol_ratio=0.03"
    "&order=premium&order_direction=desc&watchlist_name=GPT%20Filter%20"
)

POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY", os.getenv("POLYGON_API_KEY", "")).strip()

POLYGON_BASE = "https://api.polygon.io"
USER_AGENT = {"User-Agent": "streamlit-flow-news/1.0"}


# =============================
# SMALL UTIL
# =============================
def utc_now():
    return datetime.now(timezone.utc)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# =============================
# TECHNICAL INDICATORS (pandas only)
# =============================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, float("nan")))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# =============================
# POLYGON FETCH
# =============================
@st.cache_data(ttl=30, show_spinner=False)
def polygon_minute_bars(ticker: str, minutes_back: int, api_key: str) -> pd.DataFrame:
    """
    Fetch 1-minute aggregates for last `minutes_back` minutes.
    """
    if not api_key:
        return pd.DataFrame()

    end = utc_now()
    start = end - timedelta(minutes=minutes_back)

    # Polygon expects dates; but minute range works with YYYY-MM-DD.
    # We pass a slightly wider date range and rely on returned timestamps.
    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/minute/{start.date()}/{end.date()}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    r = requests.get(url, params=params, headers=USER_AGENT, timeout=20)
    if r.status_code == 429:
        return pd.DataFrame()  # rate-limited
    r.raise_for_status()
    data = r.json() or {}
    results = data.get("results", []) or []
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # Polygon fields: t=ms epoch, o,h,l,c,v
    df["dt"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    # Keep only window
    df = df[df.index >= start]
    return df[["open", "high", "low", "close", "volume"]].copy()


@st.cache_data(ttl=60, show_spinner=False)
def polygon_news(ticker: str, minutes_back: int, api_key: str) -> list[dict]:
    """
    Fetch news for ticker and then filter locally by published_utc within minutes_back.
    """
    if not api_key:
        return []

    url = f"{POLYGON_BASE}/v2/reference/news"
    params = {
        "ticker": ticker,
        "limit": 50,
        "order": "desc",
        "sort": "published_utc",
        "apiKey": api_key,
    }

    r = requests.get(url, params=params, headers=USER_AGENT, timeout=20)
    if r.status_code == 429:
        return []
    r.raise_for_status()
    data = r.json() or {}
    items = data.get("results", []) or []

    cutoff = utc_now() - timedelta(minutes=minutes_back)
    out = []
    for it in items:
        try:
            pu = pd.to_datetime(it.get("published_utc"), utc=True)
            if pu >= cutoff:
                out.append(it)
        except Exception:
            continue
    return out


# =============================
# SIMPLE NEWS SENTIMENT (no extra libs)
# =============================
POS_WORDS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "record", "strong", "upgrade",
    "bull", "bullish", "growth", "wins", "win", "positive", "buy", "rebound", "raises"
}
NEG_WORDS = {
    "miss", "misses", "drop", "drops", "plunge", "plunges", "weak", "downgrade",
    "bear", "bearish", "lawsuit", "cuts", "cut", "negative", "sell", "fraud", "halt"
}


def score_headline_sentiment(headline: str) -> float:
    """
    Returns -1..+1
    """
    if not headline:
        return 0.0
    text = headline.lower()
    pos = sum(1 for w in POS_WORDS if w in text)
    neg = sum(1 for w in NEG_WORDS if w in text)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def news_dataframe(items: list[dict], ticker: str) -> pd.DataFrame:
    rows = []
    for it in items:
        title = it.get("title", "") or ""
        url = it.get("article_url", "") or ""
        source = (it.get("publisher", {}) or {}).get("name", "") or ""
        published = it.get("published_utc", "") or ""
        s = score_headline_sentiment(title)
        rows.append({
            "Ticker": ticker,
            "Published_UTC": published,
            "Sentiment": round(s, 2),
            "Title": title,
            "Source": source,
            "URL": url
        })
    return pd.DataFrame(rows)


# =============================
# SCORE + SIGNALS
# =============================
def compute_features(df: pd.DataFrame) -> dict:
    """
    df: minute bars
    Returns latest indicator values for scoring.
    """
    if df.empty or len(df) < 50:
        return {}

    close = df["close"]
    vol = df["volume"]

    r = rsi(close, 14).iloc[-1]

    macd_line, signal_line, hist = macd(close)
    macd_hist = hist.iloc[-1]

    ema20 = ema(close, 20).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    trend_up = 1 if ema20 > ema50 else 0

    # Volume spike = last 5m avg / prior 30m avg
    last_5 = vol.tail(5).mean()
    prev_30 = vol.tail(35).head(30).mean() if len(vol) >= 35 else vol.mean()
    vol_spike = (last_5 / prev_30) if prev_30 and not math.isnan(prev_30) else 1.0

    # Range spike = last 5m avg range / prior 30m avg range
    rng = (df["high"] - df["low"])
    last_5_rng = rng.tail(5).mean()
    prev_30_rng = rng.tail(35).head(30).mean() if len(rng) >= 35 else rng.mean()
    range_spike = (last_5_rng / prev_30_rng) if prev_30_rng and not math.isnan(prev_30_rng) else 1.0

    return {
        "rsi": float(r),
        "macd_hist": float(macd_hist),
        "trend_up": int(trend_up),
        "vol_spike": float(vol_spike),
        "range_spike": float(range_spike),
        "last_close": float(close.iloc[-1]),
    }


def to_0_1(x, lo, hi):
    if hi == lo:
        return 0.5
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)


def compute_score(features: dict, news_sent: float, weights: dict) -> tuple[int, str, str, str]:
    """
    Returns:
      score 0..100,
      bias ("Bullish"/"Bearish"/"Neutral"),
      signal ("BUY"/"SELL"/"WAIT"),
      unusual_alert ("YES"/"NO")
    """
    if not features:
        return 0, "Neutral", "WAIT", "NO"

    # RSI: prefer 30-70 neutral; bullish if rising from low; bearish if high
    rsi_val = features["rsi"]
    rsi_bull = to_0_1(70 - rsi_val, 0, 40)  # higher when RSI is lower (oversold)
    rsi_bear = to_0_1(rsi_val - 30, 0, 40)  # higher when RSI is higher (overbought)
    rsi_component = (rsi_bull - rsi_bear)  # -1..+1-ish

    # MACD hist: positive bullish, negative bearish
    macd_component = clamp(features["macd_hist"] / 0.5, -1.0, 1.0)  # scaled

    # Trend: +1 if up, -1 if down
    trend_component = 1.0 if features["trend_up"] == 1 else -1.0

    # Volume/Range spikes -> "unusual activity"
    vol_spike = features["vol_spike"]
    range_spike = features["range_spike"]
    unusual = (vol_spike >= 2.0) or (range_spike >= 2.0)
    unusual_component = 1.0 if unusual else 0.0

    # News sentiment already -1..+1
    news_component = clamp(news_sent, -1.0, 1.0)

    # Weighted sum in -1..+1 range-ish
    w_sum = (
        weights["rsi"] * rsi_component +
        weights["macd"] * macd_component +
        weights["trend"] * trend_component +
        weights["unusual"] * unusual_component +
        weights["news"] * news_component
    )

    # Map to 0..100
    score = int(round(clamp(50 + (w_sum * 50), 0, 100)))

    if score >= 65:
        bias = "Bullish"
    elif score <= 35:
        bias = "Bearish"
    else:
        bias = "Neutral"

    # Simple signal rules (you can tighten later)
    if score >= 75 and features["trend_up"] == 1:
        signal = "BUY"
    elif score <= 25 and features["trend_up"] == 0:
        signal = "SELL"
    else:
        signal = "WAIT"

    unusual_alert = "YES" if unusual else "NO"
    return score, bias, signal, unusual_alert


# =============================
# UI
# =============================
st.title("📈 Option Flow (Unusual Whales) + 🗞️ News (Polygon) + Live Score/Signals")

with st.sidebar:
    st.header("Settings")

    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY", "QQQ", "DIA", "IWM"])
    news_minutes = st.number_input("News lookback (minutes)", 1, 240, 5, 1)
    bars_minutes = st.number_input("Price window (minutes)", 30, 600, 180, 30)

    st.divider()
    st.subheader("Refresh")
    refresh_seconds = st.slider("Auto-refresh (seconds)", 30, 300, 60, 10)
    st.caption("Tip: If you see 429 Too Many Requests, increase refresh seconds or reduce tickers.")

    st.divider()
    st.subheader("Polygon API Key")
    if POLYGON_API_KEY:
        st.success("Polygon key loaded ✅")
    else:
        st.error("Polygon key missing. Add it in Streamlit → Manage app → Settings → Secrets as POLYGON_API_KEY")

    st.divider()
    st.subheader("Scoring weights (total doesn’t have to = 1)")
    w_rsi = st.slider("RSI weight", 0.0, 1.0, 0.25, 0.05)
    w_macd = st.slider("MACD weight", 0.0, 1.0, 0.20, 0.05)
    w_trend = st.slider("Trend (EMA20/50) weight", 0.0, 1.0, 0.20, 0.05)
    w_unusual = st.slider("Unusual activity weight", 0.0, 1.0, 0.20, 0.05)
    w_news = st.slider("News sentiment weight", 0.0, 1.0, 0.15, 0.05)

    weights = {"rsi": w_rsi, "macd": w_macd, "trend": w_trend, "unusual": w_unusual, "news": w_news}

# Auto refresh
st_autorefresh(interval=int(refresh_seconds * 1000), key="autorefresh")

col1, col2 = st.columns([1.15, 1])

with col1:
    st.subheader("Unusual Whales — your screener")
    st.write("This is embedded. For true flow alerts, set alerts inside Unusual Whales (their platform).")
    st.components.v1.iframe(UW_SCREENER_URL, height=900, scrolling=True)

with col2:
    st.subheader("Live Score / Signals (Polygon price + headlines)")
    st.write(f"Last update (UTC): **{utc_now().strftime('%Y-%m-%d %H:%M:%S')}**")

    if not tickers:
        st.info("Pick at least 1 ticker in the sidebar.")
        st.stop()

    # Build score table
    rows = []
    all_news_frames = []

    for t in tickers:
        try:
            bars = polygon_minute_bars(t, int(bars_minutes), POLYGON_API_KEY)
            feats = compute_features(bars)

            news_items = polygon_news(t, int(news_minutes), POLYGON_API_KEY)
            ndf = news_dataframe(news_items, t)
            if not ndf.empty:
                all_news_frames.append(ndf)

            # Average sentiment over window
            news_sent = float(ndf["Sentiment"].mean()) if (not ndf.empty and "Sentiment" in ndf.columns) else 0.0

            score, bias, signal, unusual_alert = compute_score(feats, news_sent, weights)

            rows.append({
                "Ticker": t,
                "Score": score,
                "Bias": bias,
                "Signal": signal,
                "Unusual Alert": unusual_alert,
                "RSI": round(feats.get("rsi", float("nan")), 2) if feats else None,
                "Vol Spike": round(feats.get("vol_spike", float("nan")), 2) if feats else None,
                "Range Spike": round(feats.get("range_spike", float("nan")), 2) if feats else None,
            })

        except requests.HTTPError as e:
            # show cleaner error
            rows.append({"Ticker": t, "Score": 0, "Bias": "Error", "Signal": "WAIT", "Unusual Alert": "NO"})
        except Exception:
            rows.append({"Ticker": t, "Score": 0, "Bias": "Error", "Signal": "WAIT", "Unusual Alert": "NO"})

    score_df = pd.DataFrame(rows)
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Alerts (simple)")
    alerts = score_df[(score_df["Signal"].isin(["BUY", "SELL"])) | (score_df["Unusual Alert"] == "YES")]
    if alerts.empty:
        st.info("No BUY/SELL or unusual activity spikes right now.")
    else:
        for _, r in alerts.iterrows():
            st.warning(f"{r['Ticker']} — Signal: {r['Signal']} | Score: {r['Score']} | Unusual: {r['Unusual Alert']}")

    st.divider()
    st.subheader(f"Polygon News — last {int(news_minutes)} minutes")
    if not all_news_frames:
        st.info("No news in the last window (or Polygon key missing / rate-limited).")
    else:
        news_df = pd.concat(all_news_frames, ignore_index=True)
        st.dataframe(news_df, use_container_width=True, hide_index=True)

        st.subheader("Clickable links")
        for _, row in news_df.iterrows():
            title = row.get("Title", "") or "(no title)"
            url = row.get("URL", "") or ""
            if url:
                st.markdown(f"- **{row['Ticker']}** — [{title}]({url})")

