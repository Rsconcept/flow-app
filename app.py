import os
import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Flow + News + Score (Live)", layout="wide")

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "AMD", "NVDA"]

# Your Unusual Whales screener link
UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF&limit=250&max_dte=7"
    "&min_diff=-0.1&min_oi_change_perc=-10&min_premium=500000"
    "&min_volume_oi_ratio=1&min_volume_ticker_vol_ratio=0.03"
    "&order=premium&order_direction=desc&watchlist_name=GPT%20Filter%20"
)

POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY", os.getenv("POLYGON_API_KEY", "")).strip()

USER_AGENT = {"User-Agent": "streamlit-flow-news-app/1.0"}


# =============================
# SMALL HELPERS
# =============================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def safe_get(url: str, params: dict, timeout: int = 20, tries: int = 3) -> dict:
    """
    Polygon will rate-limit (429). We do small backoff retries.
    """
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=USER_AGENT)
            if r.status_code == 429:
                # backoff: 1s, 2s, 4s
                time.sleep(1 * (2 ** i))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(str(last_err) if last_err else "Request failed")


# =============================
# POLYGON: NEWS
# =============================
@st.cache_data(ttl=60)
def polygon_news(ticker: str, api_key: str, limit: int = 20) -> list[dict]:
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker,
        "limit": limit,
        "order": "desc",
        "sort": "published_utc",
        "apiKey": api_key,
    }
    data = safe_get(url, params=params, timeout=20, tries=3)
    return data.get("results", []) or []


def normalize_news(items: list[dict], ticker: str) -> pd.DataFrame:
    rows = []
    for it in items:
        published = it.get("published_utc", "")
        title = it.get("title", "")
        url = it.get("article_url", "") or it.get("amp_url", "") or ""
        source = (it.get("publisher", {}) or {}).get("name", "")
        rows.append(
            {
                "Ticker": ticker,
                "Published_UTC": published,
                "Title": title,
                "Source": source,
                "URL": url,
            }
        )
    df = pd.DataFrame(rows)
    return df


# Simple headline sentiment (fast + no extra libraries)
POS_WORDS = {
    "beats", "beat", "surge", "soar", "rally", "upgrades", "upgrade", "bullish",
    "record", "strong", "growth", "profit", "profits", "wins", "approval", "approved"
}
NEG_WORDS = {
    "miss", "misses", "plunge", "drop", "drops", "down", "downgrade", "downgrades", "bearish",
    "fraud", "probe", "investigation", "lawsuit", "bankruptcy", "warning", "cut", "cuts"
}

def headline_sentiment_score(title: str) -> float:
    """
    returns -1..+1 based on keywords
    """
    t = (title or "").lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    # normalize
    return max(-1.0, min(1.0, (pos - neg) / (pos + neg)))


# =============================
# POLYGON: PRICE DATA (1-min bars)
# =============================
@st.cache_data(ttl=30)
def polygon_minute_bars(ticker: str, api_key: str, minutes: int = 240) -> pd.DataFrame:
    """
    Pull last N minutes of 1-min bars using /v2/aggs.
    """
    end = utc_now()
    start = end - timedelta(minutes=minutes)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start.date()}/{end.date()}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    data = safe_get(url, params=params, timeout=20, tries=3)
    results = data.get("results", []) or []
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # Polygon columns: o,h,l,c,v,t (ms)
    df["dt"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["dt", "open", "high", "low", "close", "volume"]].copy()
    # keep only last N minutes
    df = df[df["dt"] >= (utc_now() - timedelta(minutes=minutes))]
    return df.reset_index(drop=True)


# =============================
# INDICATORS
# =============================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-10))
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, length: int = 20, mult: float = 2.0):
    mid = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return lower, mid, upper


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    # vwap on minute bars
    pv = (df["close"] * df["volume"]).cumsum()
    vv = df["volume"].cumsum().replace(0, 1e-10)
    return pv / vv


# =============================
# SCORING + SIGNALS
# =============================
def compute_score_and_signal(df: pd.DataFrame, weights: dict) -> dict:
    """
    Returns:
      score 0..100
      bias: Bullish/Bearish/Neutral
      signal: BUY/SELL/HOLD
      unusual_alert: True/False (volume/range spike proxy)
      debug metrics
    """
    if df.empty or len(df) < 50:
        return {
            "score": 50,
            "bias": "Neutral",
            "signal": "HOLD",
            "unusual_alert": False,
            "reason": "Not enough price data",
        }

    close = df["close"]

    df = df.copy()
    df["rsi"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    df["macd_hist"] = hist
    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["atr14"] = atr(df, 14)
    df["vwap"] = vwap(df)
    bb_l, bb_m, bb_u = bollinger(close, 20, 2.0)
    df["bb_l"] = bb_l
    df["bb_u"] = bb_u

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # --- Features scaled into -1..+1 ---
    # RSI: oversold bullish, overbought bearish
    rsi_val = float(last["rsi"])
    rsi_feat = 0.0
    if rsi_val <= 30:
        rsi_feat = +1.0
    elif rsi_val >= 70:
        rsi_feat = -1.0
    else:
        # map 30..70 to +0.5..-0.5
        rsi_feat = (50 - rsi_val) / 40  # 50->0, 30->+0.5, 70->-0.5

    # MACD histogram slope
    macd_feat = float(last["macd_hist"] - prev["macd_hist"])
    macd_feat = max(-1.0, min(1.0, macd_feat * 50))  # scale

    # Trend: EMA20 vs EMA50
    trend_feat = 1.0 if last["ema20"] > last["ema50"] else -1.0

    # Price vs VWAP
    vwap_feat = 1.0 if last["close"] > last["vwap"] else -1.0

    # Bollinger: near lower band bullish, near upper bearish
    bb_feat = 0.0
    if pd.notna(last["bb_l"]) and pd.notna(last["bb_u"]) and (last["bb_u"] - last["bb_l"]) > 0:
        pos = (last["close"] - last["bb_l"]) / (last["bb_u"] - last["bb_l"])
        # pos near 0 => bullish, near 1 => bearish
        bb_feat = max(-1.0, min(1.0, (0.5 - pos) * 2))

    # Unusual proxy: volume spike AND range spike
    vol = df["volume"]
    vol_mean = float(vol.rolling(60).mean().iloc[-1] or 0)
    vol_now = float(last["volume"])
    vol_ratio = (vol_now / vol_mean) if vol_mean > 0 else 0

    range_now = float(last["high"] - last["low"])
    atr_now = float(last["atr14"] or 0)
    range_ratio = (range_now / atr_now) if atr_now > 0 else 0

    unusual_alert = (vol_ratio >= 3.0 and range_ratio >= 1.5)

    # --- Weighted sum ---
    # weights expected sum about 1.0 (not required)
    raw = (
        weights["rsi"] * rsi_feat
        + weights["macd"] * macd_feat
        + weights["trend"] * trend_feat
        + weights["vwap"] * vwap_feat
        + weights["bb"] * bb_feat
        + weights["unusual"] * (1.0 if unusual_alert else 0.0)
    )

    # Convert raw (-something..+something) to 0..100
    # clamp raw to [-1, +1] for stable output
    raw = max(-1.0, min(1.0, raw))
    score = int(round((raw + 1) * 50))  # -1->0, 0->50, +1->100

    # Bias label
    if score >= 60:
        bias = "Bullish"
    elif score <= 40:
        bias = "Bearish"
    else:
        bias = "Neutral"

    # BUY/SELL rules (simple + reliable):
    # BUY if bullish + rsi not overbought + trend ok
    # SELL if bearish + rsi not oversold + trend down
    signal = "HOLD"
    if score >= 70 and rsi_val < 65 and trend_feat > 0:
        signal = "BUY"
    elif score <= 30 and rsi_val > 35 and trend_feat < 0:
        signal = "SELL"

    return {
        "score": score,
        "bias": bias,
        "signal": signal,
        "unusual_alert": unusual_alert,
        "rsi": round(rsi_val, 2),
        "vol_ratio": round(vol_ratio, 2),
        "range_ratio": round(range_ratio, 2),
    }


# =============================
# APP UI
# =============================
st.title("📈 Option Flow (Unusual Whales) + 🗞️ News (Polygon) + Live Score/Signals")

# Sidebar controls
with st.sidebar:
    st.header("Settings")

    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY", "QQQ", "DIA", "IWM"])
    lookback_news = st.number_input("News lookback (minutes)", min_value=1, max_value=120, value=5, step=1)

    st.divider()
    st.subheader("Refresh")
    refresh_seconds = st.slider("Auto-refresh (seconds)", min_value=15, max_value=300, value=60, step=15)

    st.divider()
    st.subheader("Polygon API Key")
    if POLYGON_API_KEY:
        st.success("Polygon key loaded ✅")
    else:
        st.error("Polygon key missing. Add it in Streamlit → App settings → Secrets: POLYGON_API_KEY")

    st.divider()
    st.subheader("Scoring weights")
    w_rsi = st.slider("RSI weight", 0.0, 1.0, 0.25, 0.05)
    w_macd = st.slider("MACD weight", 0.0, 1.0, 0.20, 0.05)
    w_trend = st.slider("Trend (EMA20/50) weight", 0.0, 1.0, 0.20, 0.05)
    w_vwap = st.slider("VWAP weight", 0.0, 1.0, 0.15, 0.05)
    w_bb = st.slider("Bollinger weight", 0.0, 1.0, 0.10, 0.05)
    w_unusual = st.slider("Unusual activity weight", 0.0, 1.0, 0.10, 0.05)

    weights = {
        "rsi": w_rsi,
        "macd": w_macd,
        "trend": w_trend,
        "vwap": w_vwap,
        "bb": w_bb,
        "unusual": w_unusual,
    }

# Auto refresh
st_autorefresh(interval=int(refresh_seconds * 1000), key="autorefresh")

# Layout
col1, col2 = st.columns([1.15, 1])

# LEFT: Unusual Whales
with col1:
    st.subheader("Unusual Whales — your screener")
    st.caption("This is embedded. For true flow alerts, set alerts inside Unusual Whales (their platform).")
    st.components.v1.iframe(UW_SCREENER_URL, height=900, scrolling=True)

# RIGHT: Live Score + News
with col2:
    st.subheader("Live Score / Signals (Polygon price + headlines)")
    st.write(f"Last update (UTC): **{utc_now().strftime('%Y-%m-%d %H:%M:%S')}**")

    if not POLYGON_API_KEY:
        st.stop()

    if not tickers:
        st.info("Pick at least 1 ticker in the sidebar.")
        st.stop()

    # ---- Live scoring table ----
    rows = []
    for t in tickers:
        try:
            bars = polygon_minute_bars(t, POLYGON_API_KEY, minutes=240)
            out = compute_score_and_signal(bars, weights)

            rows.append(
                {
                    "Ticker": t,
                    "Score": out.get("score", 50),
                    "Bias": out.get("bias", "Neutral"),
                    "Signal": out.get("signal", "HOLD"),
                    "Unusual Alert": "🚨 YES" if out.get("unusual_alert") else "",
                    "RSI": out.get("rsi", ""),
                    "Vol Spike": out.get("vol_ratio", ""),
                    "Range Spike": out.get("range_ratio", ""),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "Ticker": t,
                    "Score": "",
                    "Bias": "Error",
                    "Signal": "",
                    "Unusual Alert": "",
                    "RSI": "",
                    "Vol Spike": "",
                    "Range Spike": "",
                }
            )
            st.warning(f"{t} scoring error: {e}")

    score_df = pd.DataFrame(rows).sort_values(by="Score", ascending=False, na_position="last")
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    # ---- Alerts section ----
    st.divider()
    st.subheader("Alerts (simple)")
    alerts = score_df[score_df["Unusual Alert"] == "🚨 YES"]
    if not alerts.empty:
        st.error("UNUSUAL ACTIVITY detected (Polygon proxy):")
        st.write(alerts[["Ticker", "Score", "Signal", "Vol Spike", "Range Spike"]])
    else:
        st.info("No unusual activity spikes right now.")

    # ---- News section ----
    st.divider()
    st.subheader(f"Polygon News — last {lookback_news} minutes")

    news_frames = []
    cutoff = utc_now() - timedelta(minutes=int(lookback_news))

    for t in tickers:
        try:
            items = polygon_news(t, POLYGON_API_KEY, limit=20)
            df = normalize_news(items, t)
            if not df.empty:
                # Parse published UTC if possible
                df["Published_dt"] = pd.to_datetime(df["Published_UTC"], errors="coerce", utc=True)
                df = df[df["Published_dt"].notna()]
                df = df[df["Published_dt"] >= cutoff]

                # sentiment
                df["Sentiment"] = df["Title"].apply(headline_sentiment_score)
                news_frames.append(df)
        except Exception as e:
            st.warning(f"{t} news error: {e}")

    if not news_frames:
        st.info("No headlines in the lookback window.")
    else:
        news_df = pd.concat(news_frames, ignore_index=True)
        news_df = news_df.sort_values("Published_dt", ascending=False)

        st.dataframe(
            news_df[["Ticker", "Published_UTC", "Sentiment", "Title", "Source", "URL"]],
            use_container_width=True,
            hide_index=True,
        )

        st.caption("Sentiment: -1 bearish … 0 neutral … +1 bullish (headline keyword-based)")
        st.subheader("Clickable links")
        for _, row in news_df.iterrows():
            title = row.get("Title", "") or "(no title)"
            url = row.get("URL", "") or ""
            if url:
                st.markdown(f"- **{row['Ticker']}** — [{title}]({url})")
