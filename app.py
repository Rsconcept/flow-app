import os
import time
from datetime import datetime, timedelta, timezone, date

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

# Your hard rules
MIN_PREMIUM = 1_000_000          # $1M minimum premium
MAX_DTE_DAYS = 3                 # DTE <= 3 days
REQUIRE_VOL_GT_OI = True         # Volume > OI
EXCLUDE_ITM = True               # Exclude ITM

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

def pct_rank_score(value, lo, hi):
    """Map value in [lo,hi] to 0..100"""
    if hi == lo:
        return 50
    return 100 * (value - lo) / (hi - lo)

def symbol_to_eodhd(sym: str) -> str:
    # US tickers
    return f"{sym}.US"

# -----------------------------
# EODHD: PRICE BARS + NEWS
# -----------------------------
@st.cache_data(ttl=20)
def eodhd_intraday_5m(ticker: str, lookback_minutes: int, api_key: str) -> pd.DataFrame:
    """
    Pull 5m intraday bars from EODHD.
    Endpoint pattern (EODHD): /api/intraday/{SYMBOL}?interval=5m&from=...&to=...&api_token=...&fmt=json
    """
    if not api_key:
        return pd.DataFrame()

    sym = symbol_to_eodhd(ticker)
    end_ts = utc_now()
    start_ts = end_ts - timedelta(minutes=lookback_minutes)

    url = f"https://eodhd.com/api/intraday/{sym}"
    params = {
        "api_token": api_key,
        "interval": "5m",
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
    # Expected columns: datetime, open, high, low, close, volume
    # Normalize
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
        df = df.set_index("datetime")

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"])
    return df

@st.cache_data(ttl=30)
def eodhd_news(ticker: str, lookback_minutes: int, api_key: str) -> pd.DataFrame:
    """
    EODHD News endpoint (commonly):
    https://eodhd.com/api/news?api_token=KEY&s=SPY.US&from=YYYY-MM-DD&to=YYYY-MM-DD&limit=...
    We filter by time locally (minutes).
    """
    if not api_key:
        return pd.DataFrame()

    sym = symbol_to_eodhd(ticker)
    now = utc_now()
    from_dt = (now - timedelta(days=3)).date().isoformat()  # buffer
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

    # Common fields: date, title, link, source, content
    # Normalize published time
    if "date" in df.columns:
        df["published_utc"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    else:
        df["published_utc"] = pd.NaT

    cutoff = now - timedelta(minutes=lookback_minutes)
    df = df[df["published_utc"].notna() & (df["published_utc"] >= cutoff)].copy()

    # Normalize columns
    df["Ticker"] = ticker
    df["Title"] = df.get("title", "")
    df["Source"] = df.get("source", "")
    df["URL"] = df.get("link", "")

    df = df[["Ticker", "published_utc", "Source", "Title", "URL"]].sort_values("published_utc", ascending=False)
    return df

def quick_news_sentiment_score(news_df: pd.DataFrame) -> float:
    """
    SUPER simple sentiment proxy (stable, no extra APIs):
    counts positive/negative words in titles.
    Output: -1 .. +1
    """
    if news_df is None or news_df.empty:
        return 0.0

    pos_words = ["beats", "surge", "upgrade", "bull", "record", "strong", "growth", "win", "rally", "profit"]
    neg_words = ["miss", "plunge", "downgrade", "bear", "weak", "fraud", "lawsuit", "loss", "cut", "crash"]

    score = 0
    for t in news_df["Title"].fillna("").astype(str).tolist():
        lt = t.lower()
        for w in pos_words:
            if w in lt:
                score += 1
        for w in neg_words:
            if w in lt:
                score -= 1

    # normalize
    return clamp(score / 8.0, -1.0, 1.0)

# -----------------------------
# INDICATORS (5m bars)
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
    # typical price * volume / cumulative volume
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"].fillna(0)
    cum_pv = pv.cumsum()
    cum_v = df["volume"].fillna(0).cumsum().replace(0, pd.NA)
    out = (cum_pv / cum_v).fillna(method="bfill")
    return out

# -----------------------------
# UNUSUAL WHALES: FLOW ALERTS (API)
# -----------------------------
@st.cache_data(ttl=15)
def uw_flow_alerts_raw(api_url: str, bearer: str, tickers: list[str], limit: int = 200) -> list[dict]:
    """
    Calls the flow_alerts endpoint and returns raw list.
    We try common parameter shapes; if the API ignores them, we filter client-side.
    """
    if not api_url or not bearer:
        return []

    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {bearer}",
    }

    # Try with query params (if supported)
    params_variants = [
        {"limit": limit, "ticker": ",".join(tickers)},     # some APIs accept ticker=
        {"limit": limit, "tickers": ",".join(tickers)},    # or tickers=
        {"limit": limit, "symbols": ",".join(tickers)},    # or symbols=
        {"limit": limit},                                  # fallback
    ]

    last_err = None
    for params in params_variants:
        try:
            r = requests.get(api_url, headers=headers, params=params, timeout=20)
            if r.status_code == 429:
                # brief backoff
                time.sleep(1.0)
                continue
            r.raise_for_status()
            js = r.json()
            # many UW endpoints return {"data":[...]}
            if isinstance(js, dict) and "data" in js and isinstance(js["data"], list):
                return js["data"]
            # sometimes they return list directly
            if isinstance(js, list):
                return js
            return []
        except Exception as e:
            last_err = e
            continue

    # If all variants fail, return empty. Caller can show error summary.
    return []

def is_itm(option_type: str, strike: float, underlying: float) -> bool:
    ot = (option_type or "").lower()
    if ot == "call":
        return underlying > strike
    if ot == "put":
        return underlying < strike
    return False

def dte_days(expiry_str: str) -> int | None:
    """
    expiry like '2025-01-17'
    """
    try:
        exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        return (exp - utc_now().date()).days
    except Exception:
        return None

def filter_uw_alerts(alerts: list[dict], ticker: str) -> list[dict]:
    """
    Enforce your exact filters from the UW alert objects (best effort).
    Requires these fields (common in UW flow objects): underlying_symbol, premium, expiry, option_type, strike,
    underlying_price, volume, open_interest
    """
    out = []
    for a in alerts or []:
        sym = a.get("underlying_symbol") or a.get("ticker") or a.get("symbol")
        if sym != ticker:
            continue

        prem = safe_float(a.get("premium"), 0.0)
        if prem < MIN_PREMIUM:
            continue

        # DTE
        exp = a.get("expiry") or a.get("expiration") or a.get("exp")
        dte = dte_days(exp) if exp else None
        if dte is not None and dte > MAX_DTE_DAYS:
            continue

        # Exclude ITM
        if EXCLUDE_ITM:
            ot = a.get("option_type") or a.get("type")
            strike = safe_float(a.get("strike"), 0.0)
            und = safe_float(a.get("underlying_price"), 0.0)
            if und > 0 and strike > 0 and is_itm(ot, strike, und):
                continue

        # Volume > OI
        if REQUIRE_VOL_GT_OI:
            vol = safe_float(a.get("volume"), 0.0)
            oi = safe_float(a.get("open_interest"), 0.0)
            if not (vol > oi):
                continue

        out.append(a)

    return out

def uw_bias_from_alerts(filtered_alerts: list[dict]) -> tuple[str, bool]:
    """
    Determine bullish/bearish based on premium-weighted call vs put in the filtered alerts.
    Returns (bias, unusual_flag).
    """
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
# SCORING → CALLS / PUTS ONLY
# -----------------------------
def build_score(df: pd.DataFrame, news_sent: float, weights: dict) -> dict:
    """
    Returns component scores + final bullish/bearish confidence.
    """
    if df is None or df.empty or len(df) < 30:
        # Not enough bars → neutral
        return {
            "bull": 50, "bear": 50, "signal": "WAIT", "conf": 0,
            "rsi": None, "macd_hist": None, "vwap_bias": None, "ema_bias": None,
            "vol_ratio": None, "news_sent": news_sent
        }

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].fillna(0).astype(float)

    # Indicators
    rsi14 = rsi(close, 14).iloc[-1]
    mh = macd_hist(close).iloc[-1]

    ema9 = ema(close, 9).iloc[-1]
    ema20 = ema(close, 20).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]

    vw = vwap(df).iloc[-1]
    last = close.iloc[-1]

    # Volume spike ratio vs rolling mean
    vol_mean = vol.rolling(20).mean().iloc[-1]
    vol_ratio = (vol.iloc[-1] / vol_mean) if (vol_mean and vol_mean > 0) else 1.0

    # Component scores (0..100 bullishness)
    # RSI bullish when rising from oversold; bearish when >70
    rsi_bull = 50
    if rsi14 <= 30:
        rsi_bull = 70
    elif rsi14 >= 70:
        rsi_bull = 30
    else:
        rsi_bull = 50 + (50 - abs(rsi14 - 50)) / 2  # mild center bias

    # MACD hist bullish if positive, bearish if negative
    macd_bull = clamp(50 + (mh * 800), 0, 100)  # scaling to make it responsive

    # VWAP bias
    vwap_bull = 75 if last > vw else 25

    # EMA stack bias
    # bullish if 9>20>50 and price above 20; bearish if 9<20<50 and price below 20
    ema_bull = 50
    if ema9 > ema20 > ema50 and last > ema20:
        ema_bull = 75
    elif ema9 < ema20 < ema50 and last < ema20:
        ema_bull = 25

    # Volume score
    vol_bull = clamp(pct_rank_score(vol_ratio, 0.8, 2.5), 0, 100)  # high ratio adds conviction

    # News sentiment score: -1..+1 → 0..100
    news_bull = clamp(50 + news_sent * 35, 0, 100)

    # Weighted bullishness
    bull = (
        weights["vwap"] * vwap_bull +
        weights["ema"]  * ema_bull +
        weights["rsi"]  * rsi_bull +
        weights["macd"] * macd_bull +
        weights["vol"]  * vol_bull +
        weights["news"] * news_bull
    )

    bull = clamp(bull, 0, 100)
    bear = 100 - bull

    # Options-only direction
    if bull >= bear:
        direction = "CALLS"
        conf = bull
    else:
        direction = "PUTS"
        conf = bear

    # Institutional filter (>=75)
    if conf < INSTITUTIONAL_MIN_CONF:
        signal = "WAIT"
    else:
        signal = f"BUY {direction}"

    return {
        "bull": round(bull, 1),
        "bear": round(bear, 1),
        "signal": signal,
        "conf": round(conf, 1),
        "rsi": round(float(rsi14), 2),
        "macd_hist": round(float(mh), 4),
        "vwap_bias": "Above" if last > vw else "Below",
        "ema_bias": "Bull" if ema_bull >= 60 else ("Bear" if ema_bull <= 40 else "Neutral"),
        "vol_ratio": round(float(vol_ratio), 2),
        "news_sent": round(float(news_sent), 2),
    }

# -----------------------------
# UI
# -----------------------------
st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")

# Sidebar
with st.sidebar:
    st.header("Settings")

    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY", "QQQ", "DIA", "IWM"])

    price_lookback = st.number_input("Price lookback (minutes)", min_value=60, max_value=1000, value=420, step=20)
    news_lookback = st.number_input("News lookback (minutes)", min_value=5, max_value=240, value=60, step=5)

    st.divider()
    st.subheader("Refresh")
    refresh_seconds = st.slider("Auto-refresh (seconds)", min_value=15, max_value=300, value=30, step=5)

    st.divider()
    st.subheader("Keys status")
    st.write("EODHD:", "✅" if EODHD_API_KEY else "❌ (missing)")
    st.write("UW Bearer:", "✅" if UW_BEARER else "❌ (missing)")
    st.write("UW Alerts URL:", "✅" if UW_FLOW_ALERTS_URL else "❌ (missing)")

    st.caption("Institutional mode: signals only when confidence ≥ 75.")

    st.divider()
    st.subheader("Weights (institutional defaults)")
    # weights sum does NOT have to equal 1, we normalize below
    w_vwap = st.slider("VWAP weight", 0.0, 1.0, 0.15, 0.01)
    w_ema  = st.slider("EMA stack (9/20/50) weight", 0.0, 1.0, 0.18, 0.01)
    w_rsi  = st.slider("RSI weight", 0.0, 1.0, 0.15, 0.01)
    w_macd = st.slider("MACD weight", 0.0, 1.0, 0.15, 0.01)
    w_vol  = st.slider("Volume spike weight", 0.0, 1.0, 0.22, 0.01)
    w_news = st.slider("News sentiment weight", 0.0, 1.0, 0.15, 0.01)

    total_w = w_vwap + w_ema + w_rsi + w_macd + w_vol + w_news
    if total_w == 0:
        total_w = 1.0

    weights = {
        "vwap": w_vwap / total_w,
        "ema":  w_ema  / total_w,
        "rsi":  w_rsi  / total_w,
        "macd": w_macd / total_w,
        "vol":  w_vol  / total_w,
        "news": w_news / total_w,
    }

# Auto refresh
st_autorefresh(interval=int(refresh_seconds * 1000), key="autorefresh")

# Layout
col_left, col_right = st.columns([1.15, 1])

with col_left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("This is embedded. True filtering (DTE/ITM/premium rules) are best done inside the screener itself.")
    try:
        st.components.v1.iframe(UW_SCREENER_URL, height=850, scrolling=True)
    except Exception as e:
        st.error(f"Embed error. Check UW_SCREENER_URL. Details: {e}")

with col_right:
    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW flow)")
    st.write(f"Last update (UTC): **{utc_now().strftime('%Y-%m-%d %H:%M:%S')}**")

    if not tickers:
        st.info("Pick at least 1 ticker.")
        st.stop()

    # Pull UW alerts once, then filter per ticker client-side
    uw_raw = uw_flow_alerts_raw(UW_FLOW_ALERTS_URL, UW_BEARER, tickers, limit=300)

    rows = []
    alerts_out = []

    for t in tickers:
        # PRICE + NEWS
        df = pd.DataFrame()
        news_df = pd.DataFrame()

        try:
            df = eodhd_intraday_5m(t, int(price_lookback), EODHD_API_KEY)
        except Exception as e:
            st.warning(f"{t}: EODHD intraday error: {e}")

        try:
            news_df = eodhd_news(t, int(news_lookback), EODHD_API_KEY)
        except Exception as e:
            st.warning(f"{t}: EODHD news error: {e}")

        ns = quick_news_sentiment_score(news_df)

        # TECH SCORE
        s = build_score(df, ns, weights)

        # UW FLOW (hard rules)
        uw_filtered = filter_uw_alerts(uw_raw, t)
        uw_bias, uw_unusual = uw_bias_from_alerts(uw_filtered)

        # Combine: If UW unusual exists and bias is Bearish → tilt toward PUTS
        # If UW unusual exists and bias is Bullish → tilt toward CALLS
        bull_adj = s["bull"]
        bear_adj = s["bear"]

        if uw_unusual and uw_bias == "Bullish":
            bull_adj = clamp(bull_adj + 10, 0, 100)
            bear_adj = 100 - bull_adj
        elif uw_unusual and uw_bias == "Bearish":
            bear_adj = clamp(bear_adj + 10, 0, 100)
            bull_adj = 100 - bear_adj

        # Direction = CALLS/PUTS ONLY with institutional threshold
        if bull_adj >= bear_adj:
            direction = "CALLS"
            conf = bull_adj
        else:
            direction = "PUTS"
            conf = bear_adj

        if conf >= INSTITUTIONAL_MIN_CONF:
            signal = f"BUY {direction}"
        else:
            signal = "WAIT"

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
            "News_sent": s["news_sent"],
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
    st.subheader(f"News — last {int(news_lookback)} minutes (EODHD)")
    # show combined recent news for selected tickers
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
        st.write("Clickable links:")
        for _, r in nd.head(20).iterrows():
            title = r.get("Title") or "(no title)"
            url = r.get("URL") or ""
            if url:
                st.markdown(f"- **{r.get('Ticker','')}** — [{title}]({url})")
    else:
        st.info("No news in the last window (or EODHD key missing / rate-limited).")

