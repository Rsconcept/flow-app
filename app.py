import os
import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Python 3.9+ (Streamlit Cloud is fine)
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# =========================================================
# TIMEZONE (CST)
# =========================================================
CHI_TZ = ZoneInfo("America/Chicago") if ZoneInfo else None


def now_cst() -> datetime:
    if CHI_TZ:
        return datetime.now(CHI_TZ)
    # fallback (approx CST) if zoneinfo missing
    return datetime.now(timezone.utc) - timedelta(hours=6)


def fmt_cst(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Institutional Options Signals", layout="wide")

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "NVDA", "AMD"]

UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF"
    "&limit=250&max_dte=3"
    "&min_premium=1000000"
    "&min_volume_oi_ratio=1"
    "&order=premium&order_direction=desc"
    "&watchlist_name=GPT%20Filter%20"
)

# =========================================================
# SECRETS
# =========================================================
EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")).strip()
UW_TOKEN = st.secrets.get("UW_TOKEN", os.getenv("UW_TOKEN", "")).strip()

# Default to correct UW endpoint (you posted this one)
UW_FLOW_ALERTS_URL = st.secrets.get(
    "UW_FLOW_ALERTS_URL",
    os.getenv("UW_FLOW_ALERTS_URL", "https://api.unusualwhales.com/api/option-trade/flow-alerts"),
).strip()

# =========================================================
# HELPERS (HTTP)
# =========================================================
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

def http_get_json(url: str, headers=None, params=None, timeout=25):
    h = dict(UA_HEADERS)
    if headers:
        h.update(headers)
    r = requests.get(url, headers=h, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=None):
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


# =========================================================
# TECHNICALS (5m)
# =========================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss.replace(0, math.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)


def macd_hist(close: pd.Series):
    macd_line = ema(close, 12) - ema(close, 26)
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return hist


def vwap_from_ohlcv(df: pd.DataFrame) -> pd.Series:
    # Typical price VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["volume"].cumsum().replace(0, math.nan)
    return (cum_pv / cum_vol).fillna(method="ffill")


# =========================================================
# EODHD DATA
# =========================================================
@st.cache_data(ttl=30)
def eodhd_intraday_5m(symbol_us: str, days_back: int = 5) -> pd.DataFrame:
    """
    EODHD intraday docs vary per plan.
    We request last few days and compute indicators from returned bars.
    """
    if not EODHD_API_KEY:
        return pd.DataFrame()

    # EODHD expects something like AAPL.US
    url = f"https://eodhd.com/api/intraday/{symbol_us}"
    from_date = (now_cst() - timedelta(days=days_back)).date().isoformat()

    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "interval": "5m",
        "from": from_date,
    }

    data = http_get_json(url, params=params)
    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # expected keys: datetime, open, high, low, close, volume
    if "datetime" not in df.columns:
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=60)
def eodhd_news(symbol_us: str, minutes_lookback: int = 60, limit: int = 50) -> pd.DataFrame:
    """
    Simple news fetch from EODHD.
    """
    if not EODHD_API_KEY:
        return pd.DataFrame()

    url = "https://eodhd.com/api/news"
    # EODHD uses s=SYMBOL.US or symbols=...
    since = now_cst() - timedelta(minutes=minutes_lookback)
    since_iso = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "s": symbol_us,
        "limit": limit,
        "from": since_iso,
    }

    try:
        data = http_get_json(url, params=params)
    except Exception:
        return pd.DataFrame()

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # normalize
    if "date" in df.columns:
        df["published_utc"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        df["published_utc"] = pd.NaT

    df["title"] = df.get("title", "")
    df["source"] = df.get("source", "")
    df["link"] = df.get("link", "")
    df = df[["published_utc", "source", "title", "link"]].dropna(subset=["title"])
    df = df.sort_values("published_utc", ascending=False).reset_index(drop=True)
    return df


# =========================================================
# NEWS SENTIMENT (simple, stable wordlist)
# =========================================================
POS_WORDS = {"beats", "beat", "surge", "soar", "rally", "upgrade", "wins", "win", "growth", "record", "strong"}
NEG_WORDS = {"miss", "misses", "drop", "drops", "plunge", "downgrade", "lawsuit", "weak", "fraud", "halt", "crash"}

def simple_sentiment_score(titles: list[str]) -> float:
    if not titles:
        return 0.0
    score = 0
    for t in titles:
        low = (t or "").lower()
        for w in POS_WORDS:
            if w in low:
                score += 1
        for w in NEG_WORDS:
            if w in low:
                score -= 1
    # clamp to [-1, 1]
    denom = max(5, len(titles))
    s = score / denom
    return max(-1.0, min(1.0, s))


# =========================================================
# UNUSUAL WHALES (UW) DATA
# =========================================================
def uw_headers():
    # UW uses: Authorization: Bearer <token>
    if not UW_TOKEN:
        return {}
    return {"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json, text/plain"}


@st.cache_data(ttl=60)
def uw_options_volume(ticker: str) -> dict:
    """
    Correct endpoint you posted:
    GET https://api.unusualwhales.com/api/stock/{ticker}/options-volume
    """
    if not UW_TOKEN:
        return {}

    url = f"https://api.unusualwhales.com/api/stock/{ticker}/options-volume"
    data = http_get_json(url, headers=uw_headers())
    # data is {"data":[{...}]}
    rows = data.get("data", []) if isinstance(data, dict) else []
    return rows[0] if rows else {}


@st.cache_data(ttl=30)
def uw_flow_alerts(limit: int = 200) -> list[dict]:
    """
    Uses the URL you put in secrets:
    UW_FLOW_ALERTS_URL = https://api.unusualwhales.com/api/option-trade/flow-alerts
    """
    if not UW_TOKEN or not UW_FLOW_ALERTS_URL:
        return []

    params = {"limit": limit}
    # Some UW endpoints use pagination/filters. We keep it simple and filter client-side.
    data = http_get_json(UW_FLOW_ALERTS_URL, headers=uw_headers(), params=params)
    # Usually {"data":[...]}
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    # sometimes returns a list directly
    if isinstance(data, list):
        return data
    return []


# =========================================================
# FLOW FILTERING (your rules)
# =========================================================
def is_itm(option_type: str, strike: float, underlying: float) -> bool:
    if option_type == "call":
        return underlying > strike
    if option_type == "put":
        return underlying < strike
    return False


def dte_days(expiry_str: str) -> int | None:
    try:
        exp = datetime.fromisoformat(expiry_str).date()
        today = now_cst().date()
        return (exp - today).days
    except Exception:
        return None


def normalize_and_filter_alerts(alerts: list[dict], tickers_set: set[str]) -> pd.DataFrame:
    """
    Apply:
    - premium >= 1,000,000
    - DTE <= 3
    - Volume > OI
    - Exclude ITM
    - Tickers filter
    """
    out = []
    for a in alerts:
        sym = (a.get("underlying_symbol") or a.get("symbol") or "").upper().strip()
        if not sym or sym not in tickers_set:
            continue

        premium = safe_float(a.get("premium"), 0.0)
        if premium < 1_000_000:
            continue

        expiry = a.get("expiry") or a.get("expiration") or a.get("exp") or ""
        dte = dte_days(str(expiry)) if expiry else None
        if dte is None or dte > 3 or dte < 0:
            continue

        vol = safe_int(a.get("volume"), 0) or 0
        oi = safe_int(a.get("open_interest"), 0) or 0
        if not (vol > oi):
            continue

        opt_type = (a.get("option_type") or a.get("type") or "").lower().strip()
        strike = safe_float(a.get("strike"), None)
        und = safe_float(a.get("underlying_price"), None)
        if strike is not None and und is not None and opt_type in {"call", "put"}:
            if is_itm(opt_type, strike, und):
                continue

        executed_at = a.get("executed_at") or a.get("timestamp") or ""
        out.append({
            "Ticker": sym,
            "Type": opt_type.upper() if opt_type else "",
            "Strike": strike,
            "Expiry": expiry,
            "DTE": dte,
            "Premium": premium,
            "Volume": vol,
            "OI": oi,
            "SideTags": ",".join(a.get("tags", [])) if isinstance(a.get("tags"), list) else "",
            "Executed": executed_at,
        })

    df = pd.DataFrame(out)
    if df.empty:
        return df
    df = df.sort_values("Premium", ascending=False).reset_index(drop=True)
    return df


# =========================================================
# 10Y YIELD (OPTIONAL / BEST-EFFORT)
# =========================================================
@st.cache_data(ttl=120)
def ten_year_yield_optional() -> float | None:
    """
    Not all EODHD plans support macro/bonds the same way.
    We attempt a common symbol. If it fails, we return None (and UI shows 'Not available (ok)').
    """
    if not EODHD_API_KEY:
        return None

    # Common attempts (may fail depending on plan)
    candidates = [
        "US10Y.INDX",
        "US10Y.YIELD",
        "US10Y",
    ]

    for sym in candidates:
        try:
            url = f"https://eodhd.com/api/real-time/{sym}"
            params = {"api_token": EODHD_API_KEY, "fmt": "json"}
            data = http_get_json(url, params=params)
            # try common keys
            val = safe_float(data.get("close")) or safe_float(data.get("price")) or safe_float(data.get("value"))
            if val is not None:
                return val
        except Exception:
            continue
    return None


# =========================================================
# SCORING (CALLS/PUTS ONLY)
# =========================================================
def compute_signal_row(ticker: str, df_ohlcv: pd.DataFrame, uw_vol: dict, news_df: pd.DataFrame,
                       y10: float | None,
                       weights: dict,
                       institutional_min: int) -> dict:

    # defaults if we can't compute
    base = {
        "Ticker": ticker,
        "Confidence": 50,
        "Direction": "—",
        "Signal": "WAIT",
        "UW Unusual": "NO",
        "UW Bias": "Neutral",
        "Gamma bias": "Neutral",
        "RSI": None,
        "MACD_hist": None,
        "VWAP": None,
        "EMA stack": None,
        "Vol_ratio": None,
        "IV spike": "None",
        "10Y": y10 if y10 is not None else "N/A",
    }

    if df_ohlcv is None or df_ohlcv.empty or len(df_ohlcv) < 35:
        base["Signal"] = "WAIT"
        base["Direction"] = "—"
        base["Confidence"] = 50
        return base

    df = df_ohlcv.copy()

    close = df["close"]
    df["EMA9"] = ema(close, 9)
    df["EMA20"] = ema(close, 20)
    df["EMA50"] = ema(close, 50)
    df["RSI"] = rsi(close, 14)
    df["MACD_H"] = macd_hist(close)
    df["VWAP"] = vwap_from_ohlcv(df)

    # volume ratio (last volume / avg last 20)
    vol = df["volume"].fillna(0)
    vol_avg = vol.rolling(20).mean()
    vol_ratio = (vol / vol_avg.replace(0, math.nan)).iloc[-1]
    vol_ratio = float(vol_ratio) if pd.notna(vol_ratio) else None

    last_close = float(close.iloc[-1])
    last_vwap = float(df["VWAP"].iloc[-1])
    last_rsi = float(df["RSI"].iloc[-1])
    last_macd_h = float(df["MACD_H"].iloc[-1])

    ema9 = float(df["EMA9"].iloc[-1])
    ema20 = float(df["EMA20"].iloc[-1])
    ema50 = float(df["EMA50"].iloc[-1])

    # EMA stack: bullish if 9>20>50 and price above 9
    ema_stack = "Neutral"
    if ema9 > ema20 > ema50 and last_close >= ema9:
        ema_stack = "Bullish"
    elif ema9 < ema20 < ema50 and last_close <= ema9:
        ema_stack = "Bearish"

    vwap_bias = "Above" if last_close >= last_vwap else "Below"

    # UW options-volume bias
    uw_bias = "Neutral"
    gamma_bias = "Neutral"
    iv_spike = "None"

    if uw_vol:
        bull_prem = safe_float(uw_vol.get("bullish_premium"), 0.0) or 0.0
        bear_prem = safe_float(uw_vol.get("bearish_premium"), 0.0) or 0.0
        if bull_prem > bear_prem * 1.1:
            uw_bias = "Bullish"
        elif bear_prem > bull_prem * 1.1:
            uw_bias = "Bearish"

        call_oi = safe_float(uw_vol.get("call_open_interest"), 0.0) or 0.0
        put_oi = safe_float(uw_vol.get("put_open_interest"), 0.0) or 0.0
        net_call = safe_float(uw_vol.get("net_call_premium"), 0.0) or 0.0
        net_put = safe_float(uw_vol.get("net_put_premium"), 0.0) or 0.0

        # Gamma proxy: OI dominance + net premium dominance
        if call_oi > put_oi and net_call >= net_put:
            gamma_bias = "Positive Gamma (proxy)"
        elif put_oi > call_oi and net_put >= net_call:
            gamma_bias = "Negative Gamma (proxy)"

        # IV spike proxy: today's call/put volume vs 30d avg volume
        call_vol = safe_float(uw_vol.get("call_volume"), 0.0) or 0.0
        put_vol = safe_float(uw_vol.get("put_volume"), 0.0) or 0.0
        avg30c = safe_float(uw_vol.get("avg_30_day_call_volume"), None)
        avg30p = safe_float(uw_vol.get("avg_30_day_put_volume"), None)
        spike = False
        if avg30c and avg30c > 0 and call_vol / avg30c >= 1.8:
            spike = True
        if avg30p and avg30p > 0 and put_vol / avg30p >= 1.8:
            spike = True
        iv_spike = "YES (proxy)" if spike else "None"

    # News sentiment (simple)
    news_sent = 0.0
    if news_df is not None and not news_df.empty:
        titles = news_df["title"].astype(str).head(10).tolist()
        news_sent = simple_sentiment_score(titles)

    # ---------------------------------------------------------
    # SCORING → map to CALLS vs PUTS
    # ---------------------------------------------------------
    bull_points = 0.0
    bear_points = 0.0

    # VWAP
    if vwap_bias == "Above":
        bull_points += weights["vwap"]
    else:
        bear_points += weights["vwap"]

    # EMA stack
    if ema_stack == "Bullish":
        bull_points += weights["ema"]
    elif ema_stack == "Bearish":
        bear_points += weights["ema"]

    # RSI
    # bullish if rising off oversold-ish, bearish if falling from overbought-ish
    if last_rsi <= 35:
        bull_points += weights["rsi"] * 0.8
    elif last_rsi >= 65:
        bear_points += weights["rsi"] * 0.8

    # MACD hist
    if last_macd_h > 0:
        bull_points += weights["macd"]
    elif last_macd_h < 0:
        bear_points += weights["macd"]

    # Volume ratio
    if vol_ratio is not None:
        if vol_ratio >= 1.5:
            # direction depends on trend
            if ema_stack == "Bullish" or vwap_bias == "Above":
                bull_points += weights["vol"]
            elif ema_stack == "Bearish" or vwap_bias == "Below":
                bear_points += weights["vol"]

    # UW bias
    if uw_bias == "Bullish":
        bull_points += weights["uw"]
    elif uw_bias == "Bearish":
        bear_points += weights["uw"]

    # News sentiment
    if news_sent > 0.15:
        bull_points += weights["news"]
    elif news_sent < -0.15:
        bear_points += weights["news"]

    # 10Y filter (optional): rising yields can pressure growth / risk (rough)
    # If not available, it doesn't affect scoring.
    if isinstance(y10, (int, float)):
        # tiny influence only
        if y10 >= 4.5:  # rough threshold, tweak later
            bear_points += weights["y10"]
        elif y10 <= 3.5:
            bull_points += weights["y10"]

    # Normalize to 0..100 confidence
    total = max(0.0001, bull_points + bear_points)
    bull_conf = (bull_points / total) * 100.0
    bear_conf = (bear_points / total) * 100.0

    # Pick direction by stronger side
    if bull_conf > bear_conf:
        direction = "CALLS"
        confidence = int(round(bull_conf))
    elif bear_conf > bull_conf:
        direction = "PUTS"
        confidence = int(round(bear_conf))
    else:
        direction = "—"
        confidence = 50

    signal = "WAIT"
    if confidence >= institutional_min:
        signal = f"BUY {direction}"

    base.update({
        "Confidence": confidence,
        "Direction": direction if direction != "—" else "—",
        "Signal": signal,
        "UW Bias": uw_bias,
        "Gamma bias": gamma_bias,
        "RSI": round(last_rsi, 1),
        "MACD_hist": round(last_macd_h, 4),
        "VWAP": vwap_bias,
        "EMA stack": ema_stack,
        "Vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        "IV spike": iv_spike,
    })

    return base


# =========================================================
# UI (SIDEBAR)
# =========================================================
st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")
st.caption(f"Last update (CST): **{fmt_cst(now_cst())}**")

with st.sidebar:
    st.header("Settings")

    # ✅ You can type ANY ticker(s)
    tickers_text = st.text_input(
        "Tickers (type any, comma-separated)",
        value="SPY, TSLA",
        help="Example: SPY, QQQ, NVDA, AAPL, MSFT (commas or spaces work)",
    )

    # parse tickers
    raw = tickers_text.replace("\n", " ").replace(";", ",").replace("|", ",")
    parts = [p.strip().upper() for p in raw.replace(" ", ",").split(",") if p.strip()]
    # remove garbage
    tickers = []
    for t in parts:
        t = "".join(ch for ch in t if ch.isalnum() or ch in {".", "-", "_"})
        if t and t not in tickers:
            tickers.append(t)

    if not tickers:
        tickers = ["SPY"]

    news_minutes = st.number_input("News lookback (minutes)", 1, 240, 60, 5)
    price_lookback_minutes = st.number_input("Price lookback (minutes)", 60, 1440, 240, 60)

    st.divider()
    st.subheader("Institutional mode")
    institutional_min = st.slider("Minimum confidence (signals)", 50, 95, 75, 1)

    st.divider()
    st.subheader("Refresh")
    refresh_seconds = st.slider("Auto-refresh (seconds)", 10, 180, 30, 5)

    st.divider()
    st.subheader("Weights (total doesn't have to be 1)")
    w_vwap = st.slider("VWAP", 0.0, 0.5, 0.15, 0.01)
    w_ema  = st.slider("EMA stack (9/20/50)", 0.0, 0.5, 0.18, 0.01)
    w_rsi  = st.slider("RSI", 0.0, 0.5, 0.15, 0.01)
    w_macd = st.slider("MACD hist", 0.0, 0.5, 0.18, 0.01)
    w_vol  = st.slider("Volume ratio", 0.0, 0.5, 0.12, 0.01)
    w_uw   = st.slider("UW options-volume bias", 0.0, 0.7, 0.18, 0.01)
    w_news = st.slider("News sentiment", 0.0, 0.5, 0.04, 0.01)
    w_y10  = st.slider("10Y yield (optional)", 0.0, 0.2, 0.05, 0.01)

weights = {
    "vwap": w_vwap,
    "ema": w_ema,
    "rsi": w_rsi,
    "macd": w_macd,
    "vol": w_vol,
    "uw": w_uw,
    "news": w_news,
    "y10": w_y10,
}

# Auto-refresh
st_autorefresh(interval=int(refresh_seconds * 1000), key="auto_refresh")


# =========================================================
# STATUS PANEL (GREEN/RED)
# =========================================================
with st.sidebar:
    st.divider()
    st.subheader("Keys status (green/red)")

    if EODHD_API_KEY:
        st.success("EODHD_API_KEY ✅")
    else:
        st.error("EODHD_API_KEY ❌ (missing)")

    if UW_TOKEN:
        st.success("UW_TOKEN (Bearer) ✅")
    else:
        st.error("UW_TOKEN ❌ (missing)")

    if UW_FLOW_ALERTS_URL and "api.unusualwhales.com" in UW_FLOW_ALERTS_URL:
        st.success("UW_FLOW_ALERTS_URL ✅")
    else:
        st.error("UW_FLOW_ALERTS_URL ❌ (must be api.unusualwhales.com)")

    st.divider()
    st.subheader("Endpoints status")

    # quick pings (best-effort)
    # EODHD intraday test
    try:
        test_sym = f"{tickers[0]}.US"
        test_df = eodhd_intraday_5m(test_sym)
        if not test_df.empty:
            st.success("EODHD intraday ✅")
        else:
            st.error("EODHD intraday ❌ (empty/no permission)")
    except Exception as e:
        st.error(f"EODHD intraday ❌ ({str(e)[:60]})")

    # EODHD news test
    try:
        test_news = eodhd_news(f"{tickers[0]}.US", minutes_lookback=int(news_minutes))
        st.success("EODHD news ✅" if not test_news.empty else "EODHD news ✅ (no headlines)")
    except Exception as e:
        st.error(f"EODHD news ❌ ({str(e)[:60]})")

    # UW options volume test
    try:
        ov = uw_options_volume(tickers[0])
        st.success("UW options-volume ✅" if ov else "UW options-volume ❌ (empty)")
    except Exception as e:
        st.error(f"UW options-volume ❌ ({str(e)[:60]})")

    # UW flow alerts test
    try:
        fa = uw_flow_alerts(limit=20)
        st.success("UW flow-alerts ✅" if isinstance(fa, list) else "UW flow-alerts ❌")
    except Exception as e:
        st.error(f"UW flow-alerts ❌ ({str(e)[:60]})")

    # 10Y (optional)
    try:
        y10 = ten_year_yield_optional()
        if y10 is None:
            st.info("10Y yield (optional) — Not available (ok)")
        else:
            st.success(f"10Y yield ✅ ({y10:.2f})")
    except Exception:
        st.info("10Y yield (optional) — Not available (ok)")


# =========================================================
# MAIN LAYOUT
# =========================================================
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("This is embedded. True filtering is best done inside the screener itself.")
    st.components.v1.iframe(UW_SCREENER_URL, height=850, scrolling=True)

with right:
    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW flow)")

    # fetch 10Y once
    try:
        y10_val = ten_year_yield_optional()
    except Exception:
        y10_val = None

    # compute per ticker
    rows = []
    debug_rows = []

    for t in tickers:
        sym_us = f"{t}.US"

        # price bars
        df_bars = pd.DataFrame()
        try:
            df_bars = eodhd_intraday_5m(sym_us, days_back=6)
        except Exception as e:
            debug_rows.append({"Ticker": t, "Data mode": "price_error", "Bars": 0, "Last bar": str(e)[:80]})

        # keep only last lookback minutes
        if not df_bars.empty:
            cutoff = now_cst().astimezone(timezone.utc) - timedelta(minutes=int(price_lookback_minutes))
            # bars datetime might be naive; coerce to UTC-ish by treating as UTC
            dt = pd.to_datetime(df_bars["datetime"], errors="coerce")
            df_bars = df_bars.loc[dt >= cutoff.replace(tzinfo=None)].copy()
            df_bars = df_bars.reset_index(drop=True)

        # UW options volume
        uw_vol = {}
        try:
            uw_vol = uw_options_volume(t)
        except Exception as e:
            debug_rows.append({"Ticker": t, "Data mode": "uw_vol_error", "Bars": len(df_bars), "Last bar": str(e)[:80]})

        # news
        news_df = pd.DataFrame()
        try:
            news_df = eodhd_news(sym_us, minutes_lookback=int(news_minutes))
        except Exception as e:
            debug_rows.append({"Ticker": t, "Data mode": "news_error", "Bars": len(df_bars), "Last bar": str(e)[:80]})

        row = compute_signal_row(
            ticker=t,
            df_ohlcv=df_bars,
            uw_vol=uw_vol,
            news_df=news_df,
            y10=y10_val,
            weights=weights,
            institutional_min=int(institutional_min),
        )
        rows.append(row)

    signals_df = pd.DataFrame(rows)

    st.caption(f"Last update (CST): **{fmt_cst(now_cst())}**")
    st.dataframe(signals_df, use_container_width=True, hide_index=True)

    # Institutional alerts (>= threshold only)
    st.subheader(f"Institutional Alerts (≥{institutional_min} only)")
    inst = signals_df[signals_df["Confidence"] >= int(institutional_min)].copy()
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        for _, r in inst.iterrows():
            st.success(f"{r['Ticker']}: {r['Signal']} | Confidence={r['Confidence']} | UW={r['UW Bias']} | IV={r['IV spike']} | Gamma={r['Gamma bias']}")

    # UW flow alerts section (filtered)
    st.subheader("Unusual Flow Alerts (UW API)")
    st.caption("Rules applied: premium ≥ $1M, DTE ≤ 3, Volume > OI, exclude ITM, ticker-matched.")
    try:
        raw_alerts = uw_flow_alerts(limit=400)
        alerts_df = normalize_and_filter_alerts(raw_alerts, set(tickers))
        if alerts_df.empty:
            st.info("No matching UW flow alerts right now (or market quiet / after-hours).")
        else:
            st.dataframe(alerts_df.head(50), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"UW flow fetch failed: {e}")

    # News section
    st.subheader(f"News — last {news_minutes} minutes (EODHD)")
    news_frames = []
    for t in tickers:
        sym_us = f"{t}.US"
        df_n = eodhd_news(sym_us, minutes_lookback=int(news_minutes))
        if not df_n.empty:
            df_n = df_n.copy()
            df_n.insert(0, "Ticker", t)
            news_frames.append(df_n)

    if not news_frames:
        st.info("No news in this lookback window (or EODHD returned none).")
    else:
        news_all = pd.concat(news_frames, ignore_index=True)
        # show CST time column
        news_all["published_cst"] = news_all["published_utc"].dt.tz_convert("America/Chicago") if news_all["published_utc"].dt.tz is not None else news_all["published_utc"]
        news_all = news_all[["Ticker", "published_cst", "source", "title", "link"]]
        st.dataframe(news_all, use_container_width=True, hide_index=True)
        st.caption("Clickable links:")
        for _, r in news_all.head(20).iterrows():
            if r.get("link"):
                st.markdown(f"- **{r['Ticker']}** — [{r['title']}]({r['link']})")

    # Debug
    with st.expander("Debug (why indicators might be None)"):
        if debug_rows:
            st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)
        else:
            st.write("No debug warnings right now.")
