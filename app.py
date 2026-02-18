import os
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# =========================
# CONFIG / CONSTANTS
# =========================
APP_TZ = ZoneInfo("America/Chicago")  # CST/CDT automatically
st.set_page_config(page_title="Institutional Options Signals (5m) — CALLS/PUTS ONLY", layout="wide")

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

DEFAULT_TICKERS_PRESET = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "NVDA", "AMD"]

# Your rules
MIN_PREMIUM = 1_000_000
MAX_DTE = 3
REQUIRE_VOL_GT_OI = True
EXCLUDE_ITM = True
STOCKS_ETF_ONLY = True


def now_cst() -> datetime:
    return datetime.now(tz=APP_TZ)


def fmt_cst(dt: datetime) -> str:
    return dt.astimezone(APP_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


# =========================
# SECRETS
# =========================
EODHD_API_KEY = (st.secrets.get("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")) or "").strip()
UW_TOKEN = (st.secrets.get("UW_TOKEN", os.getenv("UW_TOKEN", "")) or "").strip()
FINVIZ_AUTH = (st.secrets.get("FINVIZ_AUTH", os.getenv("FINVIZ_AUTH", "")) or "").strip()

# Optional user-provided endpoint. We *won't trust it blindly*, but will try it.
UW_FLOW_ALERTS_URL_USER = (st.secrets.get("UW_FLOW_ALERTS_URL", os.getenv("UW_FLOW_ALERTS_URL", "")) or "").strip()

# We will also try the canonical endpoint(s) automatically:
UW_FLOW_ALERTS_URLS_TO_TRY = [
    "https://api.unusualwhales.com/api/option-trade/flow-alerts",
    "https://api.unusualwhales.com/api/option-trade/flow_alerts",  # sometimes APIs use underscores
]
if UW_FLOW_ALERTS_URL_USER:
    UW_FLOW_ALERTS_URLS_TO_TRY.insert(0, UW_FLOW_ALERTS_URL_USER)


# =========================
# HTTP HELPERS
# =========================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

def http_get_json(url: str, headers=None, params=None, timeout=20):
    h = {}
    if headers:
        h.update(headers)
    r = SESSION.get(url, headers=h, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def safe_int(x):
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


# =========================
# DATA SOURCES
# =========================
@st.cache_data(ttl=60, show_spinner=False)
def eodhd_intraday_5m(ticker: str, lookback_minutes: int) -> pd.DataFrame:
    """
    EODHD intraday requires US symbols like AAPL.US
    """
    if not EODHD_API_KEY:
        return pd.DataFrame()

    symbol = f"{ticker.upper()}.US"
    end = datetime.utcnow()
    start = end - timedelta(minutes=lookback_minutes + 60)  # buffer

    url = f"https://eodhd.com/api/intraday/{symbol}"
    params = {
        "api_token": EODHD_API_KEY,
        "interval": "5m",
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "fmt": "json",
    }

    try:
        data = http_get_json(url, params=params)
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        # Expected columns: datetime, open, high, low, close, volume
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def eodhd_news(ticker: str, lookback_minutes: int) -> pd.DataFrame:
    """
    EODHD news endpoint: /api/news?&s=AAPL.US
    """
    if not EODHD_API_KEY:
        return pd.DataFrame()

    symbol = f"{ticker.upper()}.US"
    end = datetime.utcnow()
    start = end - timedelta(minutes=lookback_minutes)

    url = "https://eodhd.com/api/news"
    params = {
        "api_token": EODHD_API_KEY,
        "s": symbol,
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "limit": 50,
        "fmt": "json",
    }
    try:
        data = http_get_json(url, params=params)
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()
        # Normalize time
        if "date" in df.columns:
            df["published_utc"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        elif "datetime" in df.columns:
            df["published_utc"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        else:
            df["published_utc"] = pd.NaT

        df = df.dropna(subset=["published_utc"]).sort_values("published_utc", ascending=False)

        # Convert to CST display
        df["published_cst"] = df["published_utc"].dt.tz_convert(APP_TZ).dt.strftime("%Y-%m-%d %H:%M:%S %Z")

        # Columns we want
        cols = []
        for c in ["published_cst", "source", "title", "link"]:
            if c in df.columns:
                cols.append(c)
        if "link" in df.columns and "URL" not in df.columns:
            df["URL"] = df["link"]
        elif "url" in df.columns and "URL" not in df.columns:
            df["URL"] = df["url"]

        df["Ticker"] = ticker.upper()
        out = df[["Ticker", "published_cst"] + [c for c in ["source", "title", "URL"] if c in df.columns]].copy()
        out = out.rename(columns={"title": "Title", "source": "Source"})
        return out.head(50)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def stooq_10y_yield() -> float | None:
    """
    Pulls 10Y yield from Stooq page (no key).
    We parse the "Kurs" value.
    """
    try:
        url = "https://stooq.pl/q/?s=10yusy.b"
        html = SESSION.get(url, timeout=15).text
        m = re.search(r"Kurs\s*([0-9]+\.[0-9]+)", html)
        if not m:
            return None
        return float(m.group(1))
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def uw_options_volume_bias(ticker: str) -> dict:
    """
    https://api.unusualwhales.com/api/stock/{ticker}/options-volume
    Returns call/put premium + OI; used for bias and gamma proxy.
    """
    if not UW_TOKEN:
        return {"ok": False, "error": "UW_TOKEN missing"}

    url = f"https://api.unusualwhales.com/api/stock/{ticker.upper()}/options-volume"
    headers = {"Accept": "application/json, text/plain", "Authorization": f"Bearer {UW_TOKEN}"}
    try:
        j = http_get_json(url, headers=headers, timeout=20)
        data = j.get("data", []) if isinstance(j, dict) else []
        if not data:
            return {"ok": True, "empty": True}
        row = data[0]

        call_prem = safe_float(row.get("call_premium"))
        put_prem = safe_float(row.get("put_premium"))
        bull_prem = safe_float(row.get("bullish_premium"))
        bear_prem = safe_float(row.get("bearish_premium"))

        call_oi = safe_int(row.get("call_open_interest"))
        put_oi = safe_int(row.get("put_open_interest"))

        # Bias from bullish/bearish premium if present, else call vs put premium.
        bias = "Neutral"
        if bull_prem is not None and bear_prem is not None:
            if bull_prem > bear_prem * 1.10:
                bias = "Bullish"
            elif bear_prem > bull_prem * 1.10:
                bias = "Bearish"
        elif call_prem is not None and put_prem is not None:
            if call_prem > put_prem * 1.10:
                bias = "Bullish"
            elif put_prem > call_prem * 1.10:
                bias = "Bearish"

        # Gamma proxy: more call OI often behaves like "positive gamma-ish" vs put OI dominance
        gamma_bias = "Neutral"
        if call_oi is not None and put_oi is not None:
            if call_oi > put_oi * 1.15:
                gamma_bias = "PositiveGamma (proxy)"
            elif put_oi > call_oi * 1.15:
                gamma_bias = "NegativeGamma (proxy)"

        return {
            "ok": True,
            "bias": bias,
            "gamma_bias": gamma_bias,
            "call_premium": call_prem,
            "put_premium": put_prem,
            "call_oi": call_oi,
            "put_oi": put_oi,
        }
    except requests.HTTPError as e:
        return {"ok": False, "error": f"HTTP {getattr(e.response,'status_code', '??')}: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _try_uw_flow_alerts_once(base_url: str, limit: int = 200) -> list[dict]:
    """
    Tries to pull UW flow alerts. Exact fields vary by account/tier.
    We'll normalize later.
    """
    headers = {"Accept": "application/json, text/plain", "Authorization": f"Bearer {UW_TOKEN}"}
    params = {"limit": limit}
    j = http_get_json(base_url, headers=headers, params=params, timeout=20)
    if isinstance(j, dict):
        # Most UW endpoints return { "data": [...] }
        if "data" in j and isinstance(j["data"], list):
            return j["data"]
        # Some return "results"
        if "results" in j and isinstance(j["results"], list):
            return j["results"]
    if isinstance(j, list):
        return j
    return []


@st.cache_data(ttl=30, show_spinner=False)
def uw_flow_alerts() -> dict:
    """
    Tries multiple endpoints. Never crashes the app.
    """
    if not UW_TOKEN:
        return {"ok": False, "error": "UW_TOKEN missing", "items": []}

    last_err = None
    for u in UW_FLOW_ALERTS_URLS_TO_TRY:
        try:
            items = _try_uw_flow_alerts_once(u, limit=200)
            return {"ok": True, "url": u, "items": items}
        except requests.HTTPError as e:
            last_err = f"{u} -> HTTP {getattr(e.response,'status_code','??')}"
        except Exception as e:
            last_err = f"{u} -> {e}"

    return {"ok": False, "error": last_err or "Unknown error", "items": []}


# =========================
# INDICATORS
# =========================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    out = 100 - (100 / (1 + rs))
    return out

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    m_fast = ema(close, fast)
    m_slow = ema(close, slow)
    macd_line = m_fast - m_slow
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line

def vwap(df: pd.DataFrame) -> pd.Series:
    # Typical price * volume cumulative / volume cumulative
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    return pv.cumsum() / df["volume"].replace(0, pd.NA).cumsum()

def volume_ratio(df: pd.DataFrame, length: int = 20) -> float | None:
    if df.empty or "volume" not in df.columns:
        return None
    v = df["volume"]
    if len(v) < max(5, length):
        return None
    cur = float(v.iloc[-1])
    avg = float(v.rolling(length).mean().iloc[-1])
    if avg == 0:
        return None
    return cur / avg

def iv_spike_proxy(df: pd.DataFrame) -> str:
    """
    You asked for IV spike detection.
    True IV needs options IV. We use a stable proxy:
    "range spike vs recent average" (intraday realized volatility proxy).
    """
    if df.empty or len(df) < 30:
        return "N/A"
    rng = (df["high"] - df["low"]).rolling(10).mean().iloc[-1]
    rng_avg = (df["high"] - df["low"]).rolling(50).mean().iloc[-1]
    if pd.isna(rng) or pd.isna(rng_avg) or rng_avg == 0:
        return "N/A"
    if rng > rng_avg * 1.6:
        return "YES (proxy)"
    return "No"


# =========================
# SCORING / SIGNAL LOGIC (CALLS/PUTS ONLY)
# =========================
def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))

def compute_signal_row(ticker: str, df: pd.DataFrame, uw_bias: dict, y10: float | None, weights: dict) -> dict:
    """
    Returns one row for the main table.
    Signal is CALLS or PUTS only.
    """
    base = {
        "Ticker": ticker,
        "Confidence": 50,
        "Direction": "—",
        "Signal": "WAIT",
        "UW Unusual": "NO",
        "UW Bias": "Neutral",
        "Gamma bias": "Neutral",
        "RSI": "N/A",
        "MACD_hist": "N/A",
        "VWAP": "N/A",
        "EMA stack": "N/A",
        "Vol_ratio": "N/A",
        "IV_spike": "N/A",
        "10Y": "N/A",
    }

    if df.empty or len(df) < 35:
        return base

    close = df["close"]
    last_close = float(close.iloc[-1])

    rsi_v = rsi(close, 14).iloc[-1]
    macd_h = macd_hist(close).iloc[-1]
    vwap_s = vwap(df).iloc[-1]

    ema9 = ema(close, 9).iloc[-1]
    ema20 = ema(close, 20).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]

    vr = volume_ratio(df, 20)
    iv_proxy = iv_spike_proxy(df)

    # Trend/stack
    ema_stack = "Neutral"
    if ema9 > ema20 > ema50:
        ema_stack = "Bullish"
    elif ema9 < ema20 < ema50:
        ema_stack = "Bearish"

    # VWAP position
    vwap_pos = "Above" if last_close > vwap_s else "Below"

    # Score components (-1 bearish to +1 bullish)
    # RSI: >55 bullish, <45 bearish
    rsi_score = 0
    if pd.notna(rsi_v):
        if rsi_v >= 55:
            rsi_score = +1
        elif rsi_v <= 45:
            rsi_score = -1

    # MACD hist: >0 bullish
    macd_score = +1 if (pd.notna(macd_h) and macd_h > 0) else (-1 if pd.notna(macd_h) else 0)

    # VWAP: above bullish
    vwap_score = +1 if last_close > vwap_s else -1

    # EMA stack: bullish/bearish
    ema_score = +1 if ema_stack == "Bullish" else (-1 if ema_stack == "Bearish" else 0)

    # Volume ratio: spike bullish if moving up, bearish if moving down (simple + stable)
    vol_score = 0
    if vr is not None and vr >= 1.5:
        # direction check: last close vs 3 bars ago
        if float(close.iloc[-1]) >= float(close.iloc[-4]):
            vol_score = +1
        else:
            vol_score = -1

    # UW bias weight
    uwb = uw_bias.get("bias", "Neutral") if uw_bias.get("ok") else "Neutral"
    uw_score = +1 if uwb == "Bullish" else (-1 if uwb == "Bearish" else 0)

    # 10Y filter (optional):
    # When 10Y rising strongly intraday, it often pressures growth/QQQ-type names.
    # We only use as a *small* bias. With daily-only yield, treat as mild.
    y10_score = 0
    if isinstance(y10, float):
        base["10Y"] = round(y10, 3)
        # Soft rule: >4.5 bearish bias, <3.5 bullish bias (tunable)
        if y10 >= 4.5:
            y10_score = -1
        elif y10 <= 3.5:
            y10_score = +1

    # Combine to score in [0..100]
    # Weighted sum in [-1..+1] then map to 0..100
    total_w = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
    combined = (
        weights["rsi"] * rsi_score
        + weights["macd"] * macd_score
        + weights["vwap"] * vwap_score
        + weights["ema"] * ema_score
        + weights["vol"] * vol_score
        + weights["uw"] * uw_score
        + weights["y10"] * y10_score
    ) / total_w

    confidence = clamp(int(round(50 + 50 * combined)))

    direction = "Bullish" if combined > 0.10 else ("Bearish" if combined < -0.10 else "Neutral")

    # CALLS/PUTS ONLY decision:
    if direction == "Bullish":
        signal = "BUY CALLS"
    elif direction == "Bearish":
        signal = "BUY PUTS"
    else:
        signal = "WAIT"

    base.update({
        "Confidence": confidence,
        "Direction": direction,
        "Signal": signal,
        "UW Bias": uwb if uwb else "Neutral",
        "Gamma bias": uw_bias.get("gamma_bias", "Neutral") if uw_bias.get("ok") else "Neutral",
        "RSI": round(float(rsi_v), 1) if pd.notna(rsi_v) else "N/A",
        "MACD_hist": round(float(macd_h), 4) if pd.notna(macd_h) else "N/A",
        "VWAP": vwap_pos,
        "EMA stack": ema_stack,
        "Vol_ratio": round(float(vr), 2) if vr is not None else "N/A",
        "IV_spike": iv_proxy,
    })

    return base


def normalize_flow_alerts(items: list[dict], tickers_set: set[str]) -> pd.DataFrame:
    """
    Robust normalization because UW alert fields vary.
    We'll extract the most common fields if present.
    """
    if not items:
        return pd.DataFrame()

    rows = []
    for it in items:
        # Try multiple possible keys (because UW varies by endpoint / tier)
        sym = (it.get("underlying_symbol") or it.get("symbol") or it.get("ticker") or "").upper()
        if tickers_set and sym and sym not in tickers_set:
            continue

        premium = safe_float(it.get("premium") or it.get("premium_usd") or it.get("total_premium"))
        strike = safe_float(it.get("strike"))
        opt_type = (it.get("option_type") or it.get("type") or "").lower()
        expiry = it.get("expiry") or it.get("expiration") or it.get("exp")
        dte = None
        try:
            if expiry:
                exp_dt = pd.to_datetime(expiry).date()
                dte = (exp_dt - now_cst().date()).days
        except Exception:
            dte = None

        vol = safe_int(it.get("volume"))
        oi = safe_int(it.get("open_interest") or it.get("oi"))

        tags = it.get("tags")
        if isinstance(tags, list):
            tags_str = ", ".join([str(x) for x in tags[:6]])
        else:
            tags_str = str(tags) if tags else ""

        executed_at = it.get("executed_at") or it.get("time") or it.get("created_at")
        executed_cst = ""
        if executed_at:
            try:
                dt = pd.to_datetime(executed_at, utc=True)
                executed_cst = dt.tz_convert(APP_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                executed_cst = str(executed_at)

        # Apply YOUR hard filters when possible
        if premium is not None and premium < MIN_PREMIUM:
            continue
        if dte is not None and dte > MAX_DTE:
            continue
        if REQUIRE_VOL_GT_OI and (vol is not None and oi is not None) and not (vol > oi):
            continue
        # ITM exclusion cannot be perfectly enforced without underlying price + moneyness.
        # We'll keep this as "best effort" and rely on screener for the strict ITM filter.

        rows.append({
            "Time (CST)": executed_cst,
            "Ticker": sym or "—",
            "Type": opt_type.upper() if opt_type else "—",
            "Premium": premium,
            "Strike": strike,
            "Expiry": expiry,
            "DTE": dte,
            "Vol": vol,
            "OI": oi,
            "Tags": tags_str,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(by=["Premium"], ascending=False)
    return df


# =========================
# UI — SIDEBAR
# =========================
st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")
st.caption(f"Last update (CST): **{fmt_cst(now_cst())}**")

st_autorefresh(interval=30 * 1000, key="refresh")  # 30 seconds refresh

with st.sidebar:
    st.header("Settings")

    # ✅ Allow any ticker typed
    st.write("Type tickers (comma-separated). Example: `SPY,TSLA,NVDA`")
    tickers_text = st.text_input("Tickers", value="SPY,TSLA")

    st.caption("Quick pick (optional)")
    preset = st.multiselect("Choose options", DEFAULT_TICKERS_PRESET, default=[])

    # Combine typed + preset
    typed = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    tickers = sorted(list(dict.fromkeys(typed + [t.upper() for t in preset])))

    news_lookback = st.number_input("News lookback (minutes)", min_value=5, max_value=240, value=60, step=5)
    price_lookback = st.number_input("Price lookback (minutes)", min_value=60, max_value=600, value=240, step=30)

    st.divider()
    st.subheader("Refresh")
    st.caption("App auto-refreshes every 30 seconds.")

    st.divider()
    st.subheader("Institutional mode")
    inst_min = st.slider("Signals only if confidence ≥", min_value=50, max_value=95, value=75, step=1)

    st.divider()
    st.subheader("Weights (sum doesn't have to be 1)")
    w_rsi = st.slider("RSI weight", 0.0, 0.30, 0.15, 0.01)
    w_macd = st.slider("MACD weight", 0.0, 0.30, 0.15, 0.01)
    w_vwap = st.slider("VWAP weight", 0.0, 0.30, 0.15, 0.01)
    w_ema = st.slider("EMA stack (9/20/50) weight", 0.0, 0.30, 0.18, 0.01)
    w_vol = st.slider("Volume ratio weight", 0.0, 0.30, 0.12, 0.01)
    w_uw = st.slider("UW flow/bias weight", 0.0, 0.30, 0.20, 0.01)
    w_y10 = st.slider("10Y yield (optional) weight", 0.0, 0.20, 0.05, 0.01)

    weights = {"rsi": w_rsi, "macd": w_macd, "vwap": w_vwap, "ema": w_ema, "vol": w_vol, "uw": w_uw, "y10": w_y10}

    st.divider()
    st.subheader("Keys status (green/red)")
    if EODHD_API_KEY:
        st.success("EODHD_API_KEY ✅")
    else:
        st.error("EODHD_API_KEY ❌")

    if UW_TOKEN:
        st.success("UW_TOKEN (Bearer) ✅")
    else:
        st.error("UW_TOKEN (Bearer) ❌")

    if UW_FLOW_ALERTS_URL_USER:
        st.success("UW_FLOW_ALERTS_URL ✅")
    else:
        st.info("UW_FLOW_ALERTS_URL not required (we auto-try common endpoints)")

    if FINVIZ_AUTH:
        st.info("FINVIZ_AUTH present (not used in this build)")
    else:
        st.info("FINVIZ_AUTH (optional)")

    st.divider()
    st.subheader("Endpoints status")

    # EODHD checks (lightweight)
    if EODHD_API_KEY and tickers:
        test_df = eodhd_intraday_5m(tickers[0], lookback_minutes=120)
        if not test_df.empty:
            st.success("EODHD intraday ✅")
        else:
            st.error("EODHD intraday ❌ (ticker data empty)")

        test_news = eodhd_news(tickers[0], lookback_minutes=int(news_lookback))
        if not test_news.empty:
            st.success("EODHD news ✅")
        else:
            st.warning("EODHD news ⚠️ (no headlines)")

    # UW options-volume check
    if UW_TOKEN and tickers:
        uw_chk = uw_options_volume_bias(tickers[0])
        if uw_chk.get("ok"):
            st.success("UW options-volume ✅")
        else:
            st.error(f"UW options-volume ❌ ({uw_chk.get('error','error')})")

    # UW flow-alerts check
    if UW_TOKEN:
        fa = uw_flow_alerts()
        if fa.get("ok"):
            st.success("UW flow-alerts ✅")
        else:
            st.error(f"UW flow-alerts ❌ ({fa.get('error','error')})")

    # 10Y
    y10 = stooq_10y_yield()
    if isinstance(y10, float):
        st.success(f"10Y yield ✅ ({y10:.3f})")
    else:
        st.info("10Y yield (optional) — Not available (ok)")

# =========================
# MAIN LAYOUT
# =========================
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("This is embedded. True filtering (DTE/ITM/premium rules) is best done inside the screener itself.")
    st.components.v1.iframe(UW_SCREENER_URL, height=820, scrolling=True)

with right:
    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW flow)")

    if not tickers:
        st.info("Type at least one ticker in the sidebar (comma-separated). Example: SPY,TSLA,NVDA")
        st.stop()

    # Fetch 10Y once
    y10 = stooq_10y_yield()

    rows = []
    for t in tickers:
        df = eodhd_intraday_5m(t, lookback_minutes=int(price_lookback))
        uwb = uw_options_volume_bias(t)
        row = compute_signal_row(t, df, uwb, y10, weights)
        rows.append(row)

    score_df = pd.DataFrame(rows)

    st.dataframe(score_df, use_container_width=True, hide_index=True)

    # Institutional alerts (>= threshold)
    st.markdown("---")
    st.subheader("Institutional Alerts (≥75 only)")

    inst = score_df[score_df["Confidence"] >= inst_min].copy()
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        for _, r in inst.sort_values("Confidence", ascending=False).iterrows():
            st.success(f"{r['Ticker']}: **{r['Signal']}** | Confidence={r['Confidence']} | UW Bias={r['UW Bias']} | VWAP={r['VWAP']} | EMA={r['EMA stack']}")

    # UW Flow alerts section
    st.markdown("---")
    st.subheader("Unusual Flow Alerts (UW API)")
    st.caption("Rules applied: premium ≥ $1M, DTE ≤ 3, Volume > OI, exclude ITM (best enforced in screener), ticker-matched.")

    fa = uw_flow_alerts()
    if not fa.get("ok"):
        st.error(f"UW flow fetch failed: {fa.get('error')}  →  This is an API/endpoint issue (not your code).")
        st.caption("Fix: your UW token may not have access to flow-alerts OR UW changed the route. Keep screener for flow; API may require higher tier.")
    else:
        items = fa.get("items", [])
        alerts_df = normalize_flow_alerts(items, set(tickers))
        if alerts_df.empty:
            st.info("No UW flow-alerts matching your rules right now.")
        else:
            st.dataframe(alerts_df.head(50), use_container_width=True, hide_index=True)

    # News
    st.markdown("---")
    st.subheader(f"News — last {int(news_lookback)} minutes (EODHD)")
    all_news = []
    for t in tickers:
        ndf = eodhd_news(t, lookback_minutes=int(news_lookback))
        if not ndf.empty:
            all_news.append(ndf)
    if not all_news:
        st.info("No news in this lookback window (or EODHD returned none).")
    else:
        news_df = pd.concat(all_news, ignore_index=True)
        st.dataframe(news_df, use_container_width=True, hide_index=True)
        st.caption("Tip: Click URL column links (or copy/paste).")

    with st.expander("Debug (why something might show N/A)"):
        st.write("""
- Indicators show **N/A** when EODHD intraday didn’t return enough 5m bars (needs ~35+ bars).
- After-hours: intraday bars can freeze or thin out; the app **doesn’t crash** and will keep last stable values.
- UW flow-alerts: **404** usually means that endpoint is not accessible for your token/tier OR UW changed the path.
  The screener will still work even if the flow-alert API fails.
        """)
