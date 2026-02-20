# app.py
# Institutional Options Signals (5m) — CALLS / PUTS ONLY
# Uses: EODHD intraday + EODHD news + EODHD options chain + Unusual Whales flow alerts + FRED 10Y
# After-hours safe: will show "empty" instead of crashing.

import os
import time
import math
import json
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


# =========================
# Config
# =========================
APP_TZ = ZoneInfo("America/Chicago")  # CST/CDT automatically
EODHD_BASE = "https://eodhd.com/api"
UW_BASE = "https://api.unusualwhales.com/api"
FRED_BASE = "https://api.stlouisfed.org/fred"

DEFAULT_TICKERS = "SPY,TSLA,NVDA,QQQ,IWM,DIA,AMD,META"
DEFAULT_NEWS_LOOKBACK_MIN = 60
DEFAULT_PRICE_LOOKBACK_MIN = 240
DEFAULT_REFRESH_SEC = 30

# UW filtering rules you requested
MIN_PREMIUM_USD = 1_000_000
MAX_DTE_DAYS = 3


# =========================
# Helpers
# =========================
def now_cst() -> datetime:
    return datetime.now(tz=APP_TZ)

def to_unix_seconds(dt: datetime) -> int:
    return int(dt.timestamp())

def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("none", "nan", "null"):
            return None
        return float(s)
    except Exception:
        return None

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def is_market_hours_cst(dt: datetime) -> bool:
    # US equities: 8:30–15:00 CST (9:30–16:00 ET) Mon–Fri
    if dt.weekday() >= 5:
        return False
    open_cst = dt.replace(hour=8, minute=30, second=0, microsecond=0)
    close_cst = dt.replace(hour=15, minute=0, second=0, microsecond=0)
    return open_cst <= dt <= close_cst

def parse_tickers(text: str) -> list[str]:
    # Allow ANY ticker typed by user
    raw = (text or "").upper().replace(";", ",").replace("\n", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    # de-dup preserving order
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def normalize_iv(iv_value):
    """
    Fix insane IV numbers.
    - Some feeds return 0.3154 (31.54%)
    - Some return 31.54 (already %)
    - Some bad parses produce 3154 or 3719.49 -> treat as percent*100
    """
    iv = safe_float(iv_value)
    if iv is None:
        return None
    # If it's clearly a fraction
    if 0 < iv < 3:
        return iv * 100.0
    # If it's already a reasonable percent
    if 3 <= iv <= 250:
        return iv
    # If it's huge, assume it's percent * 100
    if iv > 250:
        return iv / 100.0
    return None


# =========================
# HTTP wrappers
# =========================
@dataclass
class EndpointResult:
    ok: bool
    status: str
    data: any = None
    error: str = ""

def http_get_json(url: str, headers=None, timeout=20) -> EndpointResult:
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        status = f"http_{r.status_code}"
        if r.status_code != 200:
            # try to show server message (often helpful)
            msg = ""
            try:
                msg = r.text[:500]
            except Exception:
                pass
            return EndpointResult(False, status, None, msg)
        try:
            return EndpointResult(True, "ok", r.json(), "")
        except Exception as e:
            return EndpointResult(False, "parse_error", None, f"{e} | body[:200]={r.text[:200]}")
    except Exception as e:
        return EndpointResult(False, "request_error", None, str(e))


# =========================
# Data: EODHD
# =========================
@st.cache_data(ttl=25)
def eodhd_intraday_bars(ticker: str, lookback_minutes: int, interval: str, eodhd_key: str) -> EndpointResult:
    """
    EODHD intraday endpoint expects UNIX from/to (numbers) -> fixes your 422.
    """
    if not eodhd_key:
        return EndpointResult(False, "missing_key", None, "EODHD_API_KEY missing")

    t = ticker.upper().strip()
    dt_to = now_cst()
    dt_from = dt_to - timedelta(minutes=int(lookback_minutes))

    url = (
        f"{EODHD_BASE}/intraday/{t}.US"
        f"?api_token={eodhd_key}&fmt=json"
        f"&interval={interval}"
        f"&from={to_unix_seconds(dt_from)}"
        f"&to={to_unix_seconds(dt_to)}"
    )

    res = http_get_json(url)
    if not res.ok:
        return res

    rows = res.data if isinstance(res.data, list) else []
    if not rows:
        return EndpointResult(True, "empty", [], "No intraday bars returned for this window")

    df = pd.DataFrame(rows)
    # EODHD commonly returns: datetime, open, high, low, close, volume
    # Normalize datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_convert(APP_TZ)
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(APP_TZ)

    # Ensure numeric
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    return EndpointResult(True, "ok", df, "")


@st.cache_data(ttl=60)
def eodhd_news(ticker: str, lookback_minutes: int, eodhd_key: str) -> EndpointResult:
    if not eodhd_key:
        return EndpointResult(False, "missing_key", None, "EODHD_API_KEY missing")

    t = ticker.upper().strip()
    # EODHD news endpoint varies by plan; this one is commonly used:
    # /news?s=TSLA.US&offset=0&limit=50
    url = f"{EODHD_BASE}/news?s={t}.US&offset=0&limit=50&api_token={eodhd_key}&fmt=json"
    res = http_get_json(url)
    if not res.ok:
        return res

    items = res.data if isinstance(res.data, list) else []
    if not items:
        return EndpointResult(True, "ok", pd.DataFrame(), "")

    df = pd.DataFrame(items)

    # Try to standardize columns (EODHD sometimes uses different keys)
    col_map = {
        "date": "published",
        "datetime": "published",
        "publishedAt": "published",
        "title": "title",
        "source": "source",
        "link": "url",
        "url": "url",
    }
    for k, v in col_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    if "published" in df.columns:
        dt = pd.to_datetime(df["published"], utc=True, errors="coerce")
        df["published_cst"] = dt.dt.tz_convert(APP_TZ)
    else:
        df["published_cst"] = pd.NaT

    cutoff = now_cst() - timedelta(minutes=int(lookback_minutes))
    df = df[df["published_cst"].isna() | (df["published_cst"] >= cutoff)]
    df = df.sort_values("published_cst", ascending=False)

    return EndpointResult(True, "ok", df, "")


@st.cache_data(ttl=60)
def eodhd_options_chain(ticker: str, eodhd_key: str) -> EndpointResult:
    """
    Gives IV_now (single snapshot) + Put/Call volume + Put/Call OI for expiries within MAX_DTE_DAYS.
    """
    if not eodhd_key:
        return EndpointResult(False, "missing_key", None, "EODHD_API_KEY missing")

    t = ticker.upper().strip()
    url = f"{EODHD_BASE}/options/{t}.US?api_token={eodhd_key}&fmt=json"
    res = http_get_json(url)
    if not res.ok:
        return res

    data = res.data
    if not isinstance(data, dict):
        return EndpointResult(False, "parse_error", None, "Options chain not a dict")

    # EODHD returns dict keyed by expiry date often
    rows = []
    for exp, contracts in data.items():
        if isinstance(contracts, list):
            for c in contracts:
                c2 = dict(c)
                c2["expiration"] = exp
                rows.append(c2)

    if not rows:
        return EndpointResult(True, "ok", pd.DataFrame(), "")

    df = pd.DataFrame(rows)

    # Standardize
    if "type" not in df.columns and "optionType" in df.columns:
        df["type"] = df["optionType"]

    for c in ["openInterest", "volume", "strike", "impliedVolatility"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce").dt.date
    today = now_cst().date()
    df["dte"] = df["expiration"].apply(lambda d: (d - today).days if pd.notna(d) else None)

    # Keep <= MAX_DTE_DAYS (your rule)
    df = df[(df["dte"].notna()) & (df["dte"] >= 0) & (df["dte"] <= MAX_DTE_DAYS)]

    return EndpointResult(True, "ok", df, "")


# =========================
# Data: Unusual Whales
# =========================
@st.cache_data(ttl=10)
def uw_flow_alerts(limit: int, uw_token: str) -> EndpointResult:
    if not uw_token:
        return EndpointResult(False, "missing_key", None, "UW_TOKEN missing")

    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {uw_token}",
    }
    # per your spec:
    # GET /option-trades/flow-alerts?limit=100
    url = f"{UW_BASE}/option-trades/flow-alerts?limit={int(limit)}"
    res = http_get_json(url, headers=headers)
    if not res.ok:
        return res

    payload = res.data
    items = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return EndpointResult(False, "parse_error", None, "UW flow alerts not a list")

    df = pd.DataFrame(items)
    return EndpointResult(True, "ok", df, "")


# =========================
# Indicators
# =========================
def calc_vwap(df: pd.DataFrame) -> float | None:
    if df is None or df.empty:
        return None
    if not all(c in df.columns for c in ["close", "volume"]):
        return None
    pv = (df["close"] * df["volume"]).sum()
    vv = df["volume"].sum()
    if vv == 0:
        return None
    return float(pv / vv)

def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(close: pd.Series, period: int = 14) -> float | None:
    if close is None or len(close) < period + 2:
        return None
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down.replace(0, math.nan)
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    return None if pd.isna(v) else float(v)

def calc_macd_hist(close: pd.Series) -> float | None:
    if close is None or len(close) < 35:
        return None
    ema12 = calc_ema(close, 12)
    ema26 = calc_ema(close, 26)
    macd = ema12 - ema26
    signal = calc_ema(macd, 9)
    hist = macd - signal
    v = hist.iloc[-1]
    return None if pd.isna(v) else float(v)

def calc_vol_ratio(df: pd.DataFrame, window: int = 20) -> float | None:
    if df is None or df.empty or "volume" not in df.columns or len(df) < window + 1:
        return None
    cur = df["volume"].iloc[-1]
    avg = df["volume"].iloc[-window:].mean()
    if avg == 0 or pd.isna(avg) or pd.isna(cur):
        return None
    return float(cur / avg)

def ema_stack_bias(df: pd.DataFrame) -> str | None:
    if df is None or df.empty or "close" not in df.columns or len(df) < 60:
        return None
    close = df["close"]
    e9 = calc_ema(close, 9).iloc[-1]
    e20 = calc_ema(close, 20).iloc[-1]
    e50 = calc_ema(close, 50).iloc[-1]
    if any(pd.isna(x) for x in [e9, e20, e50]):
        return None
    if e9 > e20 > e50:
        return "Bullish"
    if e9 < e20 < e50:
        return "Bearish"
    return "Neutral"


# =========================
# 10Y Yield (FRED)
# =========================
@st.cache_data(ttl=3600)
def fred_10y_latest(fred_key: str) -> EndpointResult:
    if not fred_key:
        return EndpointResult(False, "missing_key", None, "FRED_API_KEY missing")
    # DGS10 is daily; latest observation used as a *macro tilt*, not tick-by-tick
    url = (
        f"{FRED_BASE}/series/observations"
        f"?series_id=DGS10&api_key={fred_key}&file_type=json&sort_order=desc&limit=2"
    )
    res = http_get_json(url)
    if not res.ok:
        return res
    obs = res.data.get("observations", []) if isinstance(res.data, dict) else []
    if not obs:
        return EndpointResult(True, "empty", None, "No FRED observations")
    latest = safe_float(obs[0].get("value"))
    prev = safe_float(obs[1].get("value")) if len(obs) > 1 else None
    return EndpointResult(True, "ok", {"latest": latest, "prev": prev}, "")


# =========================
# Scoring (CALLS / PUTS only)
# =========================
def build_signal_and_confidence(
    rsi, macd_hist, vwap_above, ema_bias, vol_ratio,
    uw_bias, put_call_vol, iv_spike, ten_y_delta
):
    """
    0–100 confidence.
    Outputs:
      direction: Bullish/Bearish/Neutral
      signal: BUY CALLS / BUY PUTS / WAIT
    """
    score = 50.0

    # RSI
    if rsi is not None:
        if rsi < 35:
            score += 6
        elif rsi > 70:
            score -= 6

    # MACD hist
    if macd_hist is not None:
        score += 8 if macd_hist > 0 else -8

    # VWAP
    if vwap_above is not None:
        score += 8 if vwap_above else -8

    # EMA stack
    if ema_bias == "Bullish":
        score += 10
    elif ema_bias == "Bearish":
        score -= 10

    # Volume ratio (momentum confirmation)
    if vol_ratio is not None:
        if vol_ratio >= 1.8:
            score += 6
        elif vol_ratio <= 0.7:
            score -= 3

    # UW bias (flow)
    if uw_bias == "Bullish":
        score += 12
    elif uw_bias == "Bearish":
        score -= 12

    # Put/Call volume tilt from options chain
    if put_call_vol is not None:
        if put_call_vol >= 1.3:
            score -= 6
        elif put_call_vol <= 0.77:
            score += 6

    # IV spike: higher IV usually helps puts more (crash/fear), but can also mean “premium expensive”.
    # We use it as *risk tilt*; if spike, reduce confidence slightly.
    if iv_spike is True:
        score -= 3

    # 10Y delta (macro tilt): rising yields often pressure growth/tech intraday; falling yields supportive.
    if ten_y_delta is not None:
        if ten_y_delta >= 0.05:
            score -= 4
        elif ten_y_delta <= -0.05:
            score += 4

    score = clamp(score, 0, 100)

    # direction & signal
    if score >= 55:
        direction = "Bullish"
        signal = "BUY CALLS"
    elif score <= 45:
        direction = "Bearish"
        signal = "BUY PUTS"
    else:
        direction = "Neutral"
        signal = "WAIT"

    return score, direction, signal


# =========================
# UI
# =========================
st.set_page_config(page_title="Institutional Options Signals", layout="wide")

st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")
st.caption(f"Last update (CST): {now_cst().strftime('%Y-%m-%d %H:%M:%S %Z')}")

# ---- Secrets
UW_TOKEN = st.secrets.get("UW_TOKEN", os.getenv("UW_TOKEN", "")).strip()
EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")).strip()
FRED_API_KEY = st.secrets.get("FRED_API_KEY", os.getenv("FRED_API_KEY", "")).strip()
FINVIZ_AUTH = st.secrets.get("FINVIZ_AUTH", os.getenv("FINVIZ_AUTH", "")).strip()  # optional, not used

# ---- Sidebar settings
with st.sidebar:
    st.header("Settings")

    tickers_text = st.text_input(
        "Type any tickers (comma-separated).",
        value=DEFAULT_TICKERS,
        help="Example: SPY,TSLA,NVDA (you can type ANY ticker)"
    )
    tickers = parse_tickers(tickers_text)
    if not tickers:
        st.warning("Please enter at least one ticker (e.g., SPY).")
        st.stop()

    selected = st.multiselect("Track tickers", options=tickers, default=tickers[:3])

    st.divider()
    news_lookback = st.number_input("News lookback (minutes)", min_value=10, max_value=360, value=DEFAULT_NEWS_LOOKBACK_MIN, step=10)
    price_lookback = st.number_input("Price lookback (minutes)", min_value=60, max_value=2000, value=DEFAULT_PRICE_LOOKBACK_MIN, step=30)
    interval = st.selectbox("Intraday interval", ["5m", "1m", "15m"], index=0)

    st.divider()
    refresh_sec = st.slider("Auto-refresh (seconds)", min_value=10, max_value=120, value=DEFAULT_REFRESH_SEC, step=5)
    institutional_threshold = st.slider("Institutional mode: signals only if confidence ≥", 50, 95, 75, 1)

    st.divider()
    st.subheader("Weights (sum doesn’t have to be 1)")
    w_rsi = st.slider("RSI weight", 0.0, 0.30, 0.15, 0.01)
    w_macd = st.slider("MACD weight", 0.0, 0.30, 0.15, 0.01)
    w_vwap = st.slider("VWAP weight", 0.0, 0.30, 0.15, 0.01)
    w_ema = st.slider("EMA stack (9/20/50) weight", 0.0, 0.40, 0.18, 0.01)
    w_vol = st.slider("Volume ratio weight", 0.0, 0.30, 0.12, 0.01)
    w_uw = st.slider("UW flow weight", 0.0, 0.40, 0.20, 0.01)
    w_news = st.slider("News weight (placeholder)", 0.0, 0.20, 0.05, 0.01)
    w_10y = st.slider("10Y yield (optional) weight", 0.0, 0.20, 0.05, 0.01)

    st.divider()
    st.subheader("Keys status (green/red)")
    def key_box(name, ok):
        st.success(name) if ok else st.error(name)

    key_box("UW_TOKEN", bool(UW_TOKEN))
    key_box("EODHD_API_KEY", bool(EODHD_API_KEY))
    key_box("FRED_API_KEY (10Y live)", bool(FRED_API_KEY))
    st.info("FINVIZ_AUTH not required (optional).")


# ---- Auto refresh (safe)
# Streamlit Cloud sometimes doesn’t have streamlit_autorefresh. We do a simple timer rerun.
if "last_rerun" not in st.session_state:
    st.session_state.last_rerun = time.time()

if time.time() - st.session_state.last_rerun >= refresh_sec:
    st.session_state.last_rerun = time.time()
    st.rerun()


# =========================
# Layout columns
# =========================
left, right = st.columns([1.45, 1.0], gap="large")

# ---- LEFT: UW Screener embed
with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("Embedded. Best filtering (DTE/ITM/premium) is still done in the UW screener UI.")
    # Embed UW screener
    components.iframe("https://unusualwhales.com/flow", height=720, scrolling=True)

# ---- RIGHT: status + signals + alerts + news
with right:
    st.subheader("Endpoints status")

    # FRED 10Y
    ten = fred_10y_latest(FRED_API_KEY)
    ten_latest = ten.data.get("latest") if ten.ok and isinstance(ten.data, dict) else None
    ten_prev = ten.data.get("prev") if ten.ok and isinstance(ten.data, dict) else None
    ten_delta = (ten_latest - ten_prev) if (ten_latest is not None and ten_prev is not None) else None

    # UW flow alerts
    uw = uw_flow_alerts(limit=120, uw_token=UW_TOKEN)

    # Endpoint cards
    def endpoint_card(label, res: EndpointResult):
        if res.ok and res.status == "ok":
            st.success(f"{label} (ok)")
        elif res.ok and res.status == "empty":
            st.warning(f"{label} (empty) — {res.error or 'No data returned'}")
        else:
            st.error(f"{label} ({res.status}) — {res.error[:140]}")

    # We show statuses *overall* (not per ticker)
    # EODHD bars status depends on first selected ticker only (for quick health check)
    bars_health = eodhd_intraday_bars(selected[0], price_lookback, interval, EODHD_API_KEY)
    endpoint_card("EODHD intraday bars", bars_health)

    endpoint_card("UW flow-alerts", uw)

    news_health = eodhd_news(selected[0], news_lookback, EODHD_API_KEY)
    endpoint_card("EODHD news", news_health)

    if ten.ok and ten.status == "ok":
        st.success("FRED 10Y yield (ok)")
    else:
        st.warning(f"FRED 10Y yield ({ten.status}) — {ten.error[:120]}")

    st.divider()

    # =========================
    # Signals table
    # =========================
    st.subheader("Live Score / Signals (EODHD intraday + EODHD headlines + UW flow)")

    # Prepare UW alerts dataframe
    uw_df = uw.data if (uw.ok and isinstance(uw.data, pd.DataFrame)) else pd.DataFrame()

    # Build filtered UW alerts by ticker (best-effort; depends on UW fields available)
    def filter_uw_by_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        t = ticker.upper()

        out = df.copy()

        # Try common field names
        underlying_cols = [c for c in out.columns if c.lower() in ("ticker", "underlying", "underlying_symbol", "symbol")]
        if underlying_cols:
            out = out[out[underlying_cols[0]].astype(str).str.upper().eq(t)]

        # Premium filter
        prem_cols = [c for c in out.columns if "premium" in c.lower()]
        if prem_cols:
            prem = pd.to_numeric(out[prem_cols[0]], errors="coerce")
            out = out[prem >= MIN_PREMIUM_USD]

        return out

    rows = []
    for tkr in selected:
        bars_res = eodhd_intraday_bars(tkr, price_lookback, interval, EODHD_API_KEY)
        bars_df = bars_res.data if (bars_res.ok and isinstance(bars_res.data, pd.DataFrame)) else pd.DataFrame()

        rsi = calc_rsi(bars_df["close"]) if not bars_df.empty and "close" in bars_df.columns else None
        macd_hist = calc_macd_hist(bars_df["close"]) if not bars_df.empty and "close" in bars_df.columns else None
        vwap = calc_vwap(bars_df) if not bars_df.empty else None
        last_close = float(bars_df["close"].iloc[-1]) if (not bars_df.empty and "close" in bars_df.columns) else None
        vwap_above = (last_close > vwap) if (last_close is not None and vwap is not None) else None
        ema_bias = ema_stack_bias(bars_df) if not bars_df.empty else None
        vol_ratio = calc_vol_ratio(bars_df) if not bars_df.empty else None

        # Options chain: Put/Call volume ratio + IV_now
        chain_res = eodhd_options_chain(tkr, EODHD_API_KEY)
        chain_df = chain_res.data if (chain_res.ok and isinstance(chain_res.data, pd.DataFrame)) else pd.DataFrame()

        put_call_vol = None
        iv_now = None
        if not chain_df.empty:
            # Put/Call vol
            if "type" in chain_df.columns and "volume" in chain_df.columns:
                puts = chain_df[chain_df["type"].astype(str).str.lower().str.contains("put", na=False)]["volume"].sum()
                calls = chain_df[chain_df["type"].astype(str).str.lower().str.contains("call", na=False)]["volume"].sum()
                if calls and calls > 0:
                    put_call_vol = float(puts / calls)

            # IV_now: median impliedVolatility
            if "impliedVolatility" in chain_df.columns:
                iv_med = chain_df["impliedVolatility"].median()
                iv_now = normalize_iv(iv_med)

        # UW bias from flow alerts (best-effort)
        uw_t = filter_uw_by_ticker(uw_df, tkr)
        uw_bias = None
        uw_unusual = "NO"
        if not uw_t.empty:
            uw_unusual = "YES"
            # Try to infer call/put dominance
            type_cols = [c for c in uw_t.columns if "type" in c.lower() or "side" in c.lower()]
            prem_cols = [c for c in uw_t.columns if "premium" in c.lower()]
            if type_cols and prem_cols:
                typ = uw_t[type_cols[0]].astype(str).str.lower()
                prem = pd.to_numeric(uw_t[prem_cols[0]], errors="coerce").fillna(0)
                call_p = prem[typ.str.contains("call", na=False)].sum()
                put_p = prem[typ.str.contains("put", na=False)].sum()
                if call_p > put_p * 1.1:
                    uw_bias = "Bullish"
                elif put_p > call_p * 1.1:
                    uw_bias = "Bearish"
                else:
                    uw_bias = "Neutral"

        # IV spike (we only flag if IV_now is present and very high vs a rough threshold)
        iv_spike = None
        if iv_now is not None:
            iv_spike = True if iv_now >= 65 else False

        # Weighted score: we’ll map weights into our scoring by scaling inputs
        # (Still outputs 0–100 and CALLS/PUTS/WAIT)
        score, direction, signal = build_signal_and_confidence(
            rsi=rsi,
            macd_hist=macd_hist,
            vwap_above=vwap_above,
            ema_bias=ema_bias,
            vol_ratio=vol_ratio,
            uw_bias=uw_bias,
            put_call_vol=put_call_vol,
            iv_spike=iv_spike,
            ten_y_delta=ten_delta,
        )

        institutional = "YES" if score >= institutional_threshold else "NO"

        rows.append({
            "Ticker": tkr,
            "Confidence": round(score, 1),
            "Direction": direction,
            "Signal": signal,
            "Institutional": institutional,
            "RSI": None if rsi is None else round(rsi, 2),
            "MACD_hist": None if macd_hist is None else round(macd_hist, 4),
            "VWAP_above": None if vwap_above is None else ("Above" if vwap_above else "Below"),
            "EMA_stack": ema_bias,
            "Vol_ratio": None if vol_ratio is None else round(vol_ratio, 2),
            "UW_unusual": uw_unusual,
            "UW_bias": uw_bias if uw_bias else "N/A",
            "Put/Call_vol": None if put_call_vol is None else round(put_call_vol, 2),
            "IV_now(%)": None if iv_now is None else round(iv_now, 2),
            "IV_spike": None if iv_spike is None else ("YES" if iv_spike else "NO"),
            "10Y": None if ten_latest is None else round(ten_latest, 2),
            "Bars": int(len(bars_df)) if isinstance(bars_df, pd.DataFrame) else 0,
            "Last_bar(CST)": None if bars_df is None or bars_df.empty else str(bars_df["datetime"].iloc[-1]),
            "Bars_status": bars_res.status if isinstance(bars_res, EndpointResult) else "N/A",
            "News_status": "ok" if news_health.ok else news_health.status,
            "UW_flow_status": "ok" if uw.ok else uw.status,
        })

    signals_df = pd.DataFrame(rows)
    st.dataframe(signals_df, use_container_width=True, height=220)

    st.subheader("Institutional Alerts (≥ threshold only)")
    inst = signals_df[signals_df["Confidence"] >= institutional_threshold].copy()
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        st.success("Signals meeting institutional threshold:")
        st.dataframe(inst[["Ticker", "Confidence", "Direction", "Signal", "UW_unusual", "UW_bias", "IV_now(%)", "10Y"]], use_container_width=True)

    st.subheader("Unusual Flow Alerts (UW API) — filtered")
    if not uw_df.empty:
        # Show latest filtered alerts for selected tickers (best effort)
        show_rows = []
        for tkr in selected:
            sub = filter_uw_by_ticker(uw_df, tkr)
            if not sub.empty:
                show_rows.append(sub.head(40))
        if show_rows:
            out = pd.concat(show_rows, ignore_index=True)
            st.dataframe(out.head(60), use_container_width=True, height=220)
        else:
            st.info("No UW alerts matching current filters/tickers (or fields not available).")
    else:
        st.info("UW flow alerts empty or unavailable.")

    st.subheader(f"News — last {int(news_lookback)} minutes (EODHD)")
    news_res = eodhd_news(selected[0], news_lookback, EODHD_API_KEY)
    if news_res.ok and isinstance(news_res.data, pd.DataFrame) and not news_res.data.empty:
        news_df = news_res.data.copy()

        # Safe display columns (avoid KeyError)
        cols = []
        for c in ["published_cst", "source", "title", "url"]:
            if c in news_df.columns:
                cols.append(c)

        if not cols:
            st.info("News returned, but fields are not in expected format for display.")
            st.write(news_df.head(10))
        else:
            news_show = news_df[cols].head(40)
            st.dataframe(news_show, use_container_width=True, height=220)
            st.caption("Tip: Click URL column links (or copy/paste).")
    elif news_res.ok and news_res.status == "ok":
        st.info("No news in this lookback window (or EODHD returned none).")
    else:
        st.error(f"News error: {news_res.status} — {news_res.error[:200]}")

    with st.expander("What None/N/A/empty means (plain English)"):
        st.write("""
- **empty** (intraday bars): the API is working, but returned no bars for your time window (common after-hours).
- **N/A**: we couldn’t compute the indicator because we didn’t have enough bars or the field wasn’t in the response.
- **http_4xx / http_5xx**: the server rejected the call (bad key, plan restriction, or endpoint changed).
""")


# =========================
# Important note: filters you asked for
# =========================
st.caption(
    "Filters requested: premium ≥ $1M, DTE ≤ 3 days, stocks/ETFs only, Volume > OI, exclude ITM. "
    "UW filtering is best enforced inside UW screener; API filtering is best-effort based on returned fields."
)
