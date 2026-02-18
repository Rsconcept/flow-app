import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Institutional Options Signals (5m) — CALLS / PUTS ONLY",
    layout="wide",
)

CST = ZoneInfo("America/Chicago")
ET = ZoneInfo("America/New_York")

DEFAULT_QUICK_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "AMD", "NVDA"]

# Embed UW screener (optional convenience)
UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF"
    "&limit=250&max_dte=3"
    "&min_premium=1000000"
    "&min_volume_oi_ratio=1"
    "&order=premium&order_direction=desc"
)

# ============================================================
# SECRETS / KEYS
# ============================================================
UW_TOKEN = st.secrets.get("UW_TOKEN", os.getenv("UW_TOKEN", "")).strip()
POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY", os.getenv("POLYGON_API_KEY", "")).strip()
EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")).strip()

# Optional
FRED_API_KEY = st.secrets.get("FRED_API_KEY", os.getenv("FRED_API_KEY", "")).strip()

# UW base + endpoints (per your confirmed working list)
UW_BASE = "https://api.unusualwhales.com/api"
UW_FLOW_ALERTS_PATH = "/option-trades/flow-alerts"         # ✅ FIXED
UW_TICKER_FLOW_PATH = "/stock/{ticker}/options-flow"       # ✅
UW_EARNINGS_PATH = "/stock/{ticker}/earnings-history"      # ✅
UW_CONTRACT_INTRADAY_PATH = "/option-contract/{id}/intraday"  # ✅ (if IDs available)

# ============================================================
# TIME HELPERS
# ============================================================
def now_cst() -> datetime:
    return datetime.now(tz=CST)

def now_et() -> datetime:
    return datetime.now(tz=ET)

def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)

def to_eodhd_symbol(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if not t:
        return ""
    return t if t.endswith(".US") else f"{t}.US"

def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def parse_tickers(typed: str, quick: list[str]) -> list[str]:
    typed_list = []
    if typed:
        typed_list = [p.strip().upper() for p in typed.split(",") if p.strip()]
    out = []
    for t in typed_list + [q.upper() for q in quick]:
        if t and t not in out:
            out.append(t)
    return out

# ============================================================
# INDICATORS (pandas)
# ============================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))

def macd_hist(series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line

def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["h"] + df["l"] + df["c"]) / 3.0
    pv = tp * df["v"]
    return pv.cumsum() / df["v"].cumsum().replace(0, pd.NA)

# ============================================================
# POLYGON (WORKING) — MINUTE BARS + PREV CLOSE
# ============================================================
@st.cache_data(ttl=20)
def polygon_minute_bars(ticker: str, lookback_minutes: int) -> tuple[pd.DataFrame, str]:
    """
    Uses Polygon 1-minute bars for today's date and filters last N minutes.
    Output columns: t (datetime UTC), o,h,l,c,v
    """
    if not POLYGON_API_KEY:
        return pd.DataFrame(), "missing_polygon_key"

    tkr = (ticker or "").strip().upper()
    if not tkr:
        return pd.DataFrame(), "bad_symbol"

    # Polygon range needs a date string (ET date is best)
    d = now_et().date().isoformat()
    url = f"https://api.polygon.io/v2/aggs/ticker/{tkr}/range/1/minute/{d}/{d}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}

    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame(), f"http_{r.status_code}"
        j = r.json()
        results = j.get("results", [])
        if not results:
            return pd.DataFrame(), "no_results"

        df = pd.DataFrame(results)
        # t is ms since epoch
        df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.sort_values("t").reset_index(drop=True)

        cutoff = utc_now() - timedelta(minutes=int(lookback_minutes))
        df = df[df["t"] >= cutoff].copy()

        if df.empty:
            return pd.DataFrame(), "no_bars_in_window"

        return df, "ok"
    except Exception as e:
        return pd.DataFrame(), f"error_{type(e).__name__}"

@st.cache_data(ttl=60)
def polygon_prev_close(ticker: str) -> tuple[float | None, str]:
    if not POLYGON_API_KEY:
        return None, "missing_polygon_key"
    tkr = (ticker or "").strip().upper()
    url = f"https://api.polygon.io/v2/aggs/ticker/{tkr}/prev"
    params = {"adjusted": "true", "apiKey": POLYGON_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        j = r.json()
        res = j.get("results", [])
        if not res:
            return None, "no_results"
        return safe_float(res[0].get("c")), "ok"
    except Exception as e:
        return None, f"error_{type(e).__name__}"

# ============================================================
# EODHD — NEWS + OPTIONS SNAPSHOT (current IV only)
# ============================================================
@st.cache_data(ttl=60)
def eodhd_news(ticker: str, lookback_minutes: int) -> tuple[pd.DataFrame, str]:
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_eodhd_key"

    sym = to_eodhd_symbol(ticker)
    url = "https://eodhd.com/api/news"
    params = {"s": sym, "offset": 0, "limit": 50, "api_token": EODHD_API_KEY, "fmt": "json"}

    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame(), f"http_{r.status_code}"
        items = r.json()
        if not isinstance(items, list) or len(items) == 0:
            return pd.DataFrame(), "no_headlines"

        df = pd.DataFrame(items)
        if "date" not in df.columns:
            return pd.DataFrame(), "no_date_field"

        df["published_utc"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        cutoff = utc_now() - timedelta(minutes=int(lookback_minutes))
        df = df[df["published_utc"] >= cutoff].copy()

        if df.empty:
            return pd.DataFrame(), "no_recent_news"

        out = pd.DataFrame({
            "Ticker": ticker.upper(),
            "Published (CST)": df["published_utc"].dt.tz_convert(CST).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Title": df["title"] if "title" in df.columns else "",
            "URL": df["link"] if "link" in df.columns else "",
            "Source": df["source"] if "source" in df.columns else "",
        }).dropna(subset=["Title"])

        return out.reset_index(drop=True), "ok"
    except Exception as e:
        return pd.DataFrame(), f"error_{type(e).__name__}"

@st.cache_data(ttl=120)
def eodhd_options_chain_iv(ticker: str) -> tuple[float | None, str]:
    """
    EODHD options chain is working, but it's a snapshot (current IV only).
    We'll take a simple average IV across near-term strikes if available.
    """
    if not EODHD_API_KEY:
        return None, "missing_eodhd_key"

    sym = to_eodhd_symbol(ticker)
    url = f"https://eodhd.com/api/options/{sym}"
    params = {"api_token": EODHD_API_KEY, "fmt": "json"}

    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        j = r.json()

        # EODHD structure can vary; we do best-effort scan
        ivs = []

        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "impliedVolatility":
                        fv = safe_float(v)
                        if fv is not None:
                            ivs.append(fv)
                    else:
                        walk(v)
            elif isinstance(obj, list):
                for it in obj:
                    walk(it)

        walk(j)

        if not ivs:
            return None, "no_iv_found"

        # impliedVolatility might be decimal (0.25) or percent-ish; normalize as percent
        avg = sum(ivs) / len(ivs)
        if avg <= 2.0:
            avg *= 100.0

        return float(avg), "ok"
    except Exception as e:
        return None, f"error_{type(e).__name__}"

# ============================================================
# UNUSUAL WHALES (WORKING) — FLOW ALERTS + TICKER FLOW
# ============================================================
def uw_headers():
    return {"Accept": "application/json, text/plain", "Authorization": f"Bearer {UW_TOKEN}"}

@st.cache_data(ttl=20)
def uw_flow_alerts(limit: int = 200) -> tuple[pd.DataFrame, str]:
    if not UW_TOKEN:
        return pd.DataFrame(), "missing_uw_token"

    url = f"{UW_BASE}{UW_FLOW_ALERTS_PATH}"
    try:
        r = requests.get(url, headers=uw_headers(), params={"limit": int(limit)}, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame(), f"http_{r.status_code}"
        j = r.json()
        rows = j.get("data") if isinstance(j, dict) else None
        if not rows or not isinstance(rows, list):
            return pd.DataFrame(), "no_data"
        df = pd.DataFrame(rows)
        # common timestamp fields
        for c in ["executed_at", "created_at", "time", "timestamp"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        return df, "ok"
    except Exception as e:
        return pd.DataFrame(), f"error_{type(e).__name__}"

@st.cache_data(ttl=60)
def uw_ticker_flow(ticker: str) -> tuple[dict, str]:
    if not UW_TOKEN:
        return {}, "missing_uw_token"
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return {}, "bad_symbol"

    url = f"{UW_BASE}{UW_TICKER_FLOW_PATH.format(ticker=tkr)}"
    try:
        r = requests.get(url, headers=uw_headers(), timeout=20)
        if r.status_code != 200:
            return {}, f"http_{r.status_code}"
        j = r.json()
        data = j.get("data") if isinstance(j, dict) else None
        # some endpoints return dict, some list
        if data is None:
            return {}, "no_data"
        return data, "ok"
    except Exception as e:
        return {}, f"error_{type(e).__name__}"

@st.cache_data(ttl=60)
def uw_contract_intraday(contract_id: str) -> tuple[pd.DataFrame, str]:
    """
    Optional: used for IV spike detection if contract IDs exist in flow alerts.
    """
    if not UW_TOKEN:
        return pd.DataFrame(), "missing_uw_token"
    cid = str(contract_id).strip()
    if not cid:
        return pd.DataFrame(), "bad_id"

    url = f"{UW_BASE}{UW_CONTRACT_INTRADAY_PATH.format(id=cid)}"
    try:
        r = requests.get(url, headers=uw_headers(), timeout=20)
        if r.status_code != 200:
            return pd.DataFrame(), f"http_{r.status_code}"
        j = r.json()
        rows = j.get("data") if isinstance(j, dict) else None
        if not rows or not isinstance(rows, list):
            return pd.DataFrame(), "no_data"
        df = pd.DataFrame(rows)
        if "start_time" in df.columns:
            df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
        return df, "ok"
    except Exception as e:
        return pd.DataFrame(), f"error_{type(e).__name__}"

# ============================================================
# 10Y (OPTIONAL) — FRED DGS10
# ============================================================
@st.cache_data(ttl=300)
def fred_10y() -> tuple[float | None, str]:
    if not FRED_API_KEY:
        return None, "missing_fred_key"
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "series_id": "DGS10",
            "sort_order": "desc",
            "limit": 1,
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        j = r.json()
        obs = j.get("observations", [])
        if not obs:
            return None, "no_data"
        val = safe_float(obs[0].get("value"))
        return val, "ok"
    except Exception as e:
        return None, f"error_{type(e).__name__}"

# ============================================================
# SCORING — CALLS / PUTS ONLY
# ============================================================
def score_from_factors(factors: dict, weights: dict) -> float:
    total_w = sum(max(0.0, w) for w in weights.values()) or 1.0
    s = 0.0
    for k, w in weights.items():
        s += (w / total_w) * float(factors.get(k, 0.0))
    return clamp(50.0 + 50.0 * s, 0.0, 100.0)

def pick_signal(score: float, threshold: float) -> str:
    if score >= threshold:
        return "BUY CALLS"
    if score <= (100 - threshold):
        return "BUY PUTS"
    return "WAIT"

# ============================================================
# BUILD ROW PER TICKER
# ============================================================
def compute_indicators_from_polygon(df: pd.DataFrame) -> dict:
    """
    df columns: t,o,h,l,c,v
    returns last values for RSI, MACD_hist, VWAP_above, EMA_stack, Vol_ratio
    """
    if df is None or df.empty or len(df) < 30:
        return {
            "RSI": "N/A",
            "MACD_hist": "N/A",
            "VWAP_above": "N/A",
            "EMA_stack": "N/A",
            "Vol_ratio": "N/A",
            "Bars": 0,
            "Last bar (CST)": "N/A",
        }

    d = df.copy().sort_values("t").reset_index(drop=True)

    close = d["c"]
    d["vwap"] = vwap(d)
    d["ema9"] = ema(close, 9)
    d["ema20"] = ema(close, 20)
    d["ema50"] = ema(close, 50)
    d["rsi14"] = rsi(close, 14)
    d["macd_hist"] = macd_hist(close)

    last = d.iloc[-1]
    rsi_val = safe_float(last.get("rsi14"))
    macd_h = safe_float(last.get("macd_hist"))
    vwap_val = safe_float(last.get("vwap"))
    last_close = safe_float(last.get("c"))

    ema9v = safe_float(last.get("ema9"))
    ema20v = safe_float(last.get("ema20"))
    ema50v = safe_float(last.get("ema50"))

    if ema9v is not None and ema20v is not None and ema50v is not None:
        ema_stack = "Bullish" if (ema9v > ema20v > ema50v) else ("Bearish" if (ema9v < ema20v < ema50v) else "Neutral")
    else:
        ema_stack = "N/A"

    vwap_above = "N/A"
    if last_close is not None and vwap_val is not None:
        vwap_above = bool(last_close > vwap_val)

    # Volume ratio: last vol / avg last 30
    last_vol = safe_float(last.get("v"), 0.0) or 0.0
    avg_vol = safe_float(d["v"].tail(30).mean(), 0.0) or 0.0
    vol_ratio = (last_vol / avg_vol) if avg_vol > 0 else None

    last_bar_cst = last["t"].tz_convert(CST).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "RSI": round(rsi_val, 2) if rsi_val is not None else "N/A",
        "MACD_hist": round(macd_h, 4) if macd_h is not None else "N/A",
        "VWAP_above": vwap_above,
        "EMA_stack": ema_stack,
        "Vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else "N/A",
        "Bars": int(len(d)),
        "Last bar (CST)": last_bar_cst,
    }

def uw_bias_from_ticker_flow(flow_data: dict) -> tuple[str, str]:
    """
    We do best-effort. Since schemas vary, we look for obvious call/put premium or volume fields.
    Returns (bias, put_call_str)
    """
    if not flow_data:
        return "N/A", "N/A"

    # Try common patterns
    # If list -> maybe first item is summary
    if isinstance(flow_data, list) and flow_data:
        flow_data = flow_data[0]

    call_p = safe_float(flow_data.get("call_premium")) or safe_float(flow_data.get("bullish_premium"))
    put_p = safe_float(flow_data.get("put_premium")) or safe_float(flow_data.get("bearish_premium"))
    call_v = safe_float(flow_data.get("call_volume"))
    put_v = safe_float(flow_data.get("put_volume"))

    put_call = "N/A"
    if put_v is not None and call_v is not None:
        put_call = f"{put_v:.0f}/{call_v:.0f}"

    if call_p is None or put_p is None:
        return "N/A", put_call

    if call_p > put_p:
        return "Bullish", put_call
    if put_p > call_p:
        return "Bearish", put_call
    return "Neutral", put_call

def iv_spike_from_contract_intraday(cid: str, minutes: int = 30) -> tuple[str, str]:
    """
    Uses UW contract intraday if we have an ID.
    Spike rule: (iv_high - iv_low) in the last window >= 0.03 (3 vol points in decimal terms)
    """
    df, stt = uw_contract_intraday(cid)
    if stt != "ok" or df.empty:
        return "N/A", stt

    if "start_time" not in df.columns:
        return "N/A", "no_start_time"

    cutoff = utc_now() - timedelta(minutes=minutes)
    d = df[df["start_time"] >= cutoff].copy()
    if d.empty:
        return "N/A", "no_recent_intraday"

    ivh = safe_float(d["iv_high"].max()) if "iv_high" in d.columns else None
    ivl = safe_float(d["iv_low"].min()) if "iv_low" in d.columns else None
    if ivh is None or ivl is None:
        return "N/A", "no_iv_fields"

    diff = ivh - ivl
    spike = diff >= 0.03
    # show as percent points
    return ("YES" if spike else "NO"), f"ivΔ={diff*100:.2f}pp"

def gamma_bias_proxy_from_flowalerts(df_alerts: pd.DataFrame, ticker: str) -> str:
    """
    If flow alerts have call/put premium for the ticker in the window, infer gamma bias.
    This is a proxy: net call premium -> Positive, net put premium -> Negative.
    """
    if df_alerts is None or df_alerts.empty:
        return "N/A"
    if "underlying_symbol" not in df_alerts.columns:
        return "N/A"

    d = df_alerts[df_alerts["underlying_symbol"].astype(str).str.upper() == ticker.upper()].copy()
    if d.empty:
        return "N/A"

    # Identify call/put type fields
    type_col = None
    for c in ["option_type", "type", "put_call", "side"]:
        if c in d.columns:
            type_col = c
            break

    prem_col = None
    for c in ["premium", "total_premium", "notional"]:
        if c in d.columns:
            prem_col = c
            break

    if type_col is None or prem_col is None:
        return "N/A"

    d[prem_col] = pd.to_numeric(d[prem_col], errors="coerce")
    d = d.dropna(subset=[prem_col])

    if d.empty:
        return "N/A"

    calls = d[d[type_col].astype(str).str.lower().str.contains("call")][prem_col].sum()
    puts = d[d[type_col].astype(str).str.lower().str.contains("put")][prem_col].sum()

    if calls > puts:
        return "Positive (proxy)"
    if puts > calls:
        return "Negative (proxy)"
    return "Neutral"

# ============================================================
# UI — SIDEBAR
# ============================================================
st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")
st.caption(f"Last update (CST): **{now_cst().strftime('%Y-%m-%d %H:%M:%S %Z')}**")

with st.sidebar:
    st.header("Settings")

    typed = st.text_input("Type any tickers (comma-separated)", value="SPY")
    quick = st.multiselect("Quick pick (optional)", DEFAULT_QUICK_TICKERS, default=[])

    tickers = parse_tickers(typed, quick)

    st.divider()
    news_lookback = st.number_input("News lookback (minutes)", min_value=5, max_value=360, value=60, step=5)
    price_lookback = st.number_input("Price lookback (minutes)", min_value=30, max_value=780, value=240, step=30)

    st.divider()
    refresh_s = st.slider("Auto-refresh (seconds)", min_value=15, max_value=300, value=30, step=5)

    st.divider()
    st.subheader("Institutional mode")
    inst_threshold = st.slider("Signals only if confidence ≥", 50, 95, 75, 1)

    st.divider()
    st.subheader("Weights (sum doesn't have to be 1)")
    w_rsi = st.slider("RSI weight", 0.0, 0.6, 0.15, 0.01)
    w_macd = st.slider("MACD weight", 0.0, 0.6, 0.15, 0.01)
    w_vwap = st.slider("VWAP weight", 0.0, 0.6, 0.15, 0.01)
    w_ema  = st.slider("EMA stack (9/20/50) weight", 0.0, 0.6, 0.18, 0.01)
    w_vol  = st.slider("Volume ratio weight", 0.0, 0.6, 0.12, 0.01)
    w_uw   = st.slider("UW flow bias weight", 0.0, 0.9, 0.20, 0.01)
    w_news = st.slider("News weight (placeholder)", 0.0, 0.3, 0.05, 0.01)
    w_10y  = st.slider("10Y yield (optional) weight", 0.0, 0.3, 0.05, 0.01)

weights = {
    "rsi": w_rsi,
    "macd": w_macd,
    "vwap": w_vwap,
    "ema": w_ema,
    "vol": w_vol,
    "uw": w_uw,
    "news": w_news,
    "teny": w_10y,
}

st_autorefresh(interval=int(refresh_s) * 1000, key="auto_refresh")

# ============================================================
# STATUS PANELS (keys + endpoints)
# ============================================================
def chip(ok: bool, label: str):
    if ok:
        st.sidebar.success(label)
    else:
        st.sidebar.error(label)

with st.sidebar:
    st.divider()
    st.subheader("Keys status (green/red)")
    chip(bool(UW_TOKEN), "UW_TOKEN")
    chip(bool(POLYGON_API_KEY), "POLYGON_API_KEY")
    chip(bool(EODHD_API_KEY), "EODHD_API_KEY")
    if FRED_API_KEY:
        st.sidebar.success("FRED_API_KEY (10Y live)")
    else:
        st.sidebar.info("10Y yield: add FRED_API_KEY to enable")

with st.sidebar:
    st.divider()
    st.subheader("Endpoints status")

    # Polygon test
    if POLYGON_API_KEY and tickers:
        _, pst = polygon_minute_bars(tickers[0], 60)
        if pst == "ok":
            st.sidebar.success("Polygon minute bars ✅")
        else:
            st.sidebar.warning(f"Polygon minute bars ⚠️ ({pst})")
    else:
        st.sidebar.info("Polygon minute bars — waiting")

    # UW flow alerts test
    if UW_TOKEN:
        _, ust = uw_flow_alerts(50)
        if ust == "ok":
            st.sidebar.success("UW flow-alerts ✅")
        else:
            st.sidebar.warning(f"UW flow-alerts ⚠️ ({ust})")
    else:
        st.sidebar.info("UW flow-alerts — waiting")

    # UW ticker flow test
    if UW_TOKEN and tickers:
        _, tst = uw_ticker_flow(tickers[0])
        if tst == "ok":
            st.sidebar.success("UW ticker options-flow ✅")
        else:
            st.sidebar.warning(f"UW ticker options-flow ⚠️ ({tst})")
    else:
        st.sidebar.info("UW ticker options-flow — waiting")

    # EODHD news test
    if EODHD_API_KEY and tickers:
        _, nst = eodhd_news(tickers[0], 60)
        if nst in ("ok", "no_recent_news", "no_headlines"):
            st.sidebar.success(f"EODHD news ✅ ({nst})")
        else:
            st.sidebar.warning(f"EODHD news ⚠️ ({nst})")
    else:
        st.sidebar.info("EODHD news — waiting")

# ============================================================
# MAIN LAYOUT
# ============================================================
left, right = st.columns([1.35, 1.0], gap="large")

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("If the embed errors, it’s UW’s front-end/CSP — your app still works.")
    st.components.v1.iframe(UW_SCREENER_URL, height=820, scrolling=True)

with right:
    st.subheader("Live Score / Signals (Polygon price + EODHD headlines + UW flow)")

    ten_y, ten_y_status = fred_10y()
    if ten_y_status == "ok":
        st.caption(f"10Y yield (FRED): **{ten_y:.2f}%**")
    else:
        st.caption(f"10Y yield: **N/A** ({ten_y_status})")

    # Pull UW flow alerts once (for gamma proxy + IV attempt)
    flow_df, flow_status = uw_flow_alerts(limit=200)

    rows = []
    all_news = []

    for tkr in tickers:
        # --- Price / Indicators from Polygon
        bars, bars_status = polygon_minute_bars(tkr, int(price_lookback))
        ind = compute_indicators_from_polygon(bars)

        # --- News (EODHD)
        news_df, news_status = eodhd_news(tkr, int(news_lookback))
        if isinstance(news_df, pd.DataFrame) and not news_df.empty:
            all_news.append(news_df)

        # --- UW ticker flow bias
        tf, tf_status = uw_ticker_flow(tkr)
        uw_bias, put_call = uw_bias_from_ticker_flow(tf) if tf_status == "ok" else ("N/A", "N/A")

        # --- Gamma bias proxy (from flow alerts)
        gamma_bias = gamma_bias_proxy_from_flowalerts(flow_df, tkr) if flow_status == "ok" else "N/A"

        # --- IV snapshot (EODHD chain) + IV spike (UW contract intraday if IDs exist)
        iv_now, iv_now_status = eodhd_options_chain_iv(tkr)
        iv_spike = "N/A"
        iv_spike_detail = ""

        # Find a contract id from flow alerts for this ticker (best-effort)
        contract_id = None
        if flow_status == "ok" and "underlying_symbol" in flow_df.columns:
            dsub = flow_df[flow_df["underlying_symbol"].astype(str).str.upper() == tkr.upper()].copy()
            # look for any ID field
            for c in ["option_chain_id", "option_contract_id", "contract_id", "id"]:
                if c in dsub.columns and len(dsub) > 0:
                    contract_id = dsub[c].dropna().astype(str).iloc[0] if len(dsub[c].dropna()) else None
                    if contract_id:
                        break

        if contract_id:
            iv_spike, iv_spike_detail = iv_spike_from_contract_intraday(contract_id, minutes=30)

        # --- Volume ratio factor
        vol_ratio_val = ind["Vol_ratio"] if isinstance(ind["Vol_ratio"], (int, float)) else None

        # ====================================================
        # FACTORS in [-1..+1]
        # ====================================================
        factors = {}

        # RSI factor
        rsi_v = ind["RSI"] if isinstance(ind["RSI"], (int, float)) else None
        factors["rsi"] = clamp(((rsi_v - 50) / 25), -1, 1) if rsi_v is not None else 0.0

        # MACD factor
        macd_v = ind["MACD_hist"] if isinstance(ind["MACD_hist"], (int, float)) else None
        factors["macd"] = clamp((macd_v / 0.5), -1, 1) if macd_v is not None else 0.0

        # VWAP factor
        if ind["VWAP_above"] is True:
            factors["vwap"] = 0.7
        elif ind["VWAP_above"] is False:
            factors["vwap"] = -0.7
        else:
            factors["vwap"] = 0.0

        # EMA stack
        if ind["EMA_stack"] == "Bullish":
            factors["ema"] = 0.7
        elif ind["EMA_stack"] == "Bearish":
            factors["ema"] = -0.7
        else:
            factors["ema"] = 0.0

        # Volume ratio
        if vol_ratio_val is None:
            factors["vol"] = 0.0
        else:
            factors["vol"] = 0.6 if vol_ratio_val >= 2.0 else (0.3 if vol_ratio_val >= 1.2 else 0.0)

        # UW bias
        if uw_bias == "Bullish":
            factors["uw"] = 0.8
        elif uw_bias == "Bearish":
            factors["uw"] = -0.8
        else:
            factors["uw"] = 0.0

        # News placeholder (no NLP guessing)
        factors["news"] = 0.0

        # 10Y nudge
        if ten_y is None:
            factors["teny"] = 0.0
        else:
            factors["teny"] = -0.3 if ten_y >= 4.5 else (0.3 if ten_y <= 3.5 else 0.0)

        score = score_from_factors(
            {"rsi": factors["rsi"], "macd": factors["macd"], "vwap": factors["vwap"], "ema": factors["ema"],
             "vol": factors["vol"], "uw": factors["uw"], "news": factors["news"], "teny": factors["teny"]},
            weights={"rsi": weights["rsi"], "macd": weights["macd"], "vwap": weights["vwap"], "ema": weights["ema"],
                     "vol": weights["vol"], "uw": weights["uw"], "news": weights["news"], "teny": weights["teny"]},
        )

        direction = "Bullish" if score > 55 else ("Bearish" if score < 45 else "Neutral")
        signal = pick_signal(score, inst_threshold)

        rows.append({
            "Ticker": tkr.upper(),
            "Confidence": round(score, 1),
            "Direction": direction,
            "Signal": signal,
            "UW_bias": uw_bias,
            "Gamma_bias": gamma_bias,
            "RSI": ind["RSI"],
            "MACD_hist": ind["MACD_hist"],
            "VWAP_above": ind["VWAP_above"],
            "EMA_stack": ind["EMA_stack"],
            "Vol_ratio": ind["Vol_ratio"],
            "Put/Call vol": put_call,
            "IV_now": (round(iv_now, 2) if isinstance(iv_now, (int, float)) else "N/A"),
            "IV_spike": (iv_spike if iv_spike_detail == "" else f"{iv_spike} ({iv_spike_detail})"),
            "Bars": ind["Bars"],
            "Last bar (CST)": ind["Last bar (CST)"],
            "Bars_status": bars_status,
            "News_status": news_status,
            "UW_flow_status": flow_status,
            "UW_tickerflow_status": tf_status,
        })

    live_df = pd.DataFrame(rows)
    st.dataframe(live_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Institutional Alerts (≥ threshold only)")

    hits = live_df[(live_df["Confidence"] >= inst_threshold) & (live_df["Signal"].isin(["BUY CALLS", "BUY PUTS"]))].copy()
    if hits.empty:
        st.info("No institutional signals right now.")
    else:
        for _, r in hits.sort_values("Confidence", ascending=False).iterrows():
            st.success(
                f"{r['Ticker']}: {r['Signal']} | Confidence={r['Confidence']} | UW_bias={r['UW_bias']} | Gamma={r['Gamma_bias']} | VolRatio={r['Vol_ratio']}"
            )

    st.divider()
    st.subheader("Unusual Flow Alerts (UW API) — filtered")

    if flow_status != "ok":
        st.error(f"Flow alerts unavailable: {flow_status}")
    else:
        df = flow_df.copy()

        # Normalize likely fields
        # These names vary by plan/schema — we apply only if present
        for col in ["premium", "open_interest", "volume", "dte"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Filter by ticker list if field exists
        if "underlying_symbol" in df.columns:
            df["underlying_symbol"] = df["underlying_symbol"].astype(str).str.upper()
            df = df[df["underlying_symbol"].isin([t.upper() for t in tickers])]

        # Premium >= 1M
        if "premium" in df.columns:
            df = df[df["premium"] >= 1_000_000]

        # DTE <= 3
        if "dte" in df.columns:
            df = df[df["dte"] <= 3]

        # Volume > OI
        if "volume" in df.columns and "open_interest" in df.columns:
            df = df[df["volume"] > df["open_interest"]]

        # Exclude ITM (best-effort)
        if "is_itm" in df.columns:
            df = df[df["is_itm"] == False]

        if df.empty:
            st.info("No flow alerts matching your rules in the current window.")
        else:
            show_cols = []
            for c in [
                "underlying_symbol", "option_type", "premium", "volume", "open_interest", "dte",
                "strike", "expiration", "implied_volatility", "gamma", "delta",
                "option_chain_id", "option_contract_id", "contract_id",
                "executed_at", "created_at", "time", "timestamp"
            ]:
                if c in df.columns:
                    show_cols.append(c)

            df_show = df[show_cols].copy()

            # convert any datetime col to CST display
            for c in ["executed_at", "created_at", "time", "timestamp"]:
                if c in df_show.columns:
                    df_show[c] = pd.to_datetime(df_show[c], errors="coerce", utc=True).dt.tz_convert(CST).dt.strftime("%Y-%m-%d %H:%M:%S")

            st.dataframe(df_show.head(50), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(f"News — last {int(news_lookback)} minutes (EODHD)")
    if not all_news:
        st.info("No news in this lookback window (or EODHD returned none).")
    else:
        news_all = pd.concat(all_news, ignore_index=True)
        st.dataframe(news_all, use_container_width=True, hide_index=True)

    with st.expander("Why you might see N/A / None (plain English)"):
        st.write(
            "- If **RSI/MACD/VWAP/EMA** are N/A, it means Polygon didn’t return minute bars for your window (market closed or ticker invalid).\n"
            "- If **Flow alerts** show http_401/403, your token doesn’t have access. If http_422, the endpoint expects different parameters or your plan restricts the data.\n"
            "- EODHD options chain is only a **snapshot IV**, not IV history (so IV spike is best from UW contract intraday)."
        )
