import os
import json
import math
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

# ============================================================
# App config
# ============================================================
st.set_page_config(
    page_title="Institutional Options Signals (5m) — CALLS / PUTS ONLY",
    layout="wide",
)

APP_TITLE = "🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY"
CST = dt.timezone(dt.timedelta(hours=-6))  # display only
UTC = dt.timezone.utc
ET = ZoneInfo("America/New_York")

# ============================================================
# Secrets / env
# ============================================================
def get_secret(name: str) -> Optional[str]:
    # Streamlit Cloud secrets
    try:
        if name in st.secrets:
            v = st.secrets.get(name)
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass
    # Fallback to env
    v = os.environ.get(name)
    return v.strip() if isinstance(v, str) and v.strip() else None


POLYGON_API_KEY = get_secret("POLYGON_API_KEY")     # REQUIRED for intraday bars
UW_TOKEN = get_secret("UW_TOKEN")                   # REQUIRED for Unusual Whales flow alerts
EODHD_API_KEY = get_secret("EODHD_API_KEY")         # OPTIONAL (used for IV + news)
FRED_API_KEY = get_secret("FRED_API_KEY")           # OPTIONAL (10Y)


# ============================================================
# HTTP helpers
# ============================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "streamlit-options-signals/1.0"})

def http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 20
) -> Tuple[int, str, str]:
    """
    Returns: (status_code, text, content_type)
    """
    try:
        resp = SESSION.get(url, headers=headers, params=params, timeout=timeout)
        ct = resp.headers.get("Content-Type", "")
        return resp.status_code, resp.text, ct
    except requests.RequestException as e:
        return 0, str(e), ""


def safe_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


# ============================================================
# Time helpers
# ============================================================
def now_cst() -> dt.datetime:
    return dt.datetime.now(tz=CST)

def fmt_cst(ts: Optional[dt.datetime]) -> str:
    if not ts:
        return "N/A"
    try:
        return ts.astimezone(CST).strftime("%Y-%m-%d %H:%M:%S CST")
    except Exception:
        return "N/A"


# ============================================================
# Indicators
# bars df expected columns: datetime (UTC), open, high, low, close, volume
# ============================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))

def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line

def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    denom = df["volume"].replace(0, pd.NA).cumsum()
    return pv.cumsum() / denom

def volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    if len(df) < max(lookback, 2):
        return float("nan")
    last = df["volume"].iloc[-1]
    avg = df["volume"].iloc[-lookback:].mean()
    if avg == 0 or pd.isna(avg):
        return float("nan")
    return float(last / avg)


# ============================================================
# Polygon intraday bars (REPLACES EODHD intraday)
# ============================================================
POLYGON_BASE = "https://api.polygon.io"

@st.cache_data(ttl=15, show_spinner=False)
def polygon_intraday_bars(
    ticker: str,
    interval: str,
    lookback_minutes: int,
    include_extended: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Returns df with columns: datetime (UTC), open, high, low, close, volume
    interval: "1m", "5m", "15m"
    """
    if not POLYGON_API_KEY:
        return pd.DataFrame(), "missing_key"

    ticker = ticker.upper().strip()

    if not interval.endswith("m"):
        return pd.DataFrame(), "bad_interval"
    try:
        multiplier = int(interval.replace("m", ""))
    except Exception:
        return pd.DataFrame(), "bad_interval"

    timespan = "minute"

    end_utc = dt.datetime.now(tz=UTC)
    start_utc = end_utc - dt.timedelta(minutes=lookback_minutes)

    # Use ET dates for Polygon range path
    from_date = start_utc.astimezone(ET).date().isoformat()
    to_date = end_utc.astimezone(ET).date().isoformat()

    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {
        "apiKey": POLYGON_API_KEY,
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
    }

    code, text, _ = http_get(url, params=params, timeout=20)
    if code != 200:
        j = safe_json(text)
        if isinstance(j, dict) and j.get("error"):
            return pd.DataFrame(), f"http_{code} — {j.get('error')}"
        return pd.DataFrame(), f"http_{code}"

    j = safe_json(text)
    if not isinstance(j, dict) or "results" not in j:
        if isinstance(j, dict) and j.get("error"):
            return pd.DataFrame(), f"parse_error — {j.get('error')}"
        return pd.DataFrame(), "parse_error"

    results = j.get("results") or []
    if not results:
        return pd.DataFrame(), "empty"

    df = pd.DataFrame(results)

    # Polygon fields: t (ms), o,h,l,c,v
    needed = ["t", "o", "h", "l", "c", "v"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame(), "schema_mismatch"

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    out["open"] = pd.to_numeric(df["o"], errors="coerce")
    out["high"] = pd.to_numeric(df["h"], errors="coerce")
    out["low"] = pd.to_numeric(df["l"], errors="coerce")
    out["close"] = pd.to_numeric(df["c"], errors="coerce")
    out["volume"] = pd.to_numeric(df["v"], errors="coerce")
    out = out.dropna(subset=["datetime", "close"]).sort_values("datetime")

    if out.empty:
        return pd.DataFrame(), "empty"

    # Optional: regular session only (9:30–16:00 ET)
    if not include_extended:
        et_times = out["datetime"].dt.tz_convert(ET)
        regular = (
            ((et_times.dt.hour > 9) | ((et_times.dt.hour == 9) & (et_times.dt.minute >= 30)))
            & (et_times.dt.hour < 16)
        )
        out = out[regular].copy()
        if out.empty:
            return pd.DataFrame(), "empty_regular_only"

    # Clip to exact lookback; if it empties (weekend), keep tail
    cutoff = end_utc - dt.timedelta(minutes=lookback_minutes)
    clipped = out[out["datetime"] >= cutoff].copy()
    if not clipped.empty:
        out = clipped
    else:
        out = out.tail(250).copy()

    return out, "ok"


# ============================================================
# EODHD news + options IV (optional)
# ============================================================
EODHD_BASE = "https://eodhd.com/api"

@st.cache_data(ttl=60, show_spinner=False)
def eodhd_news(ticker: str, lookback_minutes: int) -> Tuple[pd.DataFrame, str]:
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"

    ticker = ticker.upper().strip()
    symbol = f"{ticker}.US"
    url = f"{EODHD_BASE}/news"

    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "s": symbol,
        "limit": 50,
    }
    code, text, _ = http_get(url, params=params)
    if code != 200:
        return pd.DataFrame(), f"http_{code}"

    j = safe_json(text)
    if not isinstance(j, list):
        return pd.DataFrame(), "parse_error"

    df = pd.DataFrame(j)
    if df.empty:
        return pd.DataFrame(), "ok"

    # Normalize timestamp
    if "date" in df.columns:
        df["published_utc"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    elif "published_at" in df.columns:
        df["published_utc"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    else:
        df["published_utc"] = pd.NaT

    df["published_cst"] = df["published_utc"].dt.tz_convert(CST)

    cutoff = now_cst() - dt.timedelta(minutes=lookback_minutes)
    df = df[df["published_cst"] >= cutoff].copy()

    def pick_col(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    title_c = pick_col("title", "text")
    src_c = pick_col("source")
    url_c = pick_col("link", "url")

    out = pd.DataFrame()
    out["ticker"] = ticker
    out["published_cst"] = df["published_cst"].dt.strftime("%Y-%m-%d %H:%M:%S CST")
    out["source"] = df[src_c] if src_c else ""
    out["title"] = df[title_c] if title_c else ""
    out["url"] = df[url_c] if url_c else ""

    out = out.dropna(subset=["title"]).head(40)
    return out, "ok"


@st.cache_data(ttl=120, show_spinner=False)
def eodhd_options_chain_iv(ticker: str) -> Tuple[Optional[float], str]:
    """
    Pulls a best-effort "current IV" point by walking the EODHD options JSON
    and taking median impliedVolatility found.
    Returns IV% (e.g., 31.54) or None.
    """
    if not EODHD_API_KEY:
        return None, "missing_key"

    ticker = ticker.upper().strip()
    url = f"{EODHD_BASE}/options/{ticker}.US"
    params = {"api_token": EODHD_API_KEY, "fmt": "json"}

    code, text, _ = http_get(url, params=params)
    if code != 200:
        return None, f"http_{code}"

    j = safe_json(text)
    if not isinstance(j, dict):
        return None, "parse_error"

    iv_vals: List[float] = []

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "impliedVolatility":
                    try:
                        iv_vals.append(float(v))
                    except Exception:
                        pass
                else:
                    walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    walk(j)

    if not iv_vals:
        return None, "no_iv"

    iv = float(pd.Series(iv_vals).dropna().median())

    # Scaling fix: 0.31 -> 31%
    if iv > 200:
        return None, "iv_bad_scale"
    iv_pct = iv * 100.0 if iv <= 2.0 else iv
    return round(iv_pct, 2), "ok"


# ============================================================
# FRED 10Y yield (optional)
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def fred_10y_yield() -> Tuple[Optional[float], str]:
    if not FRED_API_KEY:
        return None, "missing_key"

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "series_id": "DGS10",
        "sort_order": "desc",
        "limit": 1,
    }
    code, text, _ = http_get(url, params=params)
    if code != 200:
        return None, f"http_{code}"

    j = safe_json(text)
    if not isinstance(j, dict) or "observations" not in j:
        return None, "parse_error"

    obs = j.get("observations") or []
    if not obs:
        return None, "empty"

    v = obs[0].get("value")
    try:
        return float(v), "ok"
    except Exception:
        return None, "parse_error"


# ============================================================
# Unusual Whales flow alerts
# ============================================================
UW_BASE = "https://api.unusualwhales.com/api"

def uw_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}" if UW_TOKEN else "",
    }

@st.cache_data(ttl=20, show_spinner=False)
def uw_flow_alerts(limit: int = 120) -> Tuple[pd.DataFrame, str]:
    if not UW_TOKEN:
        return pd.DataFrame(), "missing_key"

    url = f"{UW_BASE}/option-trades/flow-alerts"
    params = {"limit": limit}

    code, text, _ = http_get(url, headers=uw_headers(), params=params)
    if code != 200:
        return pd.DataFrame(), f"http_{code}"

    j = safe_json(text)
    if not isinstance(j, dict) or "data" not in j:
        return pd.DataFrame(), "parse_error"

    data = j.get("data", [])
    if not isinstance(data, list) or not data:
        return pd.DataFrame(), "ok"

    return pd.DataFrame(data), "ok"


def uw_put_call_bias(flow_df: pd.DataFrame, ticker: str) -> Tuple[Optional[float], str]:
    """
    Returns put/call volume ratio from UW flow alerts (proxy).
    """
    if flow_df is None or flow_df.empty:
        return None, "N/A"

    t = ticker.upper()
    df = flow_df.copy()
    if "underlying_symbol" in df.columns:
        df = df[df["underlying_symbol"].astype(str).str.upper() == t]

    if df.empty or "option_type" not in df.columns:
        return None, "N/A"

    if "volume" in df.columns:
        vols = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    elif "size" in df.columns:
        vols = pd.to_numeric(df["size"], errors="coerce").fillna(0)
    else:
        vols = pd.Series([1.0] * len(df))

    types = df["option_type"].astype(str).str.lower()
    put_vol = float(vols[types == "put"].sum())
    call_vol = float(vols[types == "call"].sum())
    if call_vol <= 0:
        return None, "N/A"
    return round(put_vol / call_vol, 2), "ok"


def gamma_bias_proxy(flow_df: pd.DataFrame, ticker: str) -> str:
    """
    Proxy gamma bias using UW flow alerts:
      sums gamma * size * sign(call=+ / put=-)
    """
    if flow_df is None or flow_df.empty:
        return "N/A"

    t = ticker.upper()
    df = flow_df.copy()
    if "underlying_symbol" in df.columns:
        df = df[df["underlying_symbol"].astype(str).str.upper() == t]
    if df.empty:
        return "N/A"

    def to_f(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    gamma = df["gamma"].apply(to_f) if "gamma" in df.columns else pd.Series([0.0] * len(df))
    size = df["size"].apply(to_f) if "size" in df.columns else pd.Series([1.0] * len(df))
    opt_type = df["option_type"].astype(str).str.lower() if "option_type" in df.columns else pd.Series([""] * len(df))

    sign = opt_type.map(lambda x: 1.0 if x == "call" else (-1.0 if x == "put" else 0.0))
    score = (gamma.fillna(0) * size.fillna(1) * sign.fillna(0)).sum()

    if abs(score) < 0.5:
        return "Neutral"
    return "Positive" if score > 0 else "Negative"


# ============================================================
# Signal scoring (0-100) — CALLS / PUTS ONLY
# ============================================================
def score_signal(
    df_bars: pd.DataFrame,
    flow_df: pd.DataFrame,
    ticker: str,
    iv_now: Optional[float],
    ten_y: Optional[float],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ticker": ticker,
        "confidence": 50,
        "direction": "—",
        "signal": "WAIT",
        "institutional": "NO",
        "RSI": "N/A",
        "MACD_hist": "N/A",
        "VWAP_above": "N/A",
        "EMA_stack": "N/A",
        "Vol_ratio": "N/A",
        "UW_bias": "N/A",
        "Put/Call_vol": "N/A",
        "IV_now": iv_now if iv_now is not None else "N/A",
        "IV_spike": "N/A",
        "Gamma_bias": "N/A",
        "10Y": ten_y if ten_y is not None else "N/A",
        "Bars": 0,
        "Last_bar(CST)": "N/A",
        "Bars_status": "empty",
        "News_status": "N/A",
        "UW_flow_status": "N/A",
    }

    if df_bars is None or df_bars.empty:
        out["Bars_status"] = "empty"
        return out
    if len(df_bars) < 30:
        out["Bars"] = int(len(df_bars))
        out["Bars_status"] = f"too_few_bars({len(df_bars)})"
        out["Last_bar(CST)"] = fmt_cst(df_bars["datetime"].iloc[-1].to_pydatetime()) if "datetime" in df_bars.columns else "N/A"
        return out

    out["Bars"] = int(len(df_bars))
    out["Bars_status"] = "ok"
    out["Last_bar(CST)"] = fmt_cst(df_bars["datetime"].iloc[-1].to_pydatetime())

    close = df_bars["close"].astype(float)

    rsi_v = rsi(close, 14).iloc[-1]
    macd_v = macd_hist(close).iloc[-1]
    vwap_line = vwap(df_bars)
    vwap_above = bool(close.iloc[-1] >= vwap_line.iloc[-1])

    ema9 = float(ema(close, 9).iloc[-1])
    ema20 = float(ema(close, 20).iloc[-1])
    ema50 = float(ema(close, 50).iloc[-1])
    ema_stack_bull = bool(ema9 > ema20 > ema50)
    ema_stack_bear = bool(ema9 < ema20 < ema50)

    vr = volume_ratio(df_bars, 20)

    out["RSI"] = round(float(rsi_v), 2) if not pd.isna(rsi_v) else "N/A"
    out["MACD_hist"] = round(float(macd_v), 4) if not pd.isna(macd_v) else "N/A"
    out["VWAP_above"] = "Above" if vwap_above else "Below"
    out["EMA_stack"] = "Bull" if ema_stack_bull else ("Bear" if ema_stack_bear else "Neutral")
    out["Vol_ratio"] = round(float(vr), 2) if not (pd.isna(vr) or math.isinf(vr)) else "N/A"

    # UW-based biases
    pc_ratio, _ = uw_put_call_bias(flow_df, ticker)
    out["Put/Call_vol"] = pc_ratio if pc_ratio is not None else "N/A"
    out["UW_bias"] = (
        "PUT" if (pc_ratio is not None and pc_ratio > 1.1)
        else ("CALL" if (pc_ratio is not None and pc_ratio < 0.9) else "Neutral")
    )
    out["Gamma_bias"] = gamma_bias_proxy(flow_df, ticker)

    # IV spike heuristic
    if iv_now is None:
        out["IV_spike"] = "N/A"
    else:
        out["IV_spike"] = "YES" if iv_now >= 65 else "NO"

    bull = 0.0
    bear = 0.0

    # RSI
    if not pd.isna(rsi_v):
        if rsi_v <= 30:
            bull += weights["rsi"]
        elif rsi_v >= 70:
            bear += weights["rsi"]

    # MACD hist
    if not pd.isna(macd_v):
        if macd_v > 0:
            bull += weights["macd"]
        elif macd_v < 0:
            bear += weights["macd"]

    # VWAP
    if vwap_above:
        bull += weights["vwap"]
    else:
        bear += weights["vwap"]

    # EMA stack
    if ema_stack_bull:
        bull += weights["ema"]
    elif ema_stack_bear:
        bear += weights["ema"]

    # Volume confirmation
    if isinstance(out["Vol_ratio"], (int, float)) and out["Vol_ratio"] != "N/A":
        if out["Vol_ratio"] >= 1.5:
            if not pd.isna(macd_v) and macd_v > 0:
                bull += weights["vol"]
            elif not pd.isna(macd_v) and macd_v < 0:
                bear += weights["vol"]

    # UW bias
    if out["UW_bias"] == "CALL":
        bull += weights["uw"]
    elif out["UW_bias"] == "PUT":
        bear += weights["uw"]

    # 10Y filter (simple)
    if ten_y is not None:
        if ten_y >= 4.75:
            bear += weights["teny"]
        elif ten_y <= 4.0:
            bull += weights["teny"]

    total = bull + bear
    if total <= 0:
        conf = 50
    else:
        edge = abs(bull - bear) / total
        conf = int(round(50 + 50 * edge))

    out["confidence"] = conf

    if bull > bear:
        out["direction"] = "BULLISH"
        out["signal"] = "BUY CALLS" if conf >= 55 else "WAIT"
    elif bear > bull:
        out["direction"] = "BEARISH"
        out["signal"] = "BUY PUTS" if conf >= 55 else "WAIT"
    else:
        out["direction"] = "NEUTRAL"
        out["signal"] = "WAIT"

    return out


# ============================================================
# UI
# ============================================================
st.title(APP_TITLE)

with st.sidebar:
    st.header("Settings")

    ticker_text = st.text_input(
        "Type any tickers (comma-separated).",
        value="SPY,TSLA",
        help="Example: SPY,TSLA,NVDA,QQQ,IWM",
    )
    tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:20]

    interval = st.selectbox("Candle interval", ["15m", "5m", "1m"], index=1)
    price_lookback = st.slider("Price lookback (minutes)", 60, 1980, 240, 30)

    include_extended = st.toggle("Include pre/after-hours (Polygon)", value=True)
    news_lookback = st.slider("News lookback (minutes)", 15, 360, 60, 15)

    refresh_sec = st.slider("Auto-refresh (seconds)", 10, 120, 30, 5)

    st.divider()
    inst_threshold = st.slider("Institutional mode: signals only if confidence ≥", 50, 95, 75, 1)

    st.divider()
    st.caption("Weights (don’t have to sum to 1)")
    w_rsi = st.slider("RSI weight", 0.00, 0.30, 0.15, 0.01)
    w_macd = st.slider("MACD weight", 0.00, 0.30, 0.15, 0.01)
    w_vwap = st.slider("VWAP weight", 0.00, 0.30, 0.15, 0.01)
    w_ema = st.slider("EMA stack (9/20/50) weight", 0.00, 0.30, 0.18, 0.01)
    w_vol = st.slider("Volume ratio weight", 0.00, 0.30, 0.12, 0.01)
    w_uw = st.slider("UW flow weight", 0.00, 0.40, 0.20, 0.01)
    w_teny = st.slider("10Y yield weight", 0.00, 0.20, 0.05, 0.01)

    weights = {"rsi": w_rsi, "macd": w_macd, "vwap": w_vwap, "ema": w_ema, "vol": w_vol, "uw": w_uw, "teny": w_teny}

    st.divider()
    st.subheader("Keys status")
    st.success("POLYGON_API_KEY") if POLYGON_API_KEY else st.error("POLYGON_API_KEY (missing)")
    st.success("UW_TOKEN") if UW_TOKEN else st.error("UW_TOKEN (missing)")
    st.info("EODHD_API_KEY (optional)") if EODHD_API_KEY else st.warning("EODHD_API_KEY (optional, missing)")
    st.info("FRED_API_KEY (optional)") if FRED_API_KEY else st.warning("FRED_API_KEY (optional, missing)")

# Auto-refresh (simple)
st.caption(f"Last update (CST): {fmt_cst(now_cst())}")
st.markdown(
    f"<script>setTimeout(()=>window.location.reload(), {refresh_sec*1000});</script>",
    unsafe_allow_html=True,
)

# ============================================================
# Fetch shared data
# ============================================================
ten_y_val, ten_y_status = fred_10y_yield()
flow_df, flow_status = uw_flow_alerts(limit=150)

# Endpoint status (top)
status_cols = st.columns([1, 1, 1, 1], gap="small")

def status_box(label: str, status: str):
    if status == "ok":
        st.success(f"{label} (ok)")
    elif status in ("empty", "N/A", "missing_key"):
        st.warning(f"{label} ({status})")
    elif status.startswith("http_"):
        st.error(f"{label} ({status})")
    else:
        st.error(f"{label} ({status})")

with status_cols[0]:
    status_box("UW flow-alerts", flow_status)
with status_cols[1]:
    status_box("Polygon intraday", "ok" if POLYGON_API_KEY else "missing_key")
with status_cols[2]:
    status_box("EODHD news/IV", "ok" if EODHD_API_KEY else "missing_key")
with status_cols[3]:
    status_box("FRED 10Y", ten_y_status)

# ============================================================
# Main layout
# ============================================================
left, right = st.columns([0.33, 0.67], gap="large")

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("Embedded. True filtering (DTE/ITM/premium rules) is best done inside the UW screener UI.")
    UW_SCREEN_URL = "https://unusualwhales.com/options-screener"
    st.components.v1.iframe(UW_SCREEN_URL, height=760, scrolling=True)

with right:
    st.subheader("Live Score / Signals (Polygon intraday + EODHD headlines + UW flow)")

    rows = []
    news_frames = []

    for t in tickers:
        # Polygon bars (reliable intraday)
        bars, bars_status = polygon_intraday_bars(
            t,
            interval=interval,
            lookback_minutes=price_lookback,
            include_extended=include_extended,
        )

        # Optional IV + news via EODHD
        iv_now, iv_status = (None, "missing_key")
        news_df, news_status = (pd.DataFrame(), "missing_key")

        if EODHD_API_KEY:
            iv_now, iv_status = eodhd_options_chain_iv(t)
            news_df, news_status = eodhd_news(t, lookback_minutes=news_lookback)
            if news_status == "ok" and news_df is not None and not news_df.empty:
                news_frames.append(news_df)

        out = score_signal(
            df_bars=bars,
            flow_df=flow_df if flow_status == "ok" else pd.DataFrame(),
            ticker=t,
            iv_now=iv_now,
            ten_y=ten_y_val if ten_y_status == "ok" else None,
            weights=weights,
        )

        out["Bars_status"] = bars_status if out["Bars_status"] in ("empty", "ok") or out["Bars_status"].startswith("too_") else bars_status
        out["News_status"] = news_status
        out["UW_flow_status"] = flow_status
        out["IV_status"] = iv_status

        out["institutional"] = "YES" if out["confidence"] >= inst_threshold and out["signal"] != "WAIT" else "NO"
        rows.append(out)

    df_out = pd.DataFrame(rows)

    show_cols = [
        "ticker", "confidence", "direction", "signal", "institutional",
        "RSI", "MACD_hist", "VWAP_above", "EMA_stack", "Vol_ratio",
        "UW_bias", "Put/Call_vol", "IV_now", "IV_spike", "Gamma_bias", "10Y",
        "Bars", "Last_bar(CST)", "Bars_status", "IV_status", "News_status", "UW_flow_status",
    ]
    for c in show_cols:
        if c not in df_out.columns:
            df_out[c] = "N/A"

    st.dataframe(df_out[show_cols], use_container_width=True, height=280)

    st.subheader(f"Institutional Alerts (≥ {inst_threshold} only)")
    inst = df_out[(df_out["institutional"] == "YES")].copy()
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        for _, r in inst.sort_values("confidence", ascending=False).iterrows():
            st.success(f"{r['ticker']}: {r['signal']} • {r['direction']} • Confidence={int(r['confidence'])}")

    st.subheader("Unusual Flow Alerts (UW API) — filtered")
    st.caption("Rules: premium ≥ $1,000,000 • DTE ≤ 3 days • Volume > OI • Exclude ITM (best-effort).")
    if flow_status != "ok" or flow_df.empty:
        st.warning("UW flow alerts not available right now (check token / plan / endpoint).")
    else:
        f = flow_df.copy()

        # DTE
        if "expiry" in f.columns:
            f["expiry_dt"] = pd.to_datetime(f["expiry"], errors="coerce", utc=True)
            f["dte"] = (f["expiry_dt"] - dt.datetime.now(tz=UTC)).dt.total_seconds() / 86400.0
        else:
            f["dte"] = pd.NA

        # premium
        if "premium" in f.columns:
            f["premium_num"] = pd.to_numeric(f["premium"], errors="coerce")
        else:
            f["premium_num"] = pd.NA

        # volume & OI
        vol_col = "volume" if "volume" in f.columns else ("size" if "size" in f.columns else None)
        oi_col = "open_interest" if "open_interest" in f.columns else None

        f["volume_num"] = pd.to_numeric(f[vol_col], errors="coerce") if vol_col else pd.NA
        f["oi_num"] = pd.to_numeric(f[oi_col], errors="coerce") if oi_col else pd.NA

        filt = pd.Series([True] * len(f))
        filt &= (f["premium_num"].fillna(0) >= 1_000_000)
        filt &= (f["dte"].fillna(999) <= 3)
        if "volume_num" in f.columns and "oi_num" in f.columns:
            filt &= (f["volume_num"].fillna(0) > f["oi_num"].fillna(10**18))

        # Stocks/ETF only (best-effort)
        if "issue_type" in f.columns:
            it = f["issue_type"].astype(str).str.lower()
            filt &= it.isin(["common stock", "etf"])

        # Exclude ITM (best-effort)
        if "strike" in f.columns and "underlying_price" in f.columns and "option_type" in f.columns:
            strike = pd.to_numeric(f["strike"], errors="coerce")
            und = pd.to_numeric(f["underlying_price"], errors="coerce")
            opt = f["option_type"].astype(str).str.lower()
            is_itm_call = (opt == "call") & (und > strike)
            is_itm_put = (opt == "put") & (und < strike)
            filt &= ~(is_itm_call | is_itm_put)

        f2 = f[filt].copy()

        cols = []
        for c in ["executed_at", "underlying_symbol", "option_type", "strike", "expiry", "premium_num", "volume_num", "oi_num", "delta", "gamma"]:
            if c in f2.columns:
                cols.append(c)

        if cols:
            show = f2[cols].head(40).copy()
            if "executed_at" in show.columns:
                show["executed_at"] = pd.to_datetime(show["executed_at"], errors="coerce", utc=True).dt.tz_convert(CST).dt.strftime("%Y-%m-%d %H:%M:%S CST")
            st.dataframe(show, use_container_width=True, height=260)
        else:
            st.warning("UW data returned, but expected fields aren’t present to display neatly.")

    st.subheader(f"News — last {news_lookback} minutes (EODHD)")
    if not EODHD_API_KEY:
        st.warning("EODHD_API_KEY missing — news disabled.")
    elif not news_frames:
        st.info("No news in this lookback window (or EODHD returned none).")
    else:
        news_all = pd.concat(news_frames, ignore_index=True)
        for c in ["ticker", "published_cst", "source", "title", "url"]:
            if c not in news_all.columns:
                news_all[c] = ""
        st.dataframe(news_all[["ticker", "published_cst", "source", "title", "url"]].head(40), use_container_width=True, height=220)
        st.caption("Tip: Click URL column links (or copy/paste).")
