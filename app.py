# app.py
# ============================================================
# Institutional Options Signals (5m) — CALLS / PUTS ONLY
# EODHD primary for intraday OHLC + news + IV
# Massive (formerly Polygon) fallback for REAL-TIME intraday bars when EODHD is stale
#
# Fixes in this rewrite:
# - "CT" timezone handling (DST-safe) + 12h clock display (06:55 PM CT)
# - Ticker input defaults BLANK; max 5 tickers
# - Rotation mode: only 1 ticker fetch per refresh; others use cached bars
# - Staleness detection: if EODHD stale during market open -> fallback to Massive
# - IV spike uses rolling baseline per ticker so you can verify it works
# - Clear diagnostics panel + explicit endpoint status
# ============================================================

import os
import json
import math
import time
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="Institutional Options Signals (5m) — CALLS / PUTS ONLY (EODHD primary)",
    layout="wide",
)

APP_TITLE = "🏛️ Institutional Options Signals — CALLS / PUTS ONLY (EODHD primary)"
UTC = dt.timezone.utc

# Display timezone: CT (DST-safe)
CT = ZoneInfo("America/Chicago") if ZoneInfo else dt.timezone(dt.timedelta(hours=-6))
ET = ZoneInfo("America/New_York") if ZoneInfo else None

def now_ct() -> dt.datetime:
    return dt.datetime.now(tz=CT)

def fmt_ct(ts: Optional[dt.datetime]) -> str:
    if not ts:
        return "N/A"
    try:
        # 12-hour, e.g. 06:55 PM CT
        return ts.astimezone(CT).strftime("%Y-%m-%d %I:%M:%S %p CT")
    except Exception:
        return "N/A"

def fmt_ct_short(ts: Optional[dt.datetime]) -> str:
    if not ts:
        return "N/A"
    try:
        return ts.astimezone(CT).strftime("%I:%M %p CT")
    except Exception:
        return "N/A"

def minutes_ago(ts: Optional[dt.datetime]) -> Optional[float]:
    if not ts:
        return None
    try:
        delta = dt.datetime.now(tz=UTC) - ts.astimezone(UTC)
        return float(delta.total_seconds() / 60.0)
    except Exception:
        return None

# ----------------------------
# Secrets / env
# ----------------------------
def get_secret(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            v = st.secrets.get(name)
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass
    v = os.environ.get(name)
    return v.strip() if isinstance(v, str) and v.strip() else None

# Massive/Polygon key (same key generally)
MASSIVE_API_KEY = get_secret("POLYGON_API_KEY") or get_secret("MASSIVE_API_KEY")
UW_TOKEN = get_secret("UW_TOKEN")                 # REQUIRED for UW flow
EODHD_API_KEY = get_secret("EODHD_API_KEY")       # REQUIRED here (primary)
FRED_API_KEY = get_secret("FRED_API_KEY")         # OPTIONAL

UW_FLOW_ALERTS_URL = get_secret("UW_FLOW_ALERTS_URL") or "https://api.unusualwhales.com/api/option-trades/flow-alerts"

# ----------------------------
# HTTP helpers
# ----------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "streamlit-options-signals/2.0"})

def http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 20
) -> Tuple[int, str, str]:
    try:
        resp = SESSION.get(url, headers=headers, params=params, timeout=timeout)
        return resp.status_code, resp.text, resp.headers.get("Content-Type", "")
    except requests.RequestException as e:
        return 0, str(e), ""

def safe_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None

# ----------------------------
# Market open helper (best-effort)
# ----------------------------
def market_is_open_estimate() -> bool:
    # Best-effort: if ET is available, treat 8:30–15:00 CT (9:30–16:00 ET) as regular session.
    # Includes weekdays only.
    now = dt.datetime.now(tz=ET) if ET else dt.datetime.now()
    if now.weekday() >= 5:
        return False
    if ET:
        h, m = now.hour, now.minute
        after_open = (h > 9) or (h == 9 and m >= 30)
        before_close = (h < 16)
        return bool(after_open and before_close)
    return True  # fallback if no zoneinfo

# ============================================================
# Indicators
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
    if df is None or df.empty or len(df) < lookback + 1:
        return float("nan")
    last = df["volume"].iloc[-1]
    avg = df["volume"].iloc[-lookback:].mean()
    if avg == 0 or pd.isna(avg) or pd.isna(last):
        return float("nan")
    return float(last / avg)

# ============================================================
# EODHD intraday (PRIMARY)
# ============================================================
EODHD_BASE = "https://eodhd.com/api"

@st.cache_data(ttl=20, show_spinner=False)
def eodhd_intraday_bars(
    ticker: str,
    interval: str,
    lookback_minutes: int,
    include_extended: bool = True,
) -> Tuple[pd.DataFrame, str]:
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"

    t = ticker.upper().strip()
    if interval not in ("1m", "5m", "15m", "1h"):
        return pd.DataFrame(), "bad_interval"

    # EODHD expects SYMBOL.EXCHANGE - for US stocks use .US
    symbol = f"{t}.US"
    now_utc = dt.datetime.now(tz=UTC)
    start_utc = now_utc - dt.timedelta(minutes=lookback_minutes)

    url = f"{EODHD_BASE}/intraday/{symbol}"
    params = {
        "api_token": EODHD_API_KEY,
        "interval": interval,
        "fmt": "json",
        "from": int(start_utc.timestamp()),
        "to": int(now_utc.timestamp()),
    }

    code, text, _ = http_get(url, params=params, timeout=20)
    if code != 200:
        j = safe_json(text)
        if isinstance(j, dict) and j.get("message"):
            return pd.DataFrame(), f"http_{code} {j.get('message')}"
        return pd.DataFrame(), f"http_{code}"

    j = safe_json(text)
    if not isinstance(j, list):
        return pd.DataFrame(), "parse_error"

    if len(j) == 0:
        # EODHD can return [] during market hours / right after close
        return pd.DataFrame(), "empty"

    df = pd.DataFrame(j)

    # Expected fields: datetime, open, high, low, close, volume
    # Sometimes datetime can be "timestamp" or "date"
    dt_col = None
    for c in ["datetime", "timestamp", "date", "time"]:
        if c in df.columns:
            dt_col = c
            break
    if dt_col is None:
        return pd.DataFrame(), "schema_mismatch"

    out = pd.DataFrame()
    # EODHD intraday "datetime" is often ISO-like in exchange timezone; treat as UTC only if it's clearly UTC.
    # Best: parse and assume it's UTC if it has 'Z' or offset, otherwise treat as ET then convert to UTC.
    parsed = pd.to_datetime(df[dt_col], errors="coerce", utc=False)
    if parsed.dt.tz is None:
        # no tz info -> assume ET if available, else UTC
        if ET:
            parsed = parsed.dt.tz_localize(ET, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert(UTC)
        else:
            parsed = parsed.dt.tz_localize(UTC)
    else:
        parsed = parsed.dt.tz_convert(UTC)

    out["datetime"] = parsed
    for k_in, k_out in [("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"), ("volume", "volume")]:
        if k_in not in df.columns:
            return pd.DataFrame(), "schema_mismatch"
        out[k_out] = pd.to_numeric(df[k_in], errors="coerce")

    out = out.dropna(subset=["datetime", "close"]).sort_values("datetime")
    if out.empty:
        return pd.DataFrame(), "empty"

    # Optional: regular session filter (9:30–16:00 ET)
    if not include_extended and ET is not None:
        et_times = out["datetime"].dt.tz_convert(ET)
        regular = (
            ((et_times.dt.hour > 9) | ((et_times.dt.hour == 9) & (et_times.dt.minute >= 30)))
            & (et_times.dt.hour < 16)
        )
        out = out[regular].copy()
        if out.empty:
            return pd.DataFrame(), "empty_regular_only"

    # Clip to exact lookback window
    cutoff = now_utc - dt.timedelta(minutes=lookback_minutes)
    clipped = out[out["datetime"] >= cutoff].copy()
    if not clipped.empty:
        out = clipped

    return out, "ok"

# ============================================================
# Massive (Polygon) intraday bars (FALLBACK when EODHD stale)
# ============================================================
MASSIVE_BASE = "https://api.massive.com"

def massive_headers() -> Dict[str, str]:
    # Massive docs show Authorization: Bearer <API_KEY>
    return {"Authorization": f"Bearer {MASSIVE_API_KEY}", "Accept": "application/json"}

@st.cache_data(ttl=20, show_spinner=False)
def massive_intraday_bars(
    ticker: str,
    interval: str,
    lookback_minutes: int,
    include_extended: bool = True
) -> Tuple[pd.DataFrame, str]:
    if not MASSIVE_API_KEY:
        return pd.DataFrame(), "missing_key"

    t = ticker.upper().strip()
    if not interval.endswith("m"):
        return pd.DataFrame(), "bad_interval"

    try:
        multiplier = int(interval.replace("m", ""))
    except Exception:
        return pd.DataFrame(), "bad_interval"

    end_utc = dt.datetime.now(tz=UTC)
    start_utc = end_utc - dt.timedelta(minutes=lookback_minutes)

    # Massive/Polygon range uses date strings; send in ET dates to avoid day-boundary issues
    if ET:
        from_date = start_utc.astimezone(ET).date().isoformat()
        to_date = end_utc.astimezone(ET).date().isoformat()
    else:
        from_date = start_utc.date().isoformat()
        to_date = end_utc.date().isoformat()

    url = f"{MASSIVE_BASE}/v2/aggs/ticker/{t}/range/{multiplier}/minute/{from_date}/{to_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}

    # Try Bearer first, then query param as fallback
    code, text, _ = http_get(url, headers=massive_headers(), params=params, timeout=20)
    if code == 401 or code == 403:
        params2 = dict(params)
        params2["apiKey"] = MASSIVE_API_KEY
        code, text, _ = http_get(url, params=params2, timeout=20)

    if code != 200:
        j = safe_json(text)
        msg = None
        if isinstance(j, dict):
            msg = j.get("message") or j.get("error")
        return pd.DataFrame(), f"http_{code} {msg}".strip()

    j = safe_json(text)
    if not isinstance(j, dict) or "results" not in j:
        return pd.DataFrame(), "parse_error"

    results = j.get("results") or []
    if not results:
        return pd.DataFrame(), "empty"

    df = pd.DataFrame(results)
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

    # Session filter
    if not include_extended and ET is not None:
        et_times = out["datetime"].dt.tz_convert(ET)
        regular = (
            ((et_times.dt.hour > 9) | ((et_times.dt.hour == 9) & (et_times.dt.minute >= 30)))
            & (et_times.dt.hour < 16)
        )
        out = out[regular].copy()
        if out.empty:
            return pd.DataFrame(), "empty_regular_only"

    cutoff = end_utc - dt.timedelta(minutes=lookback_minutes)
    clipped = out[out["datetime"] >= cutoff].copy()
    if not clipped.empty:
        out = clipped
    else:
        out = out.tail(500).copy()

    return out, "ok"

# ============================================================
# EODHD news + IV
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def eodhd_news(ticker: str, lookback_minutes: int) -> Tuple[pd.DataFrame, str]:
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"
    t = ticker.upper().strip()
    symbol = f"{t}.US"
    url = f"{EODHD_BASE}/news"
    params = {"api_token": EODHD_API_KEY, "fmt": "json", "s": symbol, "limit": 50}

    code, text, _ = http_get(url, params=params)
    if code != 200:
        return pd.DataFrame(), f"http_{code}"

    j = safe_json(text)
    if not isinstance(j, list):
        return pd.DataFrame(), "parse_error"

    df = pd.DataFrame(j)
    if df.empty:
        return pd.DataFrame(), "ok"

    # normalize time
    if "date" in df.columns:
        df["published_utc"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    elif "published_at" in df.columns:
        df["published_utc"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    else:
        df["published_utc"] = pd.NaT

    df["published_ct"] = df["published_utc"].dt.tz_convert(CT)
    cutoff = now_ct() - dt.timedelta(minutes=lookback_minutes)
    df = df[df["published_ct"] >= cutoff].copy()

    def pick_col(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    title_c = pick_col("title", "text")
    src_c = pick_col("source")
    url_c = pick_col("link", "url")

    out = pd.DataFrame()
    out["ticker"] = t
    out["published_ct"] = df["published_ct"].dt.strftime("%Y-%m-%d %I:%M:%S %p CT")
    out["source"] = df[src_c] if src_c else ""
    out["title"] = df[title_c] if title_c else ""
    out["url"] = df[url_c] if url_c else ""
    out = out.dropna(subset=["title"]).head(80)
    return out, "ok"

@st.cache_data(ttl=120, show_spinner=False)
def eodhd_options_chain_iv(ticker: str) -> Tuple[Optional[float], str]:
    if not EODHD_API_KEY:
        return None, "missing_key"
    t = ticker.upper().strip()
    url = f"{EODHD_BASE}/options/{t}.US"
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
    if iv > 200:
        return None, "iv_bad_scale"

    iv_pct = iv * 100.0 if iv <= 2.0 else iv
    return round(iv_pct, 2), "ok"

# ============================================================
# FRED 10Y
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

    try:
        return float(obs[0].get("value")), "ok"
    except Exception:
        return None, "parse_error"

# ============================================================
# Unusual Whales (flow)
# ============================================================
def uw_headers() -> Dict[str, str]:
    return {"Accept": "application/json, text/plain", "Authorization": f"Bearer {UW_TOKEN}" if UW_TOKEN else ""}

def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _to_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _uw_underlying_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_existing(df, ["underlying_symbol", "underlying", "ticker", "symbol", "root_symbol", "stock_symbol"])

def _uw_option_type_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_existing(df, ["option_type", "type", "side_type"])

def _uw_size_like_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_existing(df, ["volume", "size", "contracts", "qty", "quantity"])

def _uw_premium_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_existing(df, ["premium", "premium_usd", "total_premium", "notional", "premium_amount"])

@st.cache_data(ttl=20, show_spinner=False)
def uw_flow_alerts(limit: int = 250) -> Tuple[pd.DataFrame, str]:
    if not UW_TOKEN:
        return pd.DataFrame(), "missing_key"
    params = {"limit": limit}
    code, text, _ = http_get(UW_FLOW_ALERTS_URL, headers=uw_headers(), params=params)
    if code != 200:
        return pd.DataFrame(), f"http_{code}"
    j = safe_json(text)
    if isinstance(j, dict) and "data" in j and isinstance(j["data"], list):
        df = pd.DataFrame(j["data"])
    elif isinstance(j, list):
        df = pd.DataFrame(j)
    else:
        return pd.DataFrame(), "parse_error"

    # extract nested greeks if present
    if "greeks" in df.columns:
        try:
            greeks = df["greeks"]
            gamma_ex = greeks.apply(lambda x: (x.get("gamma") if isinstance(x, dict) else None))
            delta_ex = greeks.apply(lambda x: (x.get("delta") if isinstance(x, dict) else None))
            if "gamma" not in df.columns:
                df["gamma"] = gamma_ex
            else:
                df["gamma"] = df["gamma"].where(df["gamma"].notna(), gamma_ex)
            if "delta" not in df.columns:
                df["delta"] = delta_ex
            else:
                df["delta"] = df["delta"].where(df["delta"].notna(), delta_ex)
        except Exception:
            pass

    return df, "ok"

def uw_put_call_bias(flow_df: pd.DataFrame, ticker: str) -> Tuple[Optional[float], str]:
    if flow_df is None or flow_df.empty:
        return None, "N/A"
    ucol = _uw_underlying_col(flow_df)
    tcol = _uw_option_type_col(flow_df)
    vcol = _uw_size_like_col(flow_df)
    if tcol is None:
        return None, "N/A"

    df = flow_df.copy()
    t = ticker.upper().strip()
    if ucol is not None:
        df = df[df[ucol].astype(str).str.upper() == t].copy()
    if df.empty:
        return None, "N/A"

    types = df[tcol].astype(str).str.lower()
    vols = _to_num_series(df[vcol]) if vcol is not None else pd.Series([1.0] * len(df))
    put_vol = float(vols[types.str.contains("put")].sum())
    call_vol = float(vols[types.str.contains("call")].sum())
    if call_vol <= 0:
        return None, "N/A"
    return round(put_vol / call_vol, 2), "ok"

def gamma_bias_proxy(flow_df: pd.DataFrame, ticker: str) -> Tuple[str, str]:
    """
    Returns (Gamma_bias, method)
    method:
      - "gamma_weighted" if gamma exists
      - "premium_fallback" if gamma missing (use premium signed by call/put)
      - "N/A"
    """
    if flow_df is None or flow_df.empty:
        return "N/A", "N/A"

    df = flow_df.copy()
    ucol = _uw_underlying_col(df)
    tcol = _uw_option_type_col(df)
    scol = _uw_size_like_col(df)
    pcol = _uw_premium_col(df)
    gcol = _pick_first_existing(df, ["gamma", "g", "gamma_value", "greeks_gamma"])

    if tcol is None:
        return "N/A", "N/A"

    t = ticker.upper().strip()
    if ucol is not None:
        df = df[df[ucol].astype(str).str.upper() == t].copy()
    if df.empty:
        return "N/A", "N/A"

    opt = df[tcol].astype(str).str.lower()
    sign = opt.map(lambda x: 1.0 if "call" in x else (-1.0 if "put" in x else 0.0))

    # Gamma-weighted if available
    if gcol is not None:
        gamma = pd.to_numeric(df[gcol], errors="coerce").fillna(0)
        size = _to_num_series(df[scol]) if scol is not None else pd.Series([1.0] * len(df))
        score = float((gamma * size * sign).sum())
        if abs(score) < 0.5:
            return "Neutral", "gamma_weighted"
        return ("Positive" if score > 0 else "Negative"), "gamma_weighted"

    # Premium fallback if available
    if pcol is not None:
        prem = pd.to_numeric(df[pcol], errors="coerce").fillna(0)
        score = float((prem * sign).sum())
        if abs(score) < 1_000_000:
            return "Neutral", "premium_fallback"
        return ("Positive" if score > 0 else "Negative"), "premium_fallback"

    return "N/A", "N/A"

# ============================================================
# Scoring + IV baseline memory
# ============================================================
def get_iv_baseline(ticker: str, iv_now: Optional[float], max_hist: int = 40) -> Tuple[Optional[float], str]:
    """
    Keep a rolling IV history per ticker in session_state so IV_spike is testable.
    """
    if "iv_hist" not in st.session_state:
        st.session_state["iv_hist"] = {}

    hist: List[float] = st.session_state["iv_hist"].get(ticker, [])
    if iv_now is not None and isinstance(iv_now, (int, float)) and not math.isnan(iv_now):
        hist.append(float(iv_now))
        hist = hist[-max_hist:]
        st.session_state["iv_hist"][ticker] = hist

    if len(hist) < 5:
        return (None, "need_more_iv_samples")

    base = float(pd.Series(hist).median())
    return (round(base, 2), "ok")

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
        "IV_base": "N/A",
        "IV_spike": "N/A",
        "Gamma_bias": "N/A",
        "Gamma_method": "N/A",
        "10Y": ten_y if ten_y is not None else "N/A",
        "Bars": 0,
        "Last_bar(CT)": "N/A",
        "Last_bar_age_min": "N/A",
        "Stale?": "N/A",
        "Bars_status": "empty",
        "UW_flow_status": "N/A",
        "IV_status": "N/A",
        "News_status": "Not Yet",
    }

    if df_bars is None or df_bars.empty:
        out["Bars_status"] = "empty"
        return out

    out["Bars"] = int(len(df_bars))
    out["Bars_status"] = "ok"

    last_ts = df_bars["datetime"].iloc[-1].to_pydatetime()
    out["Last_bar(CT)"] = fmt_ct(last_ts)
    age_min = minutes_ago(last_ts)
    out["Last_bar_age_min"] = round(age_min, 2) if age_min is not None else "N/A"

    # Stale detection
    # If market open and last bar older than 20 minutes, it's stale for "live" usage.
    if age_min is None:
        out["Stale?"] = "N/A"
    else:
        out["Stale?"] = "YES" if (market_is_open_estimate() and age_min > 20) else "NO"

    if len(df_bars) < 10:
        out["Bars_status"] = f"too_few_bars({len(df_bars)})"
        return out

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

    # UW
    pc_ratio, _ = uw_put_call_bias(flow_df, ticker)
    out["Put/Call_vol"] = pc_ratio if pc_ratio is not None else "N/A"
    out["UW_bias"] = (
        "PUT" if (pc_ratio is not None and pc_ratio > 1.1)
        else ("CALL" if (pc_ratio is not None and pc_ratio < 0.9) else "Neutral")
    )
    gb, gb_m = gamma_bias_proxy(flow_df, ticker)
    out["Gamma_bias"] = gb
    out["Gamma_method"] = gb_m

    # IV baseline + spike
    iv_base, iv_base_status = get_iv_baseline(ticker, iv_now)
    out["IV_base"] = iv_base if iv_base is not None else "N/A"

    if iv_now is None or iv_base is None:
        out["IV_spike"] = "N/A"
    else:
        # spike if +30% and at least +8 IV points vs baseline
        spike = (iv_now >= iv_base * 1.30) and ((iv_now - iv_base) >= 8)
        out["IV_spike"] = "YES" if spike else "NO"

    bull = 0.0
    bear = 0.0

    if not pd.isna(rsi_v):
        if rsi_v <= 30:
            bull += weights["rsi"]
        elif rsi_v >= 70:
            bear += weights["rsi"]

    if not pd.isna(macd_v):
        if macd_v > 0:
            bull += weights["macd"]
        elif macd_v < 0:
            bear += weights["macd"]

    bull += weights["vwap"] if vwap_above else 0.0
    bear += weights["vwap"] if not vwap_above else 0.0

    if ema_stack_bull:
        bull += weights["ema"]
    elif ema_stack_bear:
        bear += weights["ema"]

    if isinstance(out["Vol_ratio"], (int, float)):
        if out["Vol_ratio"] >= 1.5:
            if not pd.isna(macd_v) and macd_v > 0:
                bull += weights["vol"]
            elif not pd.isna(macd_v) and macd_v < 0:
                bear += weights["vol"]

    if out["UW_bias"] == "CALL":
        bull += weights["uw"]
    elif out["UW_bias"] == "PUT":
        bear += weights["uw"]

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

    if st.button("Force refresh (clear cache)"):
        st.cache_data.clear()
        st.success("Cache cleared. App will reload on next refresh.")

    ticker_text = st.text_input(
        "Type tickers (comma-separated) — max 5",
        value="",  # <-- BLANK by default
        placeholder="SPY,TSLA,AMD,IWM,QQQ",
    )
    tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:5]

    interval = st.selectbox("Candle interval (EODHD)", ["15m", "5m", "1m"], index=1)
    price_lookback = st.slider("Price lookback (minutes)", 60, 1980, 900, 30)
    include_extended = st.toggle("Include extended hours", value=True)

    # Default requested: 60 min news lookback
    news_lookback = st.slider("News lookback (minutes)", 15, 720, 60, 15)
    refresh_sec = st.slider("Auto-refresh (seconds)", 10, 120, 20, 5)

    st.divider()
    inst_threshold = st.slider("Institutional mode: signals only if confidence ≥", 50, 95, 80, 1)

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
    st.success("EODHD_API_KEY") if EODHD_API_KEY else st.error("EODHD_API_KEY (missing)")
    st.success("UW_TOKEN") if UW_TOKEN else st.error("UW_TOKEN (missing)")
    st.info("MASSIVE_API_KEY (fallback)") if MASSIVE_API_KEY else st.warning("MASSIVE_API_KEY (fallback, missing)")
    st.info("FRED_API_KEY (optional)") if FRED_API_KEY else st.warning("FRED_API_KEY (optional, missing)")

# Auto-refresh
st.caption(f"Last update (CT): {fmt_ct(now_ct())}")
st.markdown(f"<script>setTimeout(()=>window.location.reload(), {refresh_sec*1000});</script>", unsafe_allow_html=True)

# Shared data
ten_y_val, ten_y_status = fred_10y_yield()
flow_df, flow_status = uw_flow_alerts(limit=250)

# Endpoint status boxes
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
    status_box("EODHD (primary)", "ok" if EODHD_API_KEY else "missing_key")
with status_cols[1]:
    status_box("UW flow-alerts", flow_status)
with status_cols[2]:
    status_box("Massive fallback", "ok" if MASSIVE_API_KEY else "missing_key")
with status_cols[3]:
    status_box("FRED 10Y", ten_y_status)

left, right = st.columns([0.33, 0.67], gap="large")

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("Embedded. True filtering is best inside UW UI.")
    st.components.v1.iframe("https://unusualwhales.com/options-screener", height=760, scrolling=True)

with right:
    st.subheader("Live Score / Signals (EODHD intraday OHLC + EODHD news/IV + UW flow)")
    if not tickers:
        st.info("Enter 1–5 tickers in the sidebar to start.")
        st.stop()

    # Rotation state
    if "rot_i" not in st.session_state:
        st.session_state["rot_i"] = 0
    rot_i = int(st.session_state["rot_i"]) % len(tickers)
    st.session_state["rot_i"] = rot_i + 1

    st.caption(f"Refresh mode: ROTATE — Updating: {tickers[rot_i]}. Others use last cached bars.")

    rows: List[Dict[str, Any]] = []
    news_frames: List[pd.DataFrame] = []

    # Store bars per ticker in session_state to avoid extra calls
    if "bars_cache" not in st.session_state:
        st.session_state["bars_cache"] = {}  # ticker -> (df, status, source)

    # Fetch ONLY the rotating ticker this refresh
    t_to_update = tickers[rot_i]
    bars, bars_status = eodhd_intraday_bars(t_to_update, interval=interval, lookback_minutes=price_lookback, include_extended=include_extended)
    source_used = "eodhd"

    # If EODHD bars are stale during market open, try Massive fallback
    if bars_status == "ok" and not bars.empty:
        last_ts = bars["datetime"].iloc[-1].to_pydatetime()
        age_min = minutes_ago(last_ts)
        if market_is_open_estimate() and age_min is not None and age_min > 20:
            mb, ms = massive_intraday_bars(t_to_update, interval=interval, lookback_minutes=price_lookback, include_extended=include_extended)
            if ms == "ok" and not mb.empty:
                bars, bars_status = mb, "ok"
                source_used = "massive_fallback"
            else:
                bars_status = f"eodhd_stale; massive_{ms}"

    # Save updated ticker cache
    st.session_state["bars_cache"][t_to_update] = (bars, bars_status, source_used)

    # For each ticker, use cached bars
    for t in tickers:
        cb = st.session_state["bars_cache"].get(t)
        if cb is None:
            df_bars, s, src = pd.DataFrame(), "not_fetched_yet", "cache"
        else:
            df_bars, s, src = cb

        iv_now, iv_status = eodhd_options_chain_iv(t) if EODHD_API_KEY else (None, "missing_key")
        news_df, news_status_raw = eodhd_news(t, lookback_minutes=news_lookback) if EODHD_API_KEY else (pd.DataFrame(), "missing_key")
        if news_status_raw == "ok" and news_df is not None and not news_df.empty:
            news_frames.append(news_df)

        news_flag = "YES" if (news_status_raw == "ok" and news_df is not None and not news_df.empty) else "Not Yet"

        out = score_signal(
            df_bars=df_bars,
            flow_df=flow_df if flow_status == "ok" else pd.DataFrame(),
            ticker=t,
            iv_now=iv_now,
            ten_y=ten_y_val if ten_y_status == "ok" else None,
            weights=weights,
        )

        out["Bars_status"] = f"{s} ({src})"
        out["IV_status"] = iv_status
        out["News_status"] = news_flag
        out["UW_flow_status"] = flow_status
        out["institutional"] = "YES" if out["confidence"] >= inst_threshold and out["signal"] != "WAIT" else "NO"
        rows.append(out)

    df_out = pd.DataFrame(rows)

    show_cols = [
        "ticker", "confidence", "direction", "signal", "institutional",
        "RSI", "MACD_hist", "VWAP_above", "EMA_stack", "Vol_ratio",
        "UW_bias", "Put/Call_vol",
        "IV_now", "IV_base", "IV_spike",
        "Gamma_bias", "Gamma_method",
        "10Y", "Bars", "Last_bar(CT)", "Last_bar_age_min", "Stale?",
        "Bars_status", "IV_status", "News_status", "UW_flow_status",
    ]
    for c in show_cols:
        if c not in df_out.columns:
            df_out[c] = "N/A"

    st.dataframe(df_out[show_cols], use_container_width=True, height=300)

    st.subheader(f"Institutional Alerts (≥ {inst_threshold} only)")
    inst = df_out[df_out["institutional"] == "YES"].copy()
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        for _, r in inst.sort_values("confidence", ascending=False).iterrows():
            st.success(f"{r['ticker']}: {r['signal']} • {r['direction']} • Confidence={int(r['confidence'])}")

    st.subheader(f"News — last {news_lookback} minutes (EODHD)")
    if not EODHD_API_KEY:
        st.warning("EODHD_API_KEY missing — news disabled.")
    elif not news_frames:
        st.info("No news in this lookback window.")
    else:
        news_all = pd.concat(news_frames, ignore_index=True)
        for c in ["ticker", "published_ct", "source", "title", "url"]:
            if c not in news_all.columns:
                news_all[c] = ""
        st.dataframe(news_all[["ticker", "published_ct", "source", "title", "url"]].head(80), use_container_width=True, height=220)

    with st.expander("Diagnostics"):
        st.write("Market open estimate:", "OPEN" if market_is_open_estimate() else "CLOSED")
        st.write("CT time now:", fmt_ct(now_ct()))
        st.write("Tickers:", tickers)
        st.write("Rotation updated this refresh:", t_to_update)
        st.write("Bars cache keys:", list(st.session_state.get("bars_cache", {}).keys()))
        st.write("IV history sample size (per ticker):", {k: len(v) for k, v in st.session_state.get("iv_hist", {}).items()})
