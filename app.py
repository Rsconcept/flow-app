# app.py
# ============================================================
# Institutional Options Signals (5m) — CALLS / PUTS ONLY
# Polygon intraday (PRIMARY) + UW flow + (optional) EODHD news/IV + (optional) FRED 10Y
#
# BEST MODIFICATION CHOSEN (per your plan limits):
# ✅ Max 5 tickers
# ✅ 1 Polygon call per ticker
# ✅ Default auto-refresh = 60 seconds
# ✅ Within 5 calls/minute (stable, no 429/403 chaos)
#
# Other changes:
# - Default news lookback = 60 minutes
# - Removed EODHD intraday fallback (your EODHD intraday returns empty list)
# - Rate-limit safe: shows "rate_limited" without crashing scoring
# - Central timezone with DST support (America/Chicago) + 12h clock
# - Ticker box blank by default (no default tickers)
# ============================================================

import os
import json
import math
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# ============================================================
# App config
# ============================================================
st.set_page_config(
    page_title="Institutional Options Signals (5m) — CALLS / PUTS ONLY",
    layout="wide",
)

APP_TITLE = "🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY"
UTC = dt.timezone.utc
ET = ZoneInfo("America/New_York") if ZoneInfo else None
CENTRAL = ZoneInfo("America/Chicago") if ZoneInfo else dt.timezone(dt.timedelta(hours=-6))

POLYGON_BASE = "https://api.polygon.io"
EODHD_BASE = "https://eodhd.com/api"


# ============================================================
# Secrets / env
# ============================================================
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


POLYGON_API_KEY = get_secret("POLYGON_API_KEY")  # REQUIRED for intraday bars
UW_TOKEN = get_secret("UW_TOKEN")                # REQUIRED for flow alerts
EODHD_API_KEY = get_secret("EODHD_API_KEY")      # OPTIONAL for news/IV (intraday not used)
FRED_API_KEY = get_secret("FRED_API_KEY")        # OPTIONAL for 10Y yield

UW_FLOW_ALERTS_URL = get_secret("UW_FLOW_ALERTS_URL") or "https://api.unusualwhales.com/api/option-trades/flow-alerts"


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


# ============================================================
# Time helpers
# ============================================================
def now_utc() -> dt.datetime:
    return dt.datetime.now(tz=UTC)

def now_ct() -> dt.datetime:
    return dt.datetime.now(tz=CENTRAL)

def fmt_ct(ts: Optional[dt.datetime]) -> str:
    if not ts:
        return "N/A"
    try:
        return ts.astimezone(CENTRAL).strftime("%Y-%m-%d %I:%M:%S %p %Z")
    except Exception:
        return "N/A"

def interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        try:
            return int(interval.replace("m", ""))
        except Exception:
            return 5
    return 5


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
    if len(df) < max(lookback, 2):
        return float("nan")
    last = df["volume"].iloc[-1]
    avg = df["volume"].iloc[-lookback:].mean()
    if avg == 0 or pd.isna(avg):
        return float("nan")
    return float(last / avg)


# ============================================================
# Polygon intraday bars (PRIMARY)
# - Rate-limit safe
# - Cache-busted by bar boundary so you don't get "yesterday stuck"
# ============================================================
@st.cache_data(ttl=55, show_spinner=False)
def polygon_intraday_bars(
    ticker: str,
    interval: str,
    lookback_minutes: int,
    include_extended: bool = True,
    cache_buster: int = 0
) -> Tuple[pd.DataFrame, str]:
    if not POLYGON_API_KEY:
        return pd.DataFrame(), "missing_key"

    t = ticker.upper().strip()
    if not interval.endswith("m"):
        return pd.DataFrame(), "bad_interval"
    try:
        multiplier = int(interval.replace("m", ""))
    except Exception:
        return pd.DataFrame(), "bad_interval"

    end_utc = now_utc()
    start_utc = end_utc - dt.timedelta(minutes=lookback_minutes)

    # Polygon uses date strings. Use ET dates for correctness around midnight.
    if ET is None:
        from_date = start_utc.date().isoformat()
        to_date = end_utc.date().isoformat()
    else:
        from_date = start_utc.astimezone(ET).date().isoformat()
        to_date = end_utc.astimezone(ET).date().isoformat()

    url = f"{POLYGON_BASE}/v2/aggs/ticker/{t}/range/{multiplier}/minute/{from_date}/{to_date}"
    params = {"apiKey": POLYGON_API_KEY, "adjusted": "true", "sort": "asc", "limit": 50000}

    code, text, _ = http_get(url, params=params, timeout=20)

    if code == 429:
        return pd.DataFrame(), "rate_limited(429)"

    if code != 200:
        j = safe_json(text)
        if isinstance(j, dict):
            msg = j.get("message") or j.get("error") or j.get("status") or ""
            msg = str(msg)[:120]
            return pd.DataFrame(), f"http_{code} {msg}".strip()
        return pd.DataFrame(), f"http_{code}"

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

    # Filter regular session if requested (ET 9:30–16:00)
    if not include_extended and ET is not None:
        et_times = out["datetime"].dt.tz_convert(ET)
        regular = (
            ((et_times.dt.hour > 9) | ((et_times.dt.hour == 9) & (et_times.dt.minute >= 30)))
            & (et_times.dt.hour < 16)
        )
        out = out[regular].copy()
        if out.empty:
            return pd.DataFrame(), "empty_regular_only"

    # Clip to lookback
    cutoff = end_utc - dt.timedelta(minutes=lookback_minutes)
    clipped = out[out["datetime"] >= cutoff].copy()
    out = clipped if not clipped.empty else out.tail(500).copy()

    return out, "ok"


# ============================================================
# EODHD news + IV (optional)
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

    if "date" in df.columns:
        df["published_utc"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    elif "published_at" in df.columns:
        df["published_utc"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    else:
        df["published_utc"] = pd.NaT

    df["published_ct"] = df["published_utc"].dt.tz_convert(CENTRAL)
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
    out["published_ct"] = df["published_ct"].dt.strftime("%Y-%m-%d %I:%M:%S %p %Z")
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

    try:
        return float(obs[0].get("value")), "ok"
    except Exception:
        return None, "parse_error"


# ============================================================
# Unusual Whales
# ============================================================
def uw_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}" if UW_TOKEN else "",
    }

def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _to_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _uw_underlying_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_existing(df, [
        "underlying_symbol", "underlying", "ticker", "symbol",
        "root_symbol", "stock_symbol", "underlying_ticker", "issue_symbol"
    ])

def _uw_option_type_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_existing(df, ["option_type", "type", "side_type"])

def _uw_size_like_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_existing(df, ["volume", "size", "contracts", "qty", "quantity"])

def _uw_premium_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_existing(df, ["premium", "premium_usd", "total_premium", "notional", "premium_amount"])

def _uw_oi_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_existing(df, ["open_interest", "oi"])

@st.cache_data(ttl=20, show_spinner=False)
def uw_flow_alerts(limit: int = 250) -> Tuple[pd.DataFrame, str]:
    if not UW_TOKEN:
        return pd.DataFrame(), "missing_key"

    params = {"limit": limit}
    code, text, _ = http_get(UW_FLOW_ALERTS_URL, headers=uw_headers(), params=params)
    if code != 200:
        return pd.DataFrame(), f"http_{code}"

    j = safe_json(text)
    if not isinstance(j, dict) or "data" not in j:
        if isinstance(j, list):
            df = pd.DataFrame(j)
        else:
            return pd.DataFrame(), "parse_error"
    else:
        data = j.get("data") or []
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame(), "ok"
        df = pd.DataFrame(data)

    if "greeks" in df.columns:
        try:
            greeks = df["greeks"]
            if greeks.notna().any():
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


def gamma_bias_proxy(flow_df: pd.DataFrame, ticker: str) -> str:
    if flow_df is None or flow_df.empty:
        return "N/A"

    df = flow_df.copy()
    ucol = _uw_underlying_col(df)
    tcol = _uw_option_type_col(df)
    scol = _uw_size_like_col(df)
    gcol = _pick_first_existing(df, ["gamma", "g", "gamma_value", "greeks_gamma"])

    if tcol is None or gcol is None:
        return "N/A"

    t = ticker.upper().strip()
    if ucol is not None:
        df = df[df[ucol].astype(str).str.upper() == t].copy()
    if df.empty:
        return "N/A"

    gamma = pd.to_numeric(df[gcol], errors="coerce").fillna(0)
    size = _to_num_series(df[scol]) if scol is not None else pd.Series([1.0] * len(df))
    opt = df[tcol].astype(str).str.lower()

    sign = opt.map(lambda x: 1.0 if "call" in x else (-1.0 if "put" in x else 0.0))
    score = float((gamma * size * sign).sum())
    if abs(score) < 0.5:
        return "Neutral"
    return "Positive" if score > 0 else "Negative"


# ============================================================
# Signal scoring
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
        "Last_bar(CT)": "N/A",
        "Last_bar_age_min": "N/A",
        "Bars_status": "empty",
        "News_status": "Not Yet",
        "UW_flow_status": "N/A",
    }

    if df_bars is None or df_bars.empty:
        out["Bars_status"] = "empty"
        return out

    out["Bars"] = int(len(df_bars))
    last_bar_utc = df_bars["datetime"].iloc[-1].to_pydatetime()
    out["Last_bar(CT)"] = fmt_ct(last_bar_utc)
    try:
        lb = last_bar_utc if last_bar_utc.tzinfo else last_bar_utc.replace(tzinfo=UTC)
        age_min = (now_utc() - lb).total_seconds() / 60.0
        out["Last_bar_age_min"] = round(age_min, 1)
    except Exception:
        out["Last_bar_age_min"] = "N/A"

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

    pc_ratio, _ = uw_put_call_bias(flow_df, ticker)
    out["Put/Call_vol"] = pc_ratio if pc_ratio is not None else "N/A"
    out["UW_bias"] = (
        "PUT" if (pc_ratio is not None and pc_ratio > 1.1)
        else ("CALL" if (pc_ratio is not None and pc_ratio < 0.9) else "Neutral")
    )
    out["Gamma_bias"] = gamma_bias_proxy(flow_df, ticker)

    if iv_now is None:
        out["IV_spike"] = "N/A"
    else:
        out["IV_spike"] = "YES" if iv_now >= 65 else "NO"

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

    if isinstance(out["Vol_ratio"], (int, float)) and out["Vol_ratio"] != "N/A":
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
        st.rerun()

    ticker_text = st.text_input(
        "Type tickers (comma-separated).",
        value="",
        placeholder="e.g. SPY, TSLA, AMD, META, IWM",
    )
    tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))

    # HARD CAP 5 tickers to respect plan limits
    if len(tickers) > 5:
        st.warning("Plan-safe mode: max 5 tickers (to avoid Polygon 429 rate limits). Extra tickers ignored.")
        tickers = tickers[:5]

    interval = st.selectbox("Candle interval", ["15m", "5m", "1m"], index=1)

    price_lookback = st.slider("Price lookback (minutes)", 60, 1980, 900, 30)
    include_extended = st.toggle("Include pre/after-hours (Polygon)", value=True)

    # ✅ Default news lookback = 60 minutes
    news_lookback = st.slider("News lookback (minutes)", 15, 720, 60, 15)

    # ✅ Default refresh = 60 seconds
    refresh_sec = st.slider("Auto-refresh (seconds)", 10, 300, 60, 5)

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
    st.success("POLYGON_API_KEY") if POLYGON_API_KEY else st.error("POLYGON_API_KEY (missing)")
    st.success("UW_TOKEN") if UW_TOKEN else st.error("UW_TOKEN (missing)")
    st.info("EODHD_API_KEY (optional)") if EODHD_API_KEY else st.warning("EODHD_API_KEY (optional, missing)")
    st.info("FRED_API_KEY (optional)") if FRED_API_KEY else st.warning("FRED_API_KEY (optional, missing)")


# Auto-refresh
st.caption(f"Last update (CT): {fmt_ct(now_ct())}")
st.markdown(f"<script>setTimeout(()=>window.location.reload(), {refresh_sec*1000});</script>", unsafe_allow_html=True)

# Fetch shared data
ten_y_val, ten_y_status = fred_10y_yield()
flow_df, flow_status = uw_flow_alerts(limit=250)

# Endpoint status (top)
status_cols = st.columns([1, 1, 1, 1], gap="small")

def status_box(label: str, status: str):
    if status == "ok":
        st.success(f"{label} (ok)")
    elif status in ("empty", "N/A", "missing_key"):
        st.warning(f"{label} ({status})")
    elif str(status).startswith("http_") or "rate_limited" in str(status):
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

# Layout
left, right = st.columns([0.33, 0.67], gap="large")

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("Embedded. True filtering (DTE/ITM/premium rules) is best done inside the UW screener UI.")
    st.components.v1.iframe("https://unusualwhales.com/options-screener", height=760, scrolling=True)

with right:
    st.subheader("Live Score / Signals (Intraday OHLC + EODHD headlines + UW flow)")

    if not tickers:
        st.info("Enter up to 5 tickers in the sidebar to start (e.g., SPY, TSLA).")
        st.stop()

    rows: List[Dict[str, Any]] = []
    news_frames: List[pd.DataFrame] = []

    # Cache buster per bar boundary (keeps cache aligned with refresh)
    mult_min = interval_to_minutes(interval)
    cache_buster = int(now_utc().timestamp() // max(60, mult_min * 60))

    for t in tickers:
        bars, bars_status = polygon_intraday_bars(
            t, interval=interval, lookback_minutes=price_lookback,
            include_extended=include_extended, cache_buster=cache_buster
        )

        # If rate-limited, keep it explicit (no misleading scoring)
        if "rate_limited" in bars_status:
            out = score_signal(
                df_bars=pd.DataFrame(),
                flow_df=flow_df if flow_status == "ok" else pd.DataFrame(),
                ticker=t,
                iv_now=None,
                ten_y=ten_y_val if ten_y_status == "ok" else None,
                weights=weights,
            )
            out["Bars_status"] = bars_status
            out["IV_status"] = "N/A"
            out["News_status"] = "Not Yet"
            out["UW_flow_status"] = flow_status
            out["institutional"] = "NO"
            rows.append(out)
            continue

        iv_now, iv_status = (None, "missing_key")
        news_df, news_status_raw = (pd.DataFrame(), "missing_key")

        if EODHD_API_KEY:
            iv_now, iv_status = eodhd_options_chain_iv(t)
            news_df, news_status_raw = eodhd_news(t, lookback_minutes=news_lookback)
            if news_status_raw == "ok" and news_df is not None and not news_df.empty:
                news_frames.append(news_df)

        news_flag = "YES" if (news_status_raw == "ok" and news_df is not None and not news_df.empty) else "Not Yet"

        out = score_signal(
            df_bars=bars,
            flow_df=flow_df if flow_status == "ok" else pd.DataFrame(),
            ticker=t,
            iv_now=iv_now,
            ten_y=ten_y_val if ten_y_status == "ok" else None,
            weights=weights,
        )

        out["Bars_status"] = bars_status
        out["IV_status"] = iv_status
        out["News_status"] = news_flag
        out["UW_flow_status"] = flow_status
        out["institutional"] = "YES" if out["confidence"] >= inst_threshold and out["signal"] != "WAIT" else "NO"
        rows.append(out)

    df_out = pd.DataFrame(rows)

    show_cols = [
        "ticker", "confidence", "direction", "signal", "institutional",
        "RSI", "MACD_hist", "VWAP_above", "EMA_stack", "Vol_ratio",
        "UW_bias", "Put/Call_vol", "IV_now", "IV_spike", "Gamma_bias", "10Y",
        "Bars", "Last_bar(CT)", "Last_bar_age_min", "Bars_status", "IV_status", "News_status", "UW_flow_status",
    ]
    for c in show_cols:
        if c not in df_out.columns:
            df_out[c] = "N/A"

    st.dataframe(df_out[show_cols], use_container_width=True, height=280)

    st.subheader(f"Institutional Alerts (≥ {inst_threshold} only)")
    inst = df_out[df_out["institutional"] == "YES"].copy()
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        for _, r in inst.sort_values("confidence", ascending=False).iterrows():
            st.success(f"{r['ticker']}: {r['signal']} • {r['direction']} • Confidence={int(r['confidence'])}")

    # UW Flow alerts filtered — DTE <= 7 days
    st.subheader("Unusual Flow Alerts (UW API) — filtered")
    st.caption("Rules: premium ≥ $1,000,000 • DTE ≤ 7 days • Exclude ITM (best-effort). Volume>OI only if OI exists.")

    if flow_status != "ok" or flow_df.empty:
        st.warning("UW flow alerts not available right now (check token / plan / endpoint).")
    else:
        f = flow_df.copy()

        ucol = _uw_underlying_col(f)
        tcol = _uw_option_type_col(f)
        vcol = _uw_size_like_col(f)
        pcol = _uw_premium_col(f)
        oicol = _uw_oi_col(f)

        if pcol is not None:
            f["premium_num"] = pd.to_numeric(f[pcol], errors="coerce")
        else:
            f["premium_num"] = pd.NA

        if vcol is not None:
            f["volume_num"] = pd.to_numeric(f[vcol], errors="coerce")
        else:
            f["volume_num"] = pd.NA

        if oicol is not None:
            f["oi_num"] = pd.to_numeric(f[oicol], errors="coerce")
        else:
            f["oi_num"] = pd.NA

        exp_col = _pick_first_existing(f, ["expiry", "expiration", "exp", "expiration_date"])
        if exp_col is not None:
            f["expiry_dt"] = pd.to_datetime(f[exp_col], errors="coerce", utc=True)
            f["dte"] = (f["expiry_dt"] - now_utc()).dt.total_seconds() / 86400.0
        else:
            f["dte"] = pd.NA

        filt = pd.Series([True] * len(f))
        filt &= (f["premium_num"].fillna(0) >= 1_000_000)
        filt &= (f["dte"].fillna(999) <= 7)

        if oicol is not None:
            filt &= (f["volume_num"].fillna(0) > f["oi_num"].fillna(10**18))

        strike_col = _pick_first_existing(f, ["strike"])
        und_col = _pick_first_existing(f, ["underlying_price", "underlyingPrice", "stock_price", "spot", "underlying_last"])
        if strike_col is not None and und_col is not None and tcol is not None:
            strike = pd.to_numeric(f[strike_col], errors="coerce")
            und = pd.to_numeric(f[und_col], errors="coerce")
            opt = f[tcol].astype(str).str.lower()
            is_itm_call = opt.str.contains("call") & (und > strike)
            is_itm_put = opt.str.contains("put") & (und < strike)
            filt &= ~(is_itm_call | is_itm_put)

        f2 = f[filt].copy()

        display_cols = []
        for c in ["executed_at", ucol, tcol, strike_col, exp_col, "premium_num", "volume_num", "oi_num", "delta", "gamma"]:
            if c and c in f2.columns and c not in display_cols:
                display_cols.append(c)

        if display_cols:
            show = f2[display_cols].head(80).copy()
            if "executed_at" in show.columns:
                show["executed_at"] = (
                    pd.to_datetime(show["executed_at"], errors="coerce", utc=True)
                    .dt.tz_convert(CENTRAL)
                    .dt.strftime("%Y-%m-%d %I:%M:%S %p %Z")
                )
            st.dataframe(show, use_container_width=True, height=260)
        else:
            st.warning("UW data returned, but expected fields aren’t present to display neatly.")

    # News
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
        st.caption("Tip: Click URL column links (or copy/paste).")
