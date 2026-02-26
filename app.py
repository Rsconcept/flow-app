# app.py
# ============================================================
# Institutional Options Signals (Intraday) — CALLS / PUTS ONLY
#
# CLEAN VERSION (EODHD PRIMARY) + UPGRADES
# - Intraday OHLC: EODHD (primary)
# - Options IV: EODHD
# - News: EODHD (default lookback 60 min)
# - Flow: UnusualWhales (UW)
# - 10Y: FRED (optional)
#
# NEW UPGRADES (requested)
# 1) Gamma_bias fallback:
#    - If UW gamma is missing, we compute a fallback using PREMIUM-weighted CALL vs PUT flow.
# 2) Vol_ratio improvements:
#    - Adds Last_vol + Avg20_vol columns
#    - Shows readable status instead of mysterious N/A when volume is missing/zero.
# 3) IV_spike = REAL spike:
#    - Keeps a per-ticker rolling IV baseline (median of last N IV values)
#    - Spike if IV_now is meaningfully above baseline (ratio + absolute)
#    - Displays IV_base so you can trust it.
#
# Other:
# - Tickers default BLANK (max 5)
# - CT time display in 12h format (e.g., 6:55 PM CT)
# - Last_bar(CT) + Last_bar_age_min + stale detection
# - Rate-safe mode: ROTATE tickers per refresh (default)
# - Clear-cache button + Diagnostics
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
    page_title="Institutional Options Signals — CALLS / PUTS ONLY",
    layout="wide",
)

APP_TITLE = "🏛️ Institutional Options Signals — CALLS / PUTS ONLY (EODHD primary)"
UTC = dt.timezone.utc
CT = ZoneInfo("America/Chicago") if ZoneInfo else dt.timezone(dt.timedelta(hours=-6))
ET = ZoneInfo("America/New_York") if ZoneInfo else None


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


EODHD_API_KEY = get_secret("EODHD_API_KEY")      # REQUIRED for intraday in this build
UW_TOKEN = get_secret("UW_TOKEN")                # OPTIONAL but strongly recommended
FRED_API_KEY = get_secret("FRED_API_KEY")        # OPTIONAL

UW_FLOW_ALERTS_URL = get_secret("UW_FLOW_ALERTS_URL") or "https://api.unusualwhales.com/api/option-trades/flow-alerts"


# ============================================================
# HTTP helpers
# ============================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "streamlit-options-signals/clean-eodhd-2.0"})


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
def now_ct() -> dt.datetime:
    return dt.datetime.now(tz=CT)


def fmt_ct(ts: Optional[dt.datetime]) -> str:
    if not ts:
        return "N/A"
    try:
        # 12-hour clock with CT label (handles CST/CDT correctly)
        return ts.astimezone(CT).strftime("%Y-%m-%d %I:%M:%S %p CT")
    except Exception:
        return "N/A"


def minutes_since(ts_utc: Optional[pd.Timestamp]) -> Optional[float]:
    if ts_utc is None or pd.isna(ts_utc):
        return None
    try:
        nowu = dt.datetime.now(tz=UTC)
        dtu = ts_utc.to_pydatetime().astimezone(UTC)
        return round((nowu - dtu).total_seconds() / 60.0, 2)
    except Exception:
        return None


# ============================================================
# Indicators
# ============================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
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


def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def volume_ratio_with_debug(df: pd.DataFrame, lookback: int = 20) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Returns:
      (vol_ratio, last_vol, avg_vol, status)
    """
    if df is None or df.empty:
        return None, None, None, "empty"

    if "volume" not in df.columns:
        return None, None, None, "missing_volume_col"

    vol = pd.to_numeric(df["volume"], errors="coerce")
    if vol.dropna().empty:
        return None, None, None, "volume_all_nan"

    if len(vol) < max(lookback, 2):
        last_vol = safe_float(vol.iloc[-1])
        avg_vol = safe_float(vol.mean())
        return None, last_vol, avg_vol, f"too_few_bars_for_vr({len(vol)})"

    last_vol = safe_float(vol.iloc[-1])
    avg_vol = safe_float(vol.iloc[-lookback:].mean())

    if avg_vol is None or avg_vol == 0:
        return None, last_vol, avg_vol, "avg20_zero_or_nan"

    if last_vol is None:
        return None, last_vol, avg_vol, "last_vol_nan"

    return round(last_vol / avg_vol, 2), last_vol, round(avg_vol, 2), "ok"


# ============================================================
# EODHD (PRIMARY): intraday bars + news + IV
# ============================================================
EODHD_BASE = "https://eodhd.com/api"


@st.cache_data(ttl=60, show_spinner=False)
def eodhd_intraday_bars(
    ticker: str,
    interval: str,
    lookback_minutes: int,
    include_extended: bool
) -> Tuple[pd.DataFrame, str]:
    """
    Primary OHLC source.
    We fetch latest available candles then locally clip to lookback window.
    """
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"

    t = ticker.upper().strip()
    if interval not in ("1m", "5m", "15m", "1h"):
        return pd.DataFrame(), "bad_interval"

    url = f"{EODHD_BASE}/intraday/{t}.US"
    params = {
        "api_token": EODHD_API_KEY,
        "interval": interval,
        "fmt": "json",
    }

    code, text, _ = http_get(url, params=params, timeout=20)
    if code != 200:
        return pd.DataFrame(), f"http_{code}"

    j = safe_json(text)
    if not isinstance(j, list):
        return pd.DataFrame(), "parse_error"
    if len(j) == 0:
        return pd.DataFrame(), "empty"

    df = pd.DataFrame(j)
    if "datetime" not in df.columns:
        return pd.DataFrame(), "schema_mismatch"

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    out["open"] = pd.to_numeric(df.get("open"), errors="coerce")
    out["high"] = pd.to_numeric(df.get("high"), errors="coerce")
    out["low"] = pd.to_numeric(df.get("low"), errors="coerce")
    out["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    out["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
    out = out.dropna(subset=["datetime", "close"]).sort_values("datetime")

    if out.empty:
        return pd.DataFrame(), "empty"

    # Optionally remove extended hours (filter regular session 9:30-16:00 ET)
    if not include_extended and ET is not None:
        et_times = out["datetime"].dt.tz_convert(ET)
        regular = (
            ((et_times.dt.hour > 9) | ((et_times.dt.hour == 9) & (et_times.dt.minute >= 30)))
            & (et_times.dt.hour < 16)
        )
        out = out[regular].copy()
        if out.empty:
            return pd.DataFrame(), "empty_regular_only"

    # Clip to lookback window
    cutoff = dt.datetime.now(tz=UTC) - dt.timedelta(minutes=lookback_minutes)
    clipped = out[out["datetime"] >= cutoff].copy()
    if not clipped.empty:
        out = clipped
    else:
        # After-hours / weekends: keep tail so indicators can compute
        out = out.tail(500).copy()

    return out, "ok"


@st.cache_data(ttl=120, show_spinner=False)
def eodhd_news(ticker: str, lookback_minutes: int) -> Tuple[pd.DataFrame, str]:
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"

    t = ticker.upper().strip()
    symbol = f"{t}.US"
    url = f"{EODHD_BASE}/news"
    params = {"api_token": EODHD_API_KEY, "fmt": "json", "s": symbol, "limit": 50}

    code, text, _ = http_get(url, params=params, timeout=20)
    if code != 200:
        return pd.DataFrame(), f"http_{code}"

    j = safe_json(text)
    if not isinstance(j, list):
        return pd.DataFrame(), "parse_error"

    df = pd.DataFrame(j)
    if df.empty:
        return pd.DataFrame(), "ok"

    # Normalize published time
    if "date" in df.columns:
        df["published_utc"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    elif "published_at" in df.columns:
        df["published_utc"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    else:
        df["published_utc"] = pd.NaT

    cutoff = dt.datetime.now(tz=UTC) - dt.timedelta(minutes=lookback_minutes)
    df = df[df["published_utc"] >= cutoff].copy()

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
    out["published_ct"] = df["published_utc"].dt.tz_convert(CT).dt.strftime("%Y-%m-%d %I:%M %p CT")
    out["source"] = df[src_c] if src_c else ""
    out["title"] = df[title_c] if title_c else ""
    out["url"] = df[url_c] if url_c else ""
    out = out.dropna(subset=["title"]).head(80)
    return out, "ok"


@st.cache_data(ttl=300, show_spinner=False)
def eodhd_options_chain_iv(ticker: str) -> Tuple[Optional[float], str]:
    if not EODHD_API_KEY:
        return None, "missing_key"

    t = ticker.upper().strip()
    url = f"{EODHD_BASE}/options/{t}.US"
    params = {"api_token": EODHD_API_KEY, "fmt": "json"}

    code, text, _ = http_get(url, params=params, timeout=20)
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

    # Normalize to %
    iv_pct = iv * 100.0 if iv <= 2.0 else iv
    return round(iv_pct, 2), "ok"


# ============================================================
# FRED 10Y yield (optional)
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
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

    code, text, _ = http_get(url, params=params, timeout=20)
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
# Unusual Whales flow alerts (optional but recommended)
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


@st.cache_data(ttl=25, show_spinner=False)
def uw_flow_alerts(limit: int = 250) -> Tuple[pd.DataFrame, str]:
    if not UW_TOKEN:
        return pd.DataFrame(), "missing_key"

    params = {"limit": limit}
    code, text, _ = http_get(UW_FLOW_ALERTS_URL, headers=uw_headers(), params=params, timeout=20)
    if code != 200:
        return pd.DataFrame(), f"http_{code}"

    j = safe_json(text)
    if isinstance(j, dict) and "data" in j:
        data = j.get("data") or []
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame(), "ok"
        df = pd.DataFrame(data)
    elif isinstance(j, list):
        df = pd.DataFrame(j)
    else:
        return pd.DataFrame(), "parse_error"

    # Extract nested greeks if present
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


def gamma_bias_or_fallback(flow_df: pd.DataFrame, ticker: str) -> Tuple[str, str]:
    """
    Returns (bias_label, method)
      method:
        - "gamma" (best)
        - "premium_fallback"
        - "N/A"
    """
    if flow_df is None or flow_df.empty:
        return "N/A", "N/A"

    df = flow_df.copy()
    ucol = _uw_underlying_col(df)
    tcol = _uw_option_type_col(df)
    scol = _uw_size_like_col(df)
    gcol = _pick_first_existing(df, ["gamma", "g", "gamma_value", "greeks_gamma"])
    pcol = _uw_premium_col(df)

    if tcol is None:
        return "N/A", "N/A"

    t = ticker.upper().strip()
    if ucol is not None:
        df = df[df[ucol].astype(str).str.upper() == t].copy()
    if df.empty:
        return "N/A", "N/A"

    # 1) Try gamma method
    if gcol is not None:
        gamma = pd.to_numeric(df[gcol], errors="coerce").fillna(0)
        size = _to_num_series(df[scol]) if scol is not None else pd.Series([1.0] * len(df))
        opt = df[tcol].astype(str).str.lower()
        sign = opt.map(lambda x: 1.0 if "call" in x else (-1.0 if "put" in x else 0.0))
        score = float((gamma * size * sign).sum())

        if abs(score) < 0.5:
            return "Neutral", "gamma"
        return ("Positive" if score > 0 else "Negative"), "gamma"

    # 2) Premium-weighted fallback
    if pcol is not None:
        prem = pd.to_numeric(df[pcol], errors="coerce").fillna(0)
        opt = df[tcol].astype(str).str.lower()
        call_p = float(prem[opt.str.contains("call")].sum())
        put_p = float(prem[opt.str.contains("put")].sum())
        total = call_p + put_p
        if total <= 0:
            return "Neutral", "premium_fallback"

        # convert to bias label similar to gamma output
        # Positive => call premium dominates, Negative => put premium dominates
        ratio = (call_p - put_p) / total  # -1..+1
        if abs(ratio) < 0.10:
            return "Neutral", "premium_fallback"
        return ("Positive" if ratio > 0 else "Negative"), "premium_fallback"

    return "N/A", "N/A"


# ============================================================
# IV baseline + spike detection (session-based)
# ============================================================
def update_iv_history(ticker: str, iv_now: Optional[float], max_len: int = 30) -> Optional[float]:
    """
    Stores iv_now into session history and returns current baseline (median).
    Baseline is median of history (including latest if present).
    """
    if "iv_hist" not in st.session_state:
        st.session_state.iv_hist = {}

    if ticker not in st.session_state.iv_hist:
        st.session_state.iv_hist[ticker] = []

    hist: List[float] = st.session_state.iv_hist[ticker]

    if iv_now is not None and math.isfinite(float(iv_now)):
        hist.append(float(iv_now))
        if len(hist) > max_len:
            hist[:] = hist[-max_len:]

    st.session_state.iv_hist[ticker] = hist
    if len(hist) >= 5:
        return float(pd.Series(hist).median())
    return None


def iv_spike_flag(iv_now: Optional[float], iv_base: Optional[float], ratio_thr: float = 1.35, abs_thr: float = 10.0) -> str:
    """
    Spike if:
      - iv_now >= 65 (absolute "hot" IV)
      OR
      - iv_base exists and iv_now >= iv_base*ratio_thr AND iv_now - iv_base >= abs_thr
    """
    if iv_now is None:
        return "N/A"

    try:
        ivn = float(iv_now)
    except Exception:
        return "N/A"

    if ivn >= 65:
        return "YES"

    if iv_base is None:
        return "NO"

    try:
        ivb = float(iv_base)
    except Exception:
        return "NO"

    if ivb <= 0:
        return "NO"

    if (ivn >= ivb * ratio_thr) and ((ivn - ivb) >= abs_thr):
        return "YES"

    return "NO"


# ============================================================
# Signal scoring
# ============================================================
def score_signal(
    df_bars: pd.DataFrame,
    flow_df: pd.DataFrame,
    ticker: str,
    iv_now: Optional[float],
    iv_base: Optional[float],
    ten_y: Optional[float],
    weights: Dict[str, float],
    stale_after_min: float,
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
        "Last_vol": "N/A",
        "Avg20_vol": "N/A",
        "Vol_status": "N/A",
        "UW_bias": "N/A",
        "Put/Call_vol": "N/A",
        "IV_now": iv_now if iv_now is not None else "N/A",
        "IV_base": iv_base if iv_base is not None else "N/A",
        "IV_spike": "N/A",
        "Gamma_bias": "N/A",
        "Gamma_method": "N/A",
        "10Y": ten_y if ten_y is not None else "N/A",
        "Bars": 0,
        "Last_bar(CT)": "N/A",
        "Last_bar_age_min": "N/A",
        "Stale?": "N/A",
        "Bars_status": "empty",
        "News_status": "Not Yet",
        "UW_flow_status": "N/A",
        "Data_source": "EODHD",
    }

    # IV spike with baseline
    out["IV_spike"] = iv_spike_flag(iv_now, iv_base)

    if df_bars is None or df_bars.empty:
        out["Bars_status"] = "empty"
        return out

    out["Bars"] = int(len(df_bars))
    out["Bars_status"] = "ok"

    last_ts = df_bars["datetime"].iloc[-1]
    out["Last_bar(CT)"] = fmt_ct(last_ts.to_pydatetime())
    age = minutes_since(last_ts)
    out["Last_bar_age_min"] = age if age is not None else "N/A"
    if age is None:
        out["Stale?"] = "N/A"
    else:
        out["Stale?"] = "YES" if age > stale_after_min else "NO"

    # Volume debug + vol ratio
    vr, last_v, avg_v, vstat = volume_ratio_with_debug(df_bars, lookback=20)
    out["Vol_ratio"] = vr if vr is not None else "N/A"
    out["Last_vol"] = last_v if last_v is not None else "N/A"
    out["Avg20_vol"] = avg_v if avg_v is not None else "N/A"
    out["Vol_status"] = vstat

    # Need a small minimum for stable calculations
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

    out["RSI"] = round(float(rsi_v), 2) if not pd.isna(rsi_v) else "N/A"
    out["MACD_hist"] = round(float(macd_v), 4) if not pd.isna(macd_v) else "N/A"
    out["VWAP_above"] = "Above" if vwap_above else "Below"
    out["EMA_stack"] = "Bull" if ema_stack_bull else ("Bear" if ema_stack_bear else "Neutral")

    # UW biases
    pc_ratio, _ = uw_put_call_bias(flow_df, ticker) if (flow_df is not None and not flow_df.empty) else (None, "N/A")
    out["Put/Call_vol"] = pc_ratio if pc_ratio is not None else "N/A"
    out["UW_bias"] = (
        "PUT" if (pc_ratio is not None and pc_ratio > 1.1)
        else ("CALL" if (pc_ratio is not None and pc_ratio < 0.9) else "Neutral")
    )

    gb, gm = gamma_bias_or_fallback(flow_df, ticker) if (flow_df is not None and not flow_df.empty) else ("N/A", "N/A")
    out["Gamma_bias"] = gb
    out["Gamma_method"] = gm

    bull = 0.0
    bear = 0.0

    # RSI
    if not pd.isna(rsi_v):
        if rsi_v <= 30:
            bull += weights["rsi"]
        elif rsi_v >= 70:
            bear += weights["rsi"]

    # MACD
    if not pd.isna(macd_v):
        if macd_v > 0:
            bull += weights["macd"]
        elif macd_v < 0:
            bear += weights["macd"]

    # VWAP
    bull += weights["vwap"] if vwap_above else 0.0
    bear += weights["vwap"] if not vwap_above else 0.0

    # EMA stack
    if ema_stack_bull:
        bull += weights["ema"]
    elif ema_stack_bear:
        bear += weights["ema"]

    # Volume confirmation only if ratio is valid
    if isinstance(out["Vol_ratio"], (int, float)):
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

    # 10Y filter
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
# UI / Session init
# ============================================================
st.title(APP_TITLE)

if "rotate_idx" not in st.session_state:
    st.session_state.rotate_idx = 0
if "last_good" not in st.session_state:
    st.session_state.last_good = {}
if "iv_hist" not in st.session_state:
    st.session_state.iv_hist = {}


def clear_cache_and_reload():
    try:
        st.cache_data.clear()
    except Exception:
        pass


with st.sidebar:
    st.header("Settings")

    if st.button("Force refresh (clear cache)"):
        clear_cache_and_reload()
        st.success("Cache cleared. App will refresh on next reload.")

    ticker_text = st.text_input(
        "Type tickers (comma-separated) — max 5",
        value="",
        placeholder="SPY,TSLA,AMD,META,IWM",
    )
    tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:5]

    interval = st.selectbox("Candle interval (EODHD)", ["15m", "5m", "1m", "1h"], index=1)  # default 5m
    price_lookback = st.slider("Price lookback (minutes)", 60, 1980, 900, 30)              # default 900
    include_extended = st.toggle("Include extended hours", value=True)

    # Default requested: 60 minutes
    news_lookback = st.slider("News lookback (minutes)", 15, 720, 60, 15)                  # default 60
    refresh_sec = st.slider("Auto-refresh (seconds)", 10, 120, 20, 5)                      # default 20

    st.divider()
    inst_threshold = st.slider("Institutional mode: signals only if confidence ≥", 50, 95, 80, 1)  # default 80
    stale_after_min = st.slider("Mark data as STALE after (minutes)", 5, 180, 45, 5)

    st.divider()
    st.subheader("API Call Mode")
    mode = st.selectbox(
        "How to refresh tickers",
        ["ROTATE (1 ticker per refresh)", "ALL (refresh all tickers)"],
        index=0,
        help="ROTATE keeps API usage low and stable. ALL is heavier.",
    )

    st.divider()
    st.subheader("IV Spike Settings (relative)")
    iv_hist_len = st.slider("IV baseline history length (samples)", 10, 60, 30, 5)
    iv_ratio_thr = st.slider("Spike ratio vs baseline", 1.10, 2.00, 1.35, 0.05)
    iv_abs_thr = st.slider("Spike absolute points above baseline", 5.0, 50.0, 10.0, 1.0)

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
    st.success("UW_TOKEN") if UW_TOKEN else st.warning("UW_TOKEN (missing — flow disabled)")
    st.info("FRED_API_KEY (optional)") if FRED_API_KEY else st.warning("FRED_API_KEY (optional, missing)")

    st.divider()
    st.subheader("Diagnostics")

    if st.button("Test EODHD intraday (SPY 5m)"):
        if not EODHD_API_KEY:
            st.error("EODHD_API_KEY missing.")
        else:
            url = f"{EODHD_BASE}/intraday/SPY.US"
            params = {"api_token": EODHD_API_KEY, "interval": "5m", "fmt": "json"}
            code, text, _ = http_get(url, params=params, timeout=20)
            st.write("HTTP:", code)
            st.code(text[:2000])

    if st.button("Test EODHD IV (SPY)"):
        iv, s = eodhd_options_chain_iv("SPY")
        st.write("IV:", iv, "Status:", s)

    if st.button("Test EODHD News (SPY last 60m)"):
        df, s = eodhd_news("SPY", 60)
        st.write("Status:", s, "Rows:", 0 if df is None else len(df))
        if df is not None and not df.empty:
            st.dataframe(df.head(20), use_container_width=True)

    if st.button("Test UW flow (limit 50)"):
        df, s = uw_flow_alerts(limit=50)
        st.write("Status:", s, "Rows:", 0 if df is None else len(df))
        if df is not None and not df.empty:
            st.dataframe(df.head(20), use_container_width=True)

    if st.button("Test FRED 10Y"):
        v, s = fred_10y_yield()
        st.write("10Y:", v, "Status:", s)


# Auto-refresh
st.caption(f"Last update (CT): {fmt_ct(now_ct())}")
st.markdown(f"<script>setTimeout(()=>window.location.reload(), {refresh_sec*1000});</script>", unsafe_allow_html=True)

# Shared data
ten_y_val, ten_y_status = fred_10y_yield()
flow_df, flow_status = uw_flow_alerts(limit=250) if UW_TOKEN else (pd.DataFrame(), "missing_key")

# Status row (top)
status_cols = st.columns([1, 1, 1], gap="small")


def status_box(label: str, status: str):
    if status == "ok":
        st.success(f"{label} (ok)")
    elif status in ("empty", "N/A", "missing_key"):
        st.warning(f"{label} ({status})")
    elif isinstance(status, str) and status.startswith("http_"):
        st.error(f"{label} ({status})")
    else:
        st.error(f"{label} ({status})")


with status_cols[0]:
    status_box("EODHD (primary)", "ok" if EODHD_API_KEY else "missing_key")
with status_cols[1]:
    status_box("UW flow-alerts", flow_status)
with status_cols[2]:
    status_box("FRED 10Y", ten_y_status)

# Layout
left, right = st.columns([0.33, 0.67], gap="large")

with left:
    st.subheader("Unusual Whales Screener (web view)")
    if not UW_TOKEN:
        st.info("UW_TOKEN missing — embedded screener still loads, but API flow metrics are disabled.")
    st.caption("Embedded. True filtering is best done in UW UI.")
    st.components.v1.iframe("https://unusualwhales.com/options-screener", height=760, scrolling=True)

with right:
    st.subheader("Live Score / Signals (EODHD intraday OHLC + EODHD news/IV + UW flow)")

    if not tickers:
        st.info("Enter up to 5 tickers in the sidebar to start.")
        st.stop()

    # Decide which tickers to refresh this cycle
    if mode.startswith("ROTATE"):
        idx = st.session_state.rotate_idx % len(tickers)
        tickers_to_refresh = [tickers[idx]]
        st.session_state.rotate_idx += 1
        st.caption(f"Refresh mode: ROTATE • Updating: {tickers_to_refresh[0]} • Others use last good cached bars.")
    else:
        tickers_to_refresh = tickers
        st.caption("Refresh mode: ALL • Updating all tickers now.")

    rows: List[Dict[str, Any]] = []
    news_frames: List[pd.DataFrame] = []

    for t in tickers:
        # Fetch fresh bars only for selected ticker(s); otherwise reuse last good
        bars = pd.DataFrame()
        bars_status = "not_refreshed_yet"

        if t in tickers_to_refresh:
            bars, bars_status = eodhd_intraday_bars(
                t, interval=interval, lookback_minutes=price_lookback, include_extended=include_extended
            )
            if bars_status == "ok" and bars is not None and not bars.empty:
                st.session_state.last_good[t] = {"bars": bars, "bars_status": bars_status}
        else:
            cached = st.session_state.last_good.get(t)
            if cached and isinstance(cached.get("bars"), pd.DataFrame) and not cached["bars"].empty:
                bars = cached["bars"]
                bars_status = "cached_last_good"
            else:
                bars = pd.DataFrame()
                bars_status = "no_cache"

        # IV + baseline + spike
        iv_now, iv_status = (None, "missing_key")
        if EODHD_API_KEY:
            iv_now, iv_status = eodhd_options_chain_iv(t)
        iv_base = update_iv_history(t, iv_now, max_len=int(iv_hist_len))

        # News
        news_df, news_status_raw = (pd.DataFrame(), "missing_key")
        if EODHD_API_KEY:
            news_df, news_status_raw = eodhd_news(t, lookback_minutes=news_lookback)
            if news_status_raw == "ok" and news_df is not None and not news_df.empty:
                news_frames.append(news_df)

        news_flag = "YES" if (news_status_raw == "ok" and news_df is not None and not news_df.empty) else "Not Yet"

        out = score_signal(
            df_bars=bars,
            flow_df=flow_df if (flow_status == "ok") else pd.DataFrame(),
            ticker=t,
            iv_now=iv_now,
            iv_base=iv_base,
            ten_y=ten_y_val if ten_y_status == "ok" else None,
            weights=weights,
            stale_after_min=float(stale_after_min),
        )

        # Override IV spike thresholds from sidebar
        # (uses the same logic but with your chosen thresholds)
        out["IV_spike"] = iv_spike_flag(iv_now, iv_base, ratio_thr=float(iv_ratio_thr), abs_thr=float(iv_abs_thr))

        out["Bars_status"] = bars_status
        out["IV_status"] = iv_status
        out["News_status"] = news_flag
        out["UW_flow_status"] = flow_status

        out["institutional"] = "YES" if out["confidence"] >= inst_threshold and out["signal"] != "WAIT" else "NO"
        rows.append(out)

    df_out = pd.DataFrame(rows)

    show_cols = [
        "ticker", "confidence", "direction", "signal", "institutional",
        "RSI", "MACD_hist", "VWAP_above", "EMA_stack",
        "Vol_ratio", "Last_vol", "Avg20_vol", "Vol_status",
        "UW_bias", "Put/Call_vol",
        "IV_now", "IV_base", "IV_spike",
        "Gamma_bias", "Gamma_method",
        "10Y",
        "Bars", "Last_bar(CT)", "Last_bar_age_min", "Stale?",
        "Bars_status", "IV_status", "News_status", "UW_flow_status", "Data_source",
    ]
    for c in show_cols:
        if c not in df_out.columns:
            df_out[c] = "N/A"

    st.dataframe(df_out[show_cols], use_container_width=True, height=320)

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
        st.dataframe(
            news_all[["ticker", "published_ct", "source", "title", "url"]].head(80),
            use_container_width=True,
            height=240
        )
        st.caption("Tip: Click URL column links (or copy/paste).")
