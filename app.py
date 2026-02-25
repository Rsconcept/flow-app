import os, json, math, time, datetime as dt
from typing import Dict, Any, Optional, Tuple, List

import requests
import pandas as pd
import streamlit as st

# Optional autorefresh helper
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Institutional Options Signals — CALLS / PUTS ONLY", layout="wide")
APP_TITLE = "🏛️ Institutional Options Signals — CALLS / PUTS ONLY"

UTC = dt.timezone.utc
CST = dt.timezone(dt.timedelta(hours=-6))  # display only (no DST)
ET = ZoneInfo("America/New_York") if ZoneInfo else None

# -----------------------------
# Secrets / env
# -----------------------------
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

POLYGON_API_KEY = get_secret("POLYGON_API_KEY")
EODHD_API_KEY   = get_secret("EODHD_API_KEY")
UW_TOKEN        = get_secret("UW_TOKEN")
FRED_API_KEY    = get_secret("FRED_API_KEY")
UW_FLOW_ALERTS_URL = get_secret("UW_FLOW_ALERTS_URL") or "https://api.unusualwhales.com/api/option-trades/flow-alerts"

# -----------------------------
# HTTP helpers
# -----------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "flow-app/1.2", "Accept": "application/json,text/plain,*/*"})

def http_get(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, timeout: int = 25) -> Tuple[int, str, Dict[str, str]]:
    try:
        r = SESSION.get(url, headers=headers, params=params, timeout=timeout)
        return r.status_code, r.text, dict(r.headers or {})
    except requests.RequestException as e:
        return 0, str(e), {}

def try_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Returns (json_obj, error_string_if_any)
    """
    try:
        return json.loads(text), None
    except Exception:
        snippet = (text or "").strip().replace("\n", " ")
        snippet = snippet[:140] + ("..." if len(snippet) > 140 else "")
        return None, snippet if snippet else "empty_body"

# -----------------------------
# Time helpers
# -----------------------------
def now_utc() -> dt.datetime:
    return dt.datetime.now(tz=UTC)

def now_cst() -> dt.datetime:
    return dt.datetime.now(tz=CST)

def now_et() -> dt.datetime:
    return dt.datetime.now(tz=ET) if ET else now_utc()

def fmt_cst(ts: Optional[dt.datetime]) -> str:
    if not ts:
        return "N/A"
    try:
        return ts.astimezone(CST).strftime("%Y-%m-%d %H:%M:%S CST")
    except Exception:
        return "N/A"

def age_minutes(ts: Optional[dt.datetime]) -> Optional[float]:
    if not ts:
        return None
    try:
        return round((now_utc() - ts.astimezone(UTC)).total_seconds() / 60.0, 2)
    except Exception:
        return None

def parse_interval_minutes(interval: str) -> int:
    try:
        return int(interval.replace("m", ""))
    except Exception:
        return 5

def in_live_hours(include_extended: bool) -> bool:
    if ET is None:
        return True
    t = now_et()
    if t.weekday() >= 5:
        return False
    h = t.hour + t.minute / 60.0
    if include_extended:
        return (h >= 4.0) and (h <= 20.0)
    return (h >= 9.5) and (h <= 16.0)

# -----------------------------
# Indicators
# -----------------------------
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

# -----------------------------
# Session state caches
# -----------------------------
if "poly_cache" not in st.session_state:
    st.session_state["poly_cache"] = {}   # key -> {"ts": epoch, "df": df, "last_bar": dt}
if "poly_cd_until" not in st.session_state:
    st.session_state["poly_cd_until"] = {}  # key -> epoch time
if "poly_last_request_global" not in st.session_state:
    st.session_state["poly_last_request_global"] = 0.0

if "uw_cache" not in st.session_state:
    st.session_state["uw_cache"] = {"ts": 0.0, "df": pd.DataFrame(), "status": "N/A"}

# -----------------------------
# Polygon intraday (robust)
# -----------------------------
POLYGON_BASE = "https://api.polygon.io"

def polygon_bars(
    ticker: str,
    interval: str,
    lookback_minutes: int,
    include_extended: bool,
    min_fetch_sec: int,
    cooldown_429_sec: int,
    global_spacing_ms: int,
) -> Tuple[pd.DataFrame, str]:
    if not POLYGON_API_KEY:
        return pd.DataFrame(), "missing_key"

    t = ticker.upper().strip()
    mult = parse_interval_minutes(interval)
    key = (t, mult, int(lookback_minutes), bool(include_extended))

    now = time.time()
    cd_until = float(st.session_state["poly_cd_until"].get(key, 0.0))
    cache = st.session_state["poly_cache"].get(key)

    # If in cooldown, return cache if exists
    if now < cd_until:
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], f"http_429 (cooldown {int(cd_until-now)}s, cached)"
        return pd.DataFrame(), f"http_429 (cooldown {int(cd_until-now)}s)"

    # cache hit if fresh enough
    stale = False
    if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
        last_bar = cache.get("last_bar")
        a = age_minutes(last_bar) if isinstance(last_bar, dt.datetime) else None

        if in_live_hours(include_extended) and a is not None and a > max(3.0, mult * 2.4):
            stale = True
        if ET and isinstance(last_bar, dt.datetime) and in_live_hours(include_extended):
            if last_bar.astimezone(ET).date() < now_et().date():
                stale = True

        age_sec = now - float(cache.get("ts", 0.0))
        if (not stale) and (age_sec < float(min_fetch_sec)):
            return cache["df"], f"cached({int(age_sec)}s)"

    # GLOBAL SPACING (prevents bursts on Streamlit Cloud)
    last_global = float(st.session_state.get("poly_last_request_global", 0.0))
    wait = (global_spacing_ms / 1000.0) - (time.time() - last_global)
    if wait > 0:
        time.sleep(min(wait, 1.25))
    st.session_state["poly_last_request_global"] = time.time()

    # Build window to NOW (epoch-ms)
    end_dt = now_utc()
    start_dt = end_dt - dt.timedelta(minutes=lookback_minutes)
    from_ms = int(start_dt.timestamp() * 1000)
    to_ms = int(end_dt.timestamp() * 1000)

    url = f"{POLYGON_BASE}/v2/aggs/ticker/{t}/range/{mult}/minute/{from_ms}/{to_ms}"
    params = {"apiKey": POLYGON_API_KEY, "adjusted": "true", "sort": "asc", "limit": 50000}

    code, text, headers = http_get(url, params=params, timeout=25)

    # Handle 429
    if code == 429:
        st.session_state["poly_cd_until"][key] = time.time() + float(cooldown_429_sec)
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], f"http_429 (cooldown {cooldown_429_sec}s, cached)"
        return pd.DataFrame(), f"http_429 (cooldown {cooldown_429_sec}s)"

    # Non-200
    if code != 200:
        j, nonjson = try_json(text)
        msg = None
        if isinstance(j, dict):
            msg = j.get("error") or j.get("message")
        status = f"http_{code}"
        if msg:
            status += f" — {msg}"
        elif nonjson:
            status += f" — {nonjson}"
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], f"{status} (cached)"
        return pd.DataFrame(), status

    # 200 but might be non-json
    j, nonjson = try_json(text)
    if j is None:
        # show snippet so you SEE what polygon returned
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], f"poly_non_json: {nonjson} (cached)"
        return pd.DataFrame(), f"poly_non_json: {nonjson}"

    # Polygon JSON error payload
    if isinstance(j, dict) and j.get("status") == "ERROR":
        err = j.get("error") or j.get("message") or "unknown_error"
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], f"poly_error: {err} (cached)"
        return pd.DataFrame(), f"poly_error: {err}"

    if not isinstance(j, dict) or "results" not in j:
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], "poly_schema_mismatch (cached)"
        return pd.DataFrame(), "poly_schema_mismatch"

    results = j.get("results") or []
    if not results:
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], "empty (cached)"
        return pd.DataFrame(), "empty"

    raw = pd.DataFrame(results)
    needed = ["t", "o", "h", "l", "c", "v"]
    if any(c not in raw.columns for c in needed):
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], "schema_mismatch (cached)"
        return pd.DataFrame(), "schema_mismatch"

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(raw["t"], unit="ms", utc=True)
    out["open"] = pd.to_numeric(raw["o"], errors="coerce")
    out["high"] = pd.to_numeric(raw["h"], errors="coerce")
    out["low"] = pd.to_numeric(raw["l"], errors="coerce")
    out["close"] = pd.to_numeric(raw["c"], errors="coerce")
    out["volume"] = pd.to_numeric(raw["v"], errors="coerce")
    out = out.dropna(subset=["datetime", "close"]).sort_values("datetime")

    if out.empty:
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], "empty (cached)"
        return pd.DataFrame(), "empty"

    # Filter to regular session if requested
    if (not include_extended) and ET:
        et_times = out["datetime"].dt.tz_convert(ET)
        regular = (
            ((et_times.dt.hour > 9) | ((et_times.dt.hour == 9) & (et_times.dt.minute >= 30)))
            & (et_times.dt.hour < 16)
        )
        out = out[regular].copy()
        if out.empty:
            if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
                return cache["df"], "empty_regular_only (cached)"
            return pd.DataFrame(), "empty_regular_only"

    last_bar = out["datetime"].iloc[-1].to_pydatetime()

    st.session_state["poly_cache"][key] = {"ts": time.time(), "df": out, "last_bar": last_bar}

    a = age_minutes(last_bar)
    if in_live_hours(include_extended) and a is not None and a > max(3.0, mult * 2.4):
        return out, f"stale(age={a}m)"

    return out, ("ok (forced_refresh)" if stale else "ok")

# -----------------------------
# FRED 10Y yield (DGS10)
# -----------------------------
@st.cache_data(ttl=180, show_spinner=False)
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
    j, nonjson = try_json(text)
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

# -----------------------------
# EODHD news + IV (unchanged)
# -----------------------------
EODHD_BASE = "https://eodhd.com/api"

@st.cache_data(ttl=120, show_spinner=False)
def eodhd_news(ticker: str, lookback_minutes: int) -> Tuple[pd.DataFrame, str]:
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"
    t = ticker.upper().strip()
    url = f"{EODHD_BASE}/news"
    params = {"api_token": EODHD_API_KEY, "fmt": "json", "s": f"{t}.US", "limit": 80}
    code, text, _ = http_get(url, params=params, timeout=20)
    if code != 200:
        return pd.DataFrame(), f"http_{code}"
    j, nonjson = try_json(text)
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

    df["published_cst"] = df["published_utc"].dt.tz_convert(CST)
    cutoff = dt.datetime.now(tz=CST) - dt.timedelta(minutes=lookback_minutes)
    df = df[df["published_cst"] >= cutoff].copy()
    if df.empty:
        return pd.DataFrame(), "ok"

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
    out["published_cst"] = df["published_cst"].dt.strftime("%Y-%m-%d %H:%M:%S CST")
    out["source"] = df[src_c] if src_c else ""
    out["title"] = df[title_c] if title_c else ""
    out["url"] = df[url_c] if url_c else ""
    out = out.dropna(subset=["title"]).head(60)
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
    j, nonjson = try_json(text)
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

# -----------------------------
# Unusual Whales
# -----------------------------
def uw_headers() -> Dict[str, str]:
    return {"Accept": "application/json, text/plain", "Authorization": f"Bearer {UW_TOKEN}" if UW_TOKEN else ""}

def uw_flow_alerts(limit: int = 250) -> Tuple[pd.DataFrame, str]:
    if not UW_TOKEN:
        return pd.DataFrame(), "missing_key"

    now = time.time()
    cache = st.session_state["uw_cache"]
    if isinstance(cache.get("df"), pd.DataFrame) and (now - float(cache.get("ts", 0.0)) < 20):
        return cache["df"], cache.get("status", "ok")

    code, text, _ = http_get(UW_FLOW_ALERTS_URL, headers=uw_headers(), params={"limit": limit}, timeout=20)
    if code != 200:
        st.session_state["uw_cache"] = {"ts": time.time(), "df": pd.DataFrame(), "status": f"http_{code}"}
        return pd.DataFrame(), f"http_{code}"

    j, nonjson = try_json(text)
    if not isinstance(j, dict) or "data" not in j:
        st.session_state["uw_cache"] = {"ts": time.time(), "df": pd.DataFrame(), "status": "parse_error"}
        return pd.DataFrame(), "parse_error"

    data = j.get("data") or []
    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
    st.session_state["uw_cache"] = {"ts": time.time(), "df": df, "status": "ok"}
    return df, "ok"

def uw_put_call_bias(flow_df: pd.DataFrame, ticker: str) -> Tuple[Optional[float], str]:
    if flow_df is None or flow_df.empty:
        return None, "N/A"
    t = ticker.upper()
    df = flow_df.copy()

    for sym_col in ["underlying_symbol", "ticker", "symbol"]:
        if sym_col in df.columns:
            df = df[df[sym_col].astype(str).str.upper() == t]
            break

    if df.empty or "option_type" not in df.columns:
        return None, "N/A"

    types = df["option_type"].astype(str).str.lower()
    vol_series = None
    for c in ["volume", "size", "total_size", "contracts"]:
        if c in df.columns:
            vol_series = pd.to_numeric(df[c], errors="coerce").fillna(0)
            break
    if vol_series is None:
        vol_series = pd.Series([1.0] * len(df))

    put_vol = float(vol_series[types == "put"].sum())
    call_vol = float(vol_series[types == "call"].sum())
    if call_vol <= 0:
        return None, "N/A"
    return round(put_vol / call_vol, 2), "ok"

def gamma_bias_proxy(flow_df: pd.DataFrame, ticker: str) -> str:
    if flow_df is None or flow_df.empty:
        return "N/A"
    t = ticker.upper()
    df = flow_df.copy()

    for sym_col in ["underlying_symbol", "ticker", "symbol"]:
        if sym_col in df.columns:
            df = df[df[sym_col].astype(str).str.upper() == t]
            break
    if df.empty:
        return "N/A"

    opt_type = df["option_type"].astype(str).str.lower() if "option_type" in df.columns else pd.Series([""] * len(df))
    sign = opt_type.map(lambda x: 1.0 if x == "call" else (-1.0 if x == "put" else 0.0))

    size = None
    for c in ["size", "total_size", "volume", "contracts"]:
        if c in df.columns:
            size = pd.to_numeric(df[c], errors="coerce").fillna(0)
            break
    if size is None:
        size = pd.Series([1.0] * len(df))

    if "gamma" in df.columns:
        g = pd.to_numeric(df["gamma"], errors="coerce").fillna(0)
        score = float((g * size * sign).sum())
        if abs(score) < 0.5:
            return "Neutral"
        return "Positive" if score > 0 else "Negative"

    prem = None
    for c in ["premium", "total_premium", "total_premium_usd"]:
        if c in df.columns:
            prem = pd.to_numeric(df[c], errors="coerce").fillna(0)
            break
    if prem is None:
        prem = size

    score = float((prem * sign).sum())
    if abs(score) < 1e-6:
        return "Neutral (premium proxy)"
    return ("Bullish (premium proxy)" if score > 0 else "Bearish (premium proxy)")

# -----------------------------
# Scoring (includes 10Y)
# -----------------------------
def score_signal(
    df_bars: pd.DataFrame,
    flow_df: pd.DataFrame,
    ticker: str,
    iv_now: Optional[float],
    ten_y: Optional[float],
    weights: Dict[str, float],
    interval: str,
    include_extended: bool,
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
        "Age_min": "N/A",
        "Bars_status": "empty",
        "IV_status": "ok",
        "News_status": "Not Yet",
        "UW_flow_status": "ok",
    }

    if df_bars is None or df_bars.empty or len(df_bars) < 30:
        return out

    out["Bars"] = int(len(df_bars))
    last_bar_dt = df_bars["datetime"].iloc[-1].to_pydatetime()
    out["Last_bar(CST)"] = fmt_cst(last_bar_dt)
    a = age_minutes(last_bar_dt)
    out["Age_min"] = a if a is not None else "N/A"
    out["Bars_status"] = "ok"

    close = df_bars["close"].astype(float)

    rsi_v = rsi(close, 14).iloc[-1]
    macd_v = macd_hist(close).iloc[-1]
    vwap_line = vwap(df_bars)
    vwap_above = bool(close.iloc[-1] >= vwap_line.iloc[-1])

    ema9 = ema(close, 9).iloc[-1]
    ema20 = ema(close, 20).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
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
    out["UW_bias"] = "PUT" if (pc_ratio is not None and pc_ratio > 1.1) else ("CALL" if (pc_ratio is not None and pc_ratio < 0.9) else "Neutral")
    out["Gamma_bias"] = gamma_bias_proxy(flow_df, ticker)

    if iv_now is None:
        out["IV_spike"] = "N/A"
    else:
        out["IV_spike"] = "YES" if iv_now >= 65 else "NO"

    mult = parse_interval_minutes(interval)
    if in_live_hours(include_extended) and a is not None and a > max(3.0, mult * 2.4):
        out["Bars_status"] = f"stale(age={a}m)"

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

    (bull if vwap_above else bear).__iadd__(weights["vwap"])

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

    ticker_text = st.text_input("Type any tickers (comma-separated).", value="SPY,TSLA,AMD,META")
    tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:9]  # limit 9 tickers

    interval = st.selectbox("Candle interval", ["5m", "1m", "15m"], index=0)
    price_lookback = st.slider("Price lookback (minutes)", 60, 1980, 900, 30)
    include_extended = st.toggle("Include pre/after-hours (Polygon)", value=True)
    news_lookback = st.slider("News lookback (minutes)", 15, 720, 360, 15)
    refresh_sec = st.slider("Auto-refresh (seconds)", 10, 120, 20, 5)

    inst_threshold = st.slider("Institutional mode: signals only if confidence ≥", 50, 95, 80, 1)

    st.divider()
    st.subheader("Throttle controls (Polygon)")
    min_fetch_sec = st.number_input("Min seconds between Polygon requests per ticker", 10, 600, 90, 5)
    cooldown_429_sec = st.number_input("Cooldown after 429 (seconds)", 30, 900, 180, 10)
    global_spacing_ms = st.number_input("Global spacing between Polygon requests (ms)", 0, 2000, 350, 50)

    st.divider()
    st.caption("Weights")
    w_rsi = st.slider("RSI weight", 0.00, 0.30, 0.15, 0.01)
    w_macd = st.slider("MACD weight", 0.00, 0.30, 0.15, 0.01)
    w_vwap = st.slider("VWAP weight", 0.00, 0.30, 0.15, 0.01)
    w_ema  = st.slider("EMA stack weight", 0.00, 0.30, 0.18, 0.01)
    w_vol  = st.slider("Volume ratio weight", 0.00, 0.30, 0.12, 0.01)
    w_uw   = st.slider("UW flow weight", 0.00, 0.40, 0.20, 0.01)
    w_teny = st.slider("10Y yield weight", 0.00, 0.20, 0.05, 0.01)
    weights = {"rsi": w_rsi, "macd": w_macd, "vwap": w_vwap, "ema": w_ema, "vol": w_vol, "uw": w_uw, "teny": w_teny}

    st.divider()
    st.subheader("Keys status")
    st.success("POLYGON_API_KEY") if POLYGON_API_KEY else st.error("POLYGON_API_KEY missing")
    st.success("UW_TOKEN") if UW_TOKEN else st.error("UW_TOKEN missing")
    st.success("EODHD_API_KEY") if EODHD_API_KEY else st.error("EODHD_API_KEY missing")
    st.success("FRED_API_KEY") if FRED_API_KEY else st.error("FRED_API_KEY missing (10Y won't work)")

st.caption(f"Now (CST): {fmt_cst(now_cst())} | Interval={interval} | Refresh={refresh_sec}s | min/ticker={min_fetch_sec}s")

if HAS_AUTOREFRESH:
    st_autorefresh(interval=int(refresh_sec * 1000), key="autorefresh")
else:
    st.markdown(f"<script>setTimeout(()=>window.location.reload(), {int(refresh_sec*1000)});</script>", unsafe_allow_html=True)

# Shared data
ten_y_val, ten_y_status = fred_10y_yield()
flow_df, flow_status = uw_flow_alerts(limit=250)

# Status bar
cols = st.columns(4)
def pill(col, label, status):
    with col:
        if status == "ok":
            st.success(f"{label} (ok)")
        elif status in ("missing_key", "N/A", "empty"):
            st.warning(f"{label} ({status})")
        else:
            st.error(f"{label} ({status})")

pill(cols[0], "UW flow-alerts", flow_status)
pill(cols[1], "Polygon intraday", "ok" if POLYGON_API_KEY else "missing_key")
pill(cols[2], "EODHD news/IV", "ok" if EODHD_API_KEY else "missing_key")
pill(cols[3], "FRED 10Y", ten_y_status)

left, right = st.columns([0.35, 0.65], gap="large")

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.components.v1.iframe("https://unusualwhales.com/options-screener", height=720, scrolling=True)

with right:
    st.subheader("Live Score / Signals (Polygon intraday + EODHD headlines + UW flow + FRED 10Y)")

    rows = []
    news_frames = []

    for t in tickers:
        bars, bars_status = polygon_bars(
            ticker=t,
            interval=interval,
            lookback_minutes=int(price_lookback),
            include_extended=bool(include_extended),
            min_fetch_sec=int(min_fetch_sec),
            cooldown_429_sec=int(cooldown_429_sec),
            global_spacing_ms=int(global_spacing_ms),
        )

        iv_now, iv_status = eodhd_options_chain_iv(t)
        news_df, _news_status = eodhd_news(t, lookback_minutes=int(news_lookback))

        news_yes = (news_df is not None) and (not news_df.empty)
        if news_yes:
            news_frames.append(news_df)

        out = score_signal(
            df_bars=bars,
            flow_df=flow_df if flow_status == "ok" else pd.DataFrame(),
            ticker=t,
            iv_now=iv_now,
            ten_y=ten_y_val if ten_y_status == "ok" else None,
            weights=weights,
            interval=interval,
            include_extended=bool(include_extended),
        )

        out["Bars_status"] = bars_status or out.get("Bars_status", "empty")
        out["IV_status"] = iv_status
        out["News_status"] = "YES" if news_yes else "Not Yet"
        out["UW_flow_status"] = flow_status
        out["institutional"] = "YES" if (out["confidence"] >= inst_threshold and out["signal"] != "WAIT") else "NO"

        rows.append(out)

    df_out = pd.DataFrame(rows)

    show_cols = [
        "ticker","confidence","direction","signal","institutional",
        "RSI","MACD_hist","VWAP_above","EMA_stack","Vol_ratio",
        "UW_bias","Put/Call_vol","IV_now","IV_spike","Gamma_bias","10Y",
        "Bars","Last_bar(CST)","Age_min",
        "Bars_status","IV_status","News_status","UW_flow_status"
    ]
    for c in show_cols:
        if c not in df_out.columns:
            df_out[c] = "N/A"

    st.dataframe(df_out[show_cols], use_container_width=True, height=260)

    st.subheader(f"Institutional Alerts (≥ {inst_threshold} only)")
    inst = df_out[(df_out["institutional"] == "YES")].copy()
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        for _, r in inst.sort_values("confidence", ascending=False).iterrows():
            st.success(f"{r['ticker']}: {r['signal']} • {r['direction']} • Confidence={int(r['confidence'])}")

    st.subheader(f"News — last {news_lookback} minutes (EODHD)")
    if not news_frames:
        st.info("No news in this lookback window.")
    else:
        news_all = pd.concat(news_frames, ignore_index=True)
        for c in ["ticker","published_cst","source","title","url"]:
            if c not in news_all.columns:
                news_all[c] = ""
        st.dataframe(news_all[["ticker","published_cst","source","title","url"]].head(60), use_container_width=True, height=240)
