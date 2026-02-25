# app.py
# ============================================================
# Institutional Options Signals (5m) — CALLS / PUTS ONLY
# FIXES:
#  - Polygon "stale cache" issue (02/24 showing during 02/25 live)
#  - Forces refresh when last bar is stale
#  - Polygon end-date inclusion: request through tomorrow
#  - 429 cooldown per ticker + won't lock you into yesterday
#  - Gamma bias: if UW gamma missing, use NET FLOW premium proxy
#  - Auto-refresh: streamlit-autorefresh if available, JS fallback always
# ============================================================

import os, json, math, time, datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

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
# App config
# -----------------------------
st.set_page_config(
    page_title="Institutional Options Signals (5m) — CALLS / PUTS ONLY",
    layout="wide",
)

APP_TITLE = "🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY"
UTC = dt.timezone.utc
CST = dt.timezone(dt.timedelta(hours=-6))  # display only
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

POLYGON_API_KEY = get_secret("POLYGON_API_KEY")  # REQUIRED
UW_TOKEN = get_secret("UW_TOKEN")                # REQUIRED
EODHD_API_KEY = get_secret("EODHD_API_KEY")      # OPTIONAL
FRED_API_KEY = get_secret("FRED_API_KEY")        # OPTIONAL
UW_FLOW_ALERTS_URL = get_secret("UW_FLOW_ALERTS_URL") or "https://api.unusualwhales.com/api/option-trades/flow-alerts"

# -----------------------------
# HTTP
# -----------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "streamlit-options-signals/1.0"})

def http_get(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, timeout: int = 20) -> Tuple[int, str, str]:
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

# -----------------------------
# Time helpers
# -----------------------------
def now_cst() -> dt.datetime:
    return dt.datetime.now(tz=CST)

def now_et() -> dt.datetime:
    if ET:
        return dt.datetime.now(tz=ET)
    return dt.datetime.now(tz=UTC)

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
        return round((dt.datetime.now(tz=UTC) - ts.astimezone(UTC)).total_seconds() / 60.0, 2)
    except Exception:
        return None

def parse_interval_minutes(interval: str) -> int:
    # supports "1m", "5m", "15m"
    try:
        return int(interval.replace("m", ""))
    except Exception:
        return 5

def in_live_hours(include_extended: bool) -> bool:
    # If extended: treat 04:00–20:00 ET as "live enough"
    # else: 09:30–16:00 ET.
    if ET is None:
        return True
    t = now_et()
    if t.weekday() >= 5:
        return False
    h = t.hour + t.minute/60.0
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
# Session state
# -----------------------------
def _ss_init():
    if "polygon_cache" not in st.session_state:
        # key -> {"ts": epoch, "df": DataFrame, "last_bar_utc": datetime}
        st.session_state["polygon_cache"] = {}
    if "polygon_cooldown_until" not in st.session_state:
        st.session_state["polygon_cooldown_until"] = {}
    if "uw_cache" not in st.session_state:
        st.session_state["uw_cache"] = {"ts": 0.0, "df": pd.DataFrame(), "status": "N/A"}

_ss_init()

# -----------------------------
# Polygon bars (stale-aware + cooldown)
# -----------------------------
POLYGON_BASE = "https://api.polygon.io"

def polygon_intraday_bars(
    ticker: str,
    interval: str,
    lookback_minutes: int,
    include_extended: bool,
    min_fetch_sec: int,
    cooldown_on_429_sec: int,
) -> Tuple[pd.DataFrame, str]:
    if not POLYGON_API_KEY:
        return pd.DataFrame(), "missing_key"

    t = ticker.upper().strip()
    interval_min = parse_interval_minutes(interval)

    key = (t, interval, int(lookback_minutes), bool(include_extended))
    now = time.time()

    cache = st.session_state["polygon_cache"].get(key)
    cd_until = float(st.session_state["polygon_cooldown_until"].get(key, 0.0))

    # Determine if cache is stale (last bar too old during live hours)
    stale = False
    if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
        last_bar = cache.get("last_bar_utc")
        if isinstance(last_bar, dt.datetime):
            a = age_minutes(last_bar)
            if a is not None:
                # If we're in live hours, last bar should NOT be older than ~2 candles.
                # Example: 5m -> stale if > 12 minutes
                if in_live_hours(include_extended) and a > max(3.0, interval_min * 2.4):
                    stale = True

        # If ET day changed, force refresh once (prevents yesterday-only cache)
        if ET is not None and isinstance(last_bar, dt.datetime):
            if last_bar.astimezone(ET).date() < now_et().date() and in_live_hours(include_extended):
                stale = True

    # Cooldown: if 429 and we're cooling down, return cache if any, else cooldown
    if now < cd_until:
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], f"cooldown({int(cd_until-now)}s) using cached"
        return pd.DataFrame(), f"cooldown({int(cd_until-now)}s)"

    # Fresh cache check ONLY if not stale
    if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty and not stale:
        age = now - float(cache.get("ts", 0.0))
        if age < float(min_fetch_sec):
            return cache["df"], f"cached({int(age)}s)"

    # Parse multiplier
    try:
        multiplier = int(interval.replace("m", ""))
    except Exception:
        return pd.DataFrame(), "bad_interval"

    end_utc = dt.datetime.now(tz=UTC)
    start_utc = end_utc - dt.timedelta(minutes=lookback_minutes)

    # Polygon range endpoint uses dates; to avoid "end date exclusion" issues, request through TOMORROW
    if ET is not None:
        from_date = start_utc.astimezone(ET).date().isoformat()
        to_date = (end_utc.astimezone(ET).date() + dt.timedelta(days=1)).isoformat()
    else:
        from_date = start_utc.date().isoformat()
        to_date = (end_utc.date() + dt.timedelta(days=1)).isoformat()

    url = f"{POLYGON_BASE}/v2/aggs/ticker/{t}/range/{multiplier}/minute/{from_date}/{to_date}"
    params = {"apiKey": POLYGON_API_KEY, "adjusted": "true", "sort": "asc", "limit": 50000}

    code, text, _ = http_get(url, params=params, timeout=20)

    if code == 429:
        # set cooldown and return cache if any
        st.session_state["polygon_cooldown_until"][key] = time.time() + float(cooldown_on_429_sec)
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], f"http_429 (cooldown {cooldown_on_429_sec}s, using cached)"
        return pd.DataFrame(), f"http_429 (cooldown {cooldown_on_429_sec}s)"

    if code != 200:
        j = safe_json(text)
        if isinstance(j, dict) and j.get("error"):
            if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
                return cache["df"], f"http_{code} — {j.get('error')} (using cached)"
            return pd.DataFrame(), f"http_{code} — {j.get('error')}"
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], f"http_{code} (using cached)"
        return pd.DataFrame(), f"http_{code}"

    j = safe_json(text)
    if not isinstance(j, dict) or "results" not in j:
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], "parse_error (using cached)"
        return pd.DataFrame(), "parse_error"

    results = j.get("results") or []
    if not results:
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], "empty (using cached)"
        return pd.DataFrame(), "empty"

    df = pd.DataFrame(results)
    needed = ["t", "o", "h", "l", "c", "v"]
    if any(c not in df.columns for c in needed):
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], "schema_mismatch (using cached)"
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
        if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
            return cache["df"], "empty (using cached)"
        return pd.DataFrame(), "empty"

    # Regular hours filter if needed
    if not include_extended and ET is not None:
        et_times = out["datetime"].dt.tz_convert(ET)
        regular = (
            ((et_times.dt.hour > 9) | ((et_times.dt.hour == 9) & (et_times.dt.minute >= 30)))
            & (et_times.dt.hour < 16)
        )
        out = out[regular].copy()
        if out.empty:
            if cache and isinstance(cache.get("df"), pd.DataFrame) and not cache["df"].empty:
                return cache["df"], "empty_regular_only (using cached)"
            return pd.DataFrame(), "empty_regular_only"

    # Clip to lookback
    cutoff = end_utc - dt.timedelta(minutes=lookback_minutes)
    clipped = out[out["datetime"] >= cutoff].copy()
    if not clipped.empty:
        out = clipped

    last_bar_utc = out["datetime"].iloc[-1].to_pydatetime()

    # Save cache
    st.session_state["polygon_cache"][key] = {"ts": time.time(), "df": out, "last_bar_utc": last_bar_utc}

    # polite spacing
    time.sleep(0.12)

    return out, ("ok (forced_refresh)" if stale else "ok")

# -----------------------------
# EODHD (optional) - minimal (same as before)
# -----------------------------
EODHD_BASE = "https://eodhd.com/api"

@st.cache_data(ttl=90, show_spinner=False)
def eodhd_news(ticker: str, lookback_minutes: int) -> Tuple[pd.DataFrame, str]:
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"

    t = ticker.upper().strip()
    url = f"{EODHD_BASE}/news"
    params = {"api_token": EODHD_API_KEY, "fmt": "json", "s": f"{t}.US", "limit": 80}

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
    out["ticker"] = t
    out["published_cst"] = df["published_cst"].dt.strftime("%Y-%m-%d %H:%M:%S CST")
    out["source"] = df[src_c] if src_c else ""
    out["title"] = df[title_c] if title_c else ""
    out["url"] = df[url_c] if url_c else ""
    out = out.dropna(subset=["title"]).head(120)
    return out, "ok"

# -----------------------------
# Unusual Whales (cache + gamma proxy)
# -----------------------------
def uw_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}" if UW_TOKEN else "",
    }

def uw_flow_alerts(limit: int = 250, ttl_sec: int = 25) -> Tuple[pd.DataFrame, str]:
    if not UW_TOKEN:
        return pd.DataFrame(), "missing_key"

    now = time.time()
    c = st.session_state["uw_cache"]
    if (now - float(c["ts"])) < float(ttl_sec) and isinstance(c.get("df"), pd.DataFrame) and not c["df"].empty:
        return c["df"], f"cached({int(now - c['ts'])}s)"

    code, text, _ = http_get(UW_FLOW_ALERTS_URL, headers=uw_headers(), params={"limit": limit}, timeout=20)
    if code != 200:
        if isinstance(c.get("df"), pd.DataFrame) and not c["df"].empty:
            return c["df"], f"http_{code} (using cached)"
        return pd.DataFrame(), f"http_{code}"

    j = safe_json(text)
    if isinstance(j, dict) and "data" in j:
        data = j.get("data") or []
        if not isinstance(data, list) or not data:
            return pd.DataFrame(), "ok"
        df = pd.DataFrame(data)
    elif isinstance(j, list):
        df = pd.DataFrame(j)
    else:
        return pd.DataFrame(), "parse_error"

    # Some responses have nested greeks
    if "greeks" in df.columns and "gamma" not in df.columns:
        try:
            df["gamma"] = df["greeks"].apply(lambda x: x.get("gamma") if isinstance(x, dict) else None)
        except Exception:
            pass

    st.session_state["uw_cache"] = {"ts": now, "df": df, "status": "ok"}
    return df, "ok"

def _pick(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def uw_put_call_ratio(flow_df: pd.DataFrame, ticker: str) -> Tuple[Optional[float], str]:
    if flow_df is None or flow_df.empty:
        return None, "N/A"
    ucol = _pick(flow_df, ["underlying_symbol", "underlying", "ticker", "symbol", "root_symbol"])
    tcol = _pick(flow_df, ["option_type", "type"])
    vcol = _pick(flow_df, ["volume", "size", "contracts", "qty", "quantity"])
    if tcol is None:
        return None, "N/A"

    df = flow_df.copy()
    t = ticker.upper().strip()
    if ucol:
        df = df[df[ucol].astype(str).str.upper() == t]
    if df.empty:
        return None, "N/A"

    types = df[tcol].astype(str).str.lower()
    vols = pd.to_numeric(df[vcol], errors="coerce").fillna(1.0) if vcol else pd.Series([1.0] * len(df))
    put_vol = float(vols[types.str.contains("put")].sum())
    call_vol = float(vols[types.str.contains("call")].sum())
    if call_vol <= 0:
        return None, "N/A"
    return round(put_vol / call_vol, 2), "ok"

def uw_flow_bias(flow_df: pd.DataFrame, ticker: str) -> str:
    """
    If gamma exists -> gamma-weighted direction.
    If gamma missing -> premium-weighted direction proxy (still useful).
    """
    if flow_df is None or flow_df.empty:
        return "N/A"

    ucol = _pick(flow_df, ["underlying_symbol", "underlying", "ticker", "symbol", "root_symbol"])
    tcol = _pick(flow_df, ["option_type", "type"])
    gcol = _pick(flow_df, ["gamma"])
    pcol = _pick(flow_df, ["total_premium", "premium", "premium_num", "notional", "total_value"])
    scol = _pick(flow_df, ["size", "volume", "contracts", "qty", "quantity"])

    if tcol is None:
        return "N/A"

    df = flow_df.copy()
    t = ticker.upper().strip()
    if ucol:
        df = df[df[ucol].astype(str).str.upper() == t].copy()
    if df.empty:
        return "N/A"

    opt = df[tcol].astype(str).str.lower()
    sign = opt.map(lambda x: 1.0 if "call" in x else (-1.0 if "put" in x else 0.0))

    size = pd.to_numeric(df[scol], errors="coerce").fillna(1.0) if scol else pd.Series([1.0] * len(df))

    if gcol is not None:
        gamma = pd.to_numeric(df[gcol], errors="coerce").fillna(0.0)
        score = float((gamma * size * sign).sum())
        if abs(score) < 0.5:
            return "Neutral"
        return "Positive" if score > 0 else "Negative"

    # No gamma -> premium proxy
    prem = pd.to_numeric(df[pcol], errors="coerce").fillna(0.0) if pcol else (size * 1.0)
    score = float((prem * sign).sum())
    if abs(score) < 1e-9:
        return "Neutral (premium proxy)"
    return ("Bullish (premium proxy)" if score > 0 else "Bearish (premium proxy)")

# -----------------------------
# Score
# -----------------------------
def score_signal(
    df_bars: pd.DataFrame,
    flow_df: pd.DataFrame,
    ticker: str,
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
        "Gamma_bias": "N/A",
        "10Y": ten_y if ten_y is not None else "N/A",
        "Bars": 0,
        "Last_bar(CST)": "N/A",
        "Age_min": "N/A",
    }

    if df_bars is None or df_bars.empty:
        return out

    out["Bars"] = int(len(df_bars))
    last_bar_utc = df_bars["datetime"].iloc[-1].to_pydatetime()
    out["Last_bar(CST)"] = fmt_cst(last_bar_utc)
    out["Age_min"] = age_minutes(last_bar_utc) if last_bar_utc else "N/A"

    if len(df_bars) < 10:
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

    pc_ratio, _ = uw_put_call_ratio(flow_df, ticker)
    out["Put/Call_vol"] = pc_ratio if pc_ratio is not None else "N/A"
    out["UW_bias"] = (
        "PUT" if (pc_ratio is not None and pc_ratio > 1.1)
        else ("CALL" if (pc_ratio is not None and pc_ratio < 0.9) else "Neutral")
    )
    out["Gamma_bias"] = uw_flow_bias(flow_df, ticker)

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

    # 10Y: optional heuristic
    if ten_y is not None:
        if ten_y >= 4.75:
            bear += weights["teny"]
        elif ten_y <= 4.0:
            bull += weights["teny"]

    total = bull + bear
    conf = 50 if total <= 0 else int(round(50 + 50 * (abs(bull - bear) / total)))
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

# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)

with st.sidebar:
    st.header("Settings")
    tickers_in = st.text_input("Type any tickers (comma-separated).", value="SPY,TSLA,AMD,META")
    tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:25]

    interval = st.selectbox("Candle interval", ["15m", "5m", "1m"], index=1)  # default 5m
    price_lookback = st.slider("Price lookback (minutes)", 60, 1980, 900, 30)  # default 900
    include_extended = st.toggle("Include pre/after-hours (Polygon)", value=True)

    news_lookback = st.slider("News lookback (minutes)", 15, 720, 360, 15)  # default 360
    refresh_sec = st.slider("Auto-refresh (seconds)", 5, 120, 20, 1)        # default 20
    inst_threshold = st.slider("Institutional mode: signals only if confidence ≥", 50, 95, 80, 1)  # default 80

    st.divider()
    min_polygon_fetch = st.slider("Min seconds between Polygon requests per ticker", 10, 240, 90, 5)
    cooldown_429 = st.slider("Cooldown seconds after Polygon 429", 20, 600, 180, 10)

    st.divider()
    st.caption("Weights")
    w_rsi = st.slider("RSI weight", 0.00, 0.30, 0.15, 0.01)
    w_macd = st.slider("MACD weight", 0.00, 0.30, 0.15, 0.01)
    w_vwap = st.slider("VWAP weight", 0.00, 0.30, 0.15, 0.01)
    w_ema = st.slider("EMA stack weight", 0.00, 0.30, 0.18, 0.01)
    w_vol = st.slider("Volume ratio weight", 0.00, 0.30, 0.12, 0.01)
    w_uw = st.slider("UW flow weight", 0.00, 0.40, 0.20, 0.01)
    w_teny = st.slider("10Y yield weight", 0.00, 0.20, 0.05, 0.01)
    weights = {"rsi": w_rsi, "macd": w_macd, "vwap": w_vwap, "ema": w_ema, "vol": w_vol, "uw": w_uw, "teny": w_teny}

    st.divider()
    st.subheader("Keys status")
    st.success("POLYGON_API_KEY") if POLYGON_API_KEY else st.error("POLYGON_API_KEY (missing)")
    st.success("UW_TOKEN") if UW_TOKEN else st.error("UW_TOKEN (missing)")
    st.info("EODHD_API_KEY (optional)") if EODHD_API_KEY else st.warning("EODHD_API_KEY (optional, missing)")

# Auto refresh: prefer plugin, but ALWAYS add JS fallback
if HAS_AUTOREFRESH:
    st_autorefresh(interval=refresh_sec * 1000, key="autorefresh_v4")
st.markdown(
    f"<script>setTimeout(()=>window.location.reload(), {refresh_sec*1000});</script>",
    unsafe_allow_html=True
)

st.caption(
    f"Now (CST): {fmt_cst(now_cst())} | Interval={interval} | Refresh={refresh_sec}s | "
    f"LiveHours={in_live_hours(include_extended)}"
)

# Fetch UW once
flow_df, flow_status = uw_flow_alerts(limit=250, ttl_sec=25)

# Status bar
status_cols = st.columns([1, 1, 1], gap="small")
with status_cols[0]:
    st.success(f"UW flow-alerts ({flow_status})") if "ok" in str(flow_status) or "cached" in str(flow_status) else st.warning(f"UW flow-alerts ({flow_status})")
with status_cols[1]:
    st.success("Polygon intraday (ok)") if POLYGON_API_KEY else st.error("Polygon intraday (missing_key)")
with status_cols[2]:
    st.success("EODHD news (ok)") if EODHD_API_KEY else st.warning("EODHD news (missing_key)")

left, right = st.columns([0.33, 0.67], gap="large")

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.components.v1.iframe("https://unusualwhales.com/options-screener", height=760, scrolling=True)

with right:
    st.subheader("Live Score / Signals (Polygon intraday + EODHD headlines + UW flow)")

    rows: List[Dict[str, Any]] = []
    news_frames: List[pd.DataFrame] = []

    for t in tickers:
        bars, bars_status = polygon_intraday_bars(
            t,
            interval=interval,
            lookback_minutes=price_lookback,
            include_extended=include_extended,
            min_fetch_sec=min_polygon_fetch,
            cooldown_on_429_sec=int(cooldown_429),
        )

        # News flag per your rule: YES / Not Yet
        news_flag = "Not Yet"
        if EODHD_API_KEY:
            news_df, ns = eodhd_news(t, lookback_minutes=news_lookback)
            if ns == "ok" and news_df is not None and not news_df.empty:
                news_flag = "YES"
                news_frames.append(news_df)

        out = score_signal(
            df_bars=bars,
            flow_df=flow_df if ("ok" in str(flow_status) or "cached" in str(flow_status)) else pd.DataFrame(),
            ticker=t,
            ten_y=None,
            weights=weights,
        )
        out["Bars_status"] = bars_status
        out["News_status"] = news_flag
        out["UW_flow_status"] = flow_status
        out["institutional"] = "YES" if out["confidence"] >= inst_threshold and out["signal"] != "WAIT" else "NO"

        rows.append(out)

    df_out = pd.DataFrame(rows)

    show_cols = [
        "ticker", "confidence", "direction", "signal", "institutional",
        "RSI", "MACD_hist", "VWAP_above", "EMA_stack", "Vol_ratio",
        "UW_bias", "Put/Call_vol", "Gamma_bias",
        "Bars", "Last_bar(CST)", "Age_min", "Bars_status", "News_status", "UW_flow_status",
    ]
    for c in show_cols:
        if c not in df_out.columns:
            df_out[c] = "N/A"

    st.dataframe(df_out[show_cols], use_container_width=True, height=330)

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
        for c in ["ticker", "published_cst", "source", "title", "url"]:
            if c not in news_all.columns:
                news_all[c] = ""
        st.dataframe(news_all[["ticker", "published_cst", "source", "title", "url"]].head(80), use_container_width=True, height=220)
