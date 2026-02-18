import os
import math
import time
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import streamlit as st

# =========================
#   CONFIG
# =========================
st.set_page_config(page_title="Institutional Options Signals (5m) — CALLS / PUTS ONLY", layout="wide")

APP_TITLE = "🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY"
CST_TZ = dt.timezone(dt.timedelta(hours=-6))  # CST (fixed -6, no DST handling)

DEFAULT_QUICK = ["SPY", "QQQ", "IWM", "DIA", "TSLA", "NVDA", "AMD"]
DEFAULT_TICKERS = ["SPY", "TSLA"]

UA = "Mozilla/5.0 (Streamlit; Institutional Options Signals)"

# =========================
#   SECRETS / KEYS
# =========================
def get_secret(name: str) -> Optional[str]:
    # Streamlit secrets first, then env
    if hasattr(st, "secrets") and name in st.secrets:
        val = st.secrets.get(name)
        return str(val).strip() if val else None
    val = os.getenv(name)
    return str(val).strip() if val else None

EODHD_API_KEY = get_secret("EODHD_API_KEY")
UW_TOKEN = get_secret("UW_TOKEN")
UW_FLOW_ALERTS_URL = get_secret("UW_FLOW_ALERTS_URL") or "https://api.unusualwhales.com/api/option-trades/flow-alerts"
FRED_API_KEY = get_secret("FRED_API_KEY")

# Optional (not used for data in this build; kept for your future)
FINVIZ_AUTH = get_secret("FINVIZ_AUTH")

# =========================
#   HTTP helpers
# =========================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

def safe_get(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, timeout: int = 20) -> Tuple[Optional[requests.Response], Optional[str]]:
    try:
        r = SESSION.get(url, headers=headers, params=params, timeout=timeout)
        return r, None
    except Exception as e:
        return None, str(e)

# =========================
#   INDICATORS (5m)
# =========================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line

def vwap(df: pd.DataFrame) -> pd.Series:
    # Typical price VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return cum_pv / cum_vol

# =========================
#   DATA SOURCES
# =========================
@dataclass
class EndpointStatus:
    ok: bool
    code: str
    detail: str = ""

def eodhd_intraday_bars(ticker: str, interval: str, lookback_minutes: int) -> Tuple[pd.DataFrame, EndpointStatus]:
    """
    EODHD intraday endpoint:
      https://eodhd.com/api/intraday/{ticker}.US?interval=5m&fmt=json&api_token=KEY&from=...&to=...
    """
    if not EODHD_API_KEY:
        return pd.DataFrame(), EndpointStatus(False, "missing_key", "EODHD_API_KEY missing")

    symbol = f"{ticker.upper()}.US"
    to_dt = dt.datetime.now(CST_TZ)
    from_dt = to_dt - dt.timedelta(minutes=int(lookback_minutes))

    url = f"https://eodhd.com/api/intraday/{symbol}"
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "interval": interval,
        "from": from_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "to": to_dt.strftime("%Y-%m-%d %H:%M:%S"),
    }

    r, err = safe_get(url, params=params, timeout=25)
    if err or r is None:
        return pd.DataFrame(), EndpointStatus(False, "network_error", err or "unknown network error")

    if r.status_code != 200:
        return pd.DataFrame(), EndpointStatus(False, f"http_{r.status_code}", r.text[:200])

    try:
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame(), EndpointStatus(False, "ticker_data_empty", "No intraday bars returned for this window.")
        df = pd.DataFrame(data)
        # EODHD typical columns: datetime, open, high, low, close, volume
        if "datetime" not in df.columns:
            # sometimes it's "date" depending on plan/endpoint behavior
            if "date" in df.columns:
                df = df.rename(columns={"date": "datetime"})
        # Normalize dtypes
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["datetime", "close"]).sort_values("datetime")
        return df, EndpointStatus(True, "ok", "")
    except Exception as e:
        return pd.DataFrame(), EndpointStatus(False, "parse_error", str(e))

def eodhd_news(tickers: List[str], lookback_minutes: int) -> Tuple[pd.DataFrame, EndpointStatus]:
    """
    EODHD news endpoint:
      https://eodhd.com/api/news?s=TSLA.US,SPY.US&offset=0&limit=50&api_token=KEY&fmt=json
    We'll filter by last N minutes in CST.
    """
    if not EODHD_API_KEY:
        return pd.DataFrame(), EndpointStatus(False, "missing_key", "EODHD_API_KEY missing")

    symbols = ",".join([f"{t.upper()}.US" for t in tickers])
    url = "https://eodhd.com/api/news"
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "s": symbols,
        "limit": 50,
        "offset": 0,
    }

    r, err = safe_get(url, params=params, timeout=25)
    if err or r is None:
        return pd.DataFrame(), EndpointStatus(False, "network_error", err or "unknown")

    if r.status_code != 200:
        return pd.DataFrame(), EndpointStatus(False, f"http_{r.status_code}", r.text[:200])

    try:
        data = r.json()
        if not isinstance(data, list):
            return pd.DataFrame(), EndpointStatus(False, "parse_error", "News payload not a list")

        df = pd.DataFrame(data)
        if df.empty:
            return df, EndpointStatus(True, "ok", "no headlines")

        # Columns vary by EODHD; normalize:
        # common: date, title, link, source, symbols
        if "date" in df.columns:
            df["published"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(CST_TZ)
        elif "published" in df.columns:
            df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True).dt.tz_convert(CST_TZ)
        else:
            df["published"] = pd.NaT

        # Try to infer ticker from symbols field
        if "symbols" in df.columns:
            df["ticker"] = df["symbols"].astype(str).str.split(",").str[0].str.replace(".US", "", regex=False)
        elif "symbol" in df.columns:
            df["ticker"] = df["symbol"].astype(str).str.replace(".US", "", regex=False)
        else:
            df["ticker"] = "N/A"

        if "link" in df.columns:
            df["url"] = df["link"]
        elif "url" not in df.columns:
            df["url"] = ""

        if "source" not in df.columns:
            df["source"] = ""

        if "title" not in df.columns:
            df["title"] = df.get("text", "").astype(str).str.slice(0, 120)

        cutoff = dt.datetime.now(CST_TZ) - dt.timedelta(minutes=int(lookback_minutes))
        df = df[df["published"].notna() & (df["published"] >= cutoff)]

        # final shape
        df = df.sort_values("published", ascending=False)
        return df, EndpointStatus(True, "ok", "no headlines" if df.empty else "")
    except Exception as e:
        return pd.DataFrame(), EndpointStatus(False, "parse_error", str(e))

def fred_10y_yield() -> Tuple[Optional[float], EndpointStatus]:
    """
    FRED 10Y: series DGS10 (daily)
    https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key=...&file_type=json&sort_order=desc&limit=1
    """
    if not FRED_API_KEY:
        return None, EndpointStatus(False, "missing_key", "FRED_API_KEY missing (optional)")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "DGS10",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }
    r, err = safe_get(url, params=params, timeout=20)
    if err or r is None:
        return None, EndpointStatus(False, "network_error", err or "unknown")

    if r.status_code != 200:
        return None, EndpointStatus(False, f"http_{r.status_code}", r.text[:200])

    try:
        js = r.json()
        obs = js.get("observations", [])
        if not obs:
            return None, EndpointStatus(False, "empty", "No observations")
        v = obs[0].get("value", None)
        if v in (None, ".", ""):
            return None, EndpointStatus(False, "empty", "No value")
        return float(v), EndpointStatus(True, "ok", "")
    except Exception as e:
        return None, EndpointStatus(False, "parse_error", str(e))

def uw_flow_alerts(limit: int = 200) -> Tuple[pd.DataFrame, EndpointStatus]:
    """
    UW flow alerts endpoint you listed:
      GET /option-trades/flow-alerts?limit=100
    Base: https://api.unusualwhales.com/api
    """
    if not UW_TOKEN:
        return pd.DataFrame(), EndpointStatus(False, "missing_key", "UW_TOKEN missing")

    url = UW_FLOW_ALERTS_URL
    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}",
    }
    params = {"limit": int(limit)}
    r, err = safe_get(url, headers=headers, params=params, timeout=25)
    if err or r is None:
        return pd.DataFrame(), EndpointStatus(False, "network_error", err or "unknown")

    if r.status_code != 200:
        return pd.DataFrame(), EndpointStatus(False, f"http_{r.status_code}", r.text[:200])

    try:
        js = r.json()
        rows = js.get("data", js if isinstance(js, dict) else [])
        if not rows:
            return pd.DataFrame(), EndpointStatus(True, "ok", "no alerts")
        df = pd.DataFrame(rows)

        # Normalize important fields (best-effort; UW can vary by plan)
        # created_at / executed_at
        if "created_at" in df.columns:
            df["ts"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_convert(CST_TZ)
        elif "executed_at" in df.columns:
            df["ts"] = pd.to_datetime(df["executed_at"], errors="coerce", utc=True).dt.tz_convert(CST_TZ)
        else:
            df["ts"] = pd.NaT

        # underlying
        for cand in ["underlying_symbol", "ticker", "symbol"]:
            if cand in df.columns:
                df["ticker_norm"] = df[cand].astype(str).str.upper()
                break
        if "ticker_norm" not in df.columns:
            # sometimes option_chain_id includes underlying prefix
            if "option_chain_id" in df.columns:
                df["ticker_norm"] = df["option_chain_id"].astype(str).str.extract(r"^([A-Z]+)")[0].fillna("N/A")
            else:
                df["ticker_norm"] = "N/A"

        # option type
        if "option_type" in df.columns:
            df["opt_type"] = df["option_type"].astype(str).str.lower()
        elif "type" in df.columns:
            df["opt_type"] = df["type"].astype(str).str.lower()
        else:
            df["opt_type"] = ""

        # numeric fields
        for c in ["premium", "volume", "open_interest", "implied_volatility", "gamma", "delta", "size", "strike"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # side tags (bullish/bearish)
        df["tags_str"] = df.get("tags", "").astype(str).str.lower()
        df["bearish_tag"] = df["tags_str"].str.contains("bearish")
        df["bullish_tag"] = df["tags_str"].str.contains("bullish")

        return df.sort_values("ts", ascending=False), EndpointStatus(True, "ok", "no alerts" if df.empty else "")
    except Exception as e:
        return pd.DataFrame(), EndpointStatus(False, "parse_error", str(e))

# =========================
#   SCORING LOGIC (CALLS / PUTS ONLY)
# =========================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def norm_iv_percent(iv_raw: Optional[float]) -> Optional[float]:
    """
    Normalize IV into percent.
    - If API gives 0.31 => 31%
    - If API gives 31 => 31%
    - If API gives something insane like 3719 => treat as bad and return None
    """
    if iv_raw is None or (isinstance(iv_raw, float) and (math.isnan(iv_raw) or math.isinf(iv_raw))):
        return None
    try:
        iv = float(iv_raw)
    except:
        return None

    # common cases:
    if iv <= 0:
        return None
    if iv < 5:          # decimal
        return iv * 100.0
    if iv <= 300:       # already percent
        return iv
    # insane
    return None

def compute_volume_ratio(df: pd.DataFrame, window: int = 20) -> Optional[float]:
    if df is None or df.empty or "volume" not in df.columns:
        return None
    vol = df["volume"].fillna(0)
    if len(vol) < 5:
        return None
    cur = float(vol.iloc[-1])
    base = float(vol.tail(window).mean()) if len(vol) >= window else float(vol.mean())
    if base <= 0:
        return None
    return cur / base

def ema_stack_state(close: pd.Series) -> Optional[str]:
    if close is None or close.empty:
        return None
    e9 = ema(close, 9).iloc[-1]
    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1] if len(close) >= 60 else np.nan
    c = close.iloc[-1]
    if not np.isnan(e50):
        if c > e9 > e20 > e50:
            return "Bullish"
        if c < e9 < e20 < e50:
            return "Bearish"
        return "Mixed"
    # fallback without e50
    if c > e9 > e20:
        return "Bullish"
    if c < e9 < e20:
        return "Bearish"
    return "Mixed"

def vwap_bias(df: pd.DataFrame) -> Optional[bool]:
    if df is None or df.empty:
        return None
    if not all(c in df.columns for c in ["high", "low", "close", "volume"]):
        return None
    vw = vwap(df)
    if vw.isna().all():
        return None
    return bool(df["close"].iloc[-1] > vw.iloc[-1])

def score_direction(
    rsi_val: Optional[float],
    macd_h: Optional[float],
    vwap_above: Optional[bool],
    ema_stack: Optional[str],
    vol_ratio: Optional[float],
    uw_bias: Optional[str],
    put_call_vol: Optional[str],
    iv_spike: Optional[bool],
    ten_y: Optional[float],
    ten_y_filter_on: bool,
    weights: Dict[str, float],
) -> Tuple[int, str, str]:
    """
    Returns (confidence 0..100, direction CALL/PUT, signal WAIT/BUY CALLS/BUY PUTS)
    Institutional mode will be enforced outside.
    """
    # Direction points: + for CALLS, - for PUTS
    points = 0.0
    total_w = 0.0

    def add(score: float, w_key: str):
        nonlocal points, total_w
        w = float(weights.get(w_key, 0.0))
        if w <= 0:
            return
        points += score * w
        total_w += w

    # RSI: >55 bullish, <45 bearish
    if rsi_val is not None:
        if rsi_val >= 55:
            add(+1.0, "rsi")
        elif rsi_val <= 45:
            add(-1.0, "rsi")
        else:
            add(0.0, "rsi")

    # MACD hist: positive bullish
    if macd_h is not None:
        add(+1.0 if macd_h > 0 else (-1.0 if macd_h < 0 else 0.0), "macd")

    # VWAP
    if vwap_above is not None:
        add(+1.0 if vwap_above else -1.0, "vwap")

    # EMA stack
    if ema_stack is not None:
        if ema_stack == "Bullish":
            add(+1.0, "ema")
        elif ema_stack == "Bearish":
            add(-1.0, "ema")
        else:
            add(0.0, "ema")

    # Volume ratio (spike helps direction only if trend agrees; otherwise neutral)
    if vol_ratio is not None:
        if vol_ratio >= 1.6:
            add(+0.35, "vol")  # mild
        elif vol_ratio <= 0.7:
            add(-0.15, "vol")
        else:
            add(0.0, "vol")

    # UW bias (from flow tags / call-put premium)
    if uw_bias in ("Bullish", "Bearish"):
        add(+1.0 if uw_bias == "Bullish" else -1.0, "uw")

    # Put/Call volume (from UW flow alerts filtered) – if we see aggressive put dominance, bearish
    if put_call_vol in ("Put-heavy", "Call-heavy"):
        add(-0.9 if put_call_vol == "Put-heavy" else +0.9, "uw")

    # IV spike: if IV spike + bearish flow => puts more likely; if IV spike + bullish flow => calls (but usually IV spike = risk)
    if iv_spike is not None and iv_spike:
        add(-0.2, "iv")  # conservative tilt (IV spike is usually “risk up”)
    # 10Y filter (optional): rising yields slightly bearish for growth / broad risk
    if ten_y_filter_on and ten_y is not None:
        # if 10Y above 4.5 => mild bearish; below 3.8 => mild bullish (tunable)
        if ten_y >= 4.5:
            add(-0.35, "teny")
        elif ten_y <= 3.8:
            add(+0.25, "teny")
        else:
            add(0.0, "teny")

    if total_w <= 0:
        # no inputs => WAIT
        return 50, "—", "WAIT"

    # Normalize points to confidence
    # points range roughly [-total_w, +total_w]
    norm = points / total_w  # -1..+1
    confidence = int(round(50 + 50 * norm))
    confidence = max(0, min(100, confidence))

    direction = "CALLS" if norm > 0.08 else ("PUTS" if norm < -0.08 else "—")
    signal = "BUY CALLS" if direction == "CALLS" else ("BUY PUTS" if direction == "PUTS" else "WAIT")
    return confidence, direction, signal

# =========================
#   UW FLOW FEATURES
# =========================
def uw_features_for_tickers(
    flow_df: pd.DataFrame,
    tickers: List[str],
    min_premium: float,
    max_dte: int,
    require_volume_gt_oi: bool,
    exclude_itm: bool,
) -> Dict[str, Dict[str, Any]]:
    """
    Best-effort filtering on UW flow alerts.
    NOTE: DTE/ITM fields may not exist depending on plan. We won't crash; we enforce what exists.
    """
    out: Dict[str, Dict[str, Any]] = {t: {
        "uw_status": "ok",
        "uw_unusual": "NO",
        "uw_bias": "N/A",
        "put_call_vol": "N/A",
        "iv_now": None,
        "iv_spike": None,
        "gamma_bias": "N/A",
        "contract_pressure": "N/A",
        "flow_rows": pd.DataFrame(),
    } for t in tickers}

    if flow_df is None or flow_df.empty:
        return out

    df = flow_df.copy()
    df = df[df["ticker_norm"].isin([t.upper() for t in tickers])]

    # Premium filter (if available)
    if "premium" in df.columns:
        df = df[df["premium"].fillna(0) >= float(min_premium)]

    # Volume > OI (if both exist)
    if require_volume_gt_oi and ("volume" in df.columns) and ("open_interest" in df.columns):
        df = df[(df["volume"].fillna(0) > df["open_interest"].fillna(0))]

    # Exclude ITM (only if fields exist; many plans won't expose enough; do best-effort)
    # We need underlying_price + strike + option_type
    if exclude_itm and all(c in df.columns for c in ["underlying_price", "strike", "opt_type"]):
        df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
        # calls ITM if underlying > strike, puts ITM if underlying < strike
        calls_itm = (df["opt_type"] == "call") & (df["underlying_price"] > df["strike"])
        puts_itm = (df["opt_type"] == "put") & (df["underlying_price"] < df["strike"])
        df = df[~(calls_itm | puts_itm)]

    # DTE max (if expiry exists)
    if max_dte is not None and "expiry" in df.columns:
        try:
            expiry = pd.to_datetime(df["expiry"], errors="coerce")
            now = dt.datetime.now(CST_TZ).date()
            df["dte"] = (expiry.dt.date - now).apply(lambda x: x.days if pd.notna(x) else np.nan)
            df = df[df["dte"].fillna(9999) <= int(max_dte)]
        except:
            pass

    # Compute per ticker features
    for t in tickers:
        sub = df[df["ticker_norm"] == t.upper()].copy()
        out[t]["flow_rows"] = sub

        if sub.empty:
            continue

        # Bias from tags/premium
        # If bullish/bearish premium available, use it; else infer from tags/opt_type
        bullish = 0.0
        bearish = 0.0

        if "bullish_premium" in sub.columns and "bearish_premium" in sub.columns:
            bullish = float(pd.to_numeric(sub["bullish_premium"], errors="coerce").fillna(0).sum())
            bearish = float(pd.to_numeric(sub["bearish_premium"], errors="coerce").fillna(0).sum())
        else:
            # Use tags if present
            bullish += float(sub.loc[sub["bullish_tag"] == True, "premium"].fillna(0).sum()) if "premium" in sub.columns else 0.0
            bearish += float(sub.loc[sub["bearish_tag"] == True, "premium"].fillna(0).sum()) if "premium" in sub.columns else 0.0

            # If no tags, assume calls lean bullish, puts lean bearish
            if bullish == 0.0 and bearish == 0.0 and "premium" in sub.columns and "opt_type" in sub.columns:
                bullish = float(sub.loc[sub["opt_type"] == "call", "premium"].fillna(0).sum())
                bearish = float(sub.loc[sub["opt_type"] == "put", "premium"].fillna(0).sum())

        if bullish > bearish * 1.15:
            out[t]["uw_bias"] = "Bullish"
        elif bearish > bullish * 1.15:
            out[t]["uw_bias"] = "Bearish"
        else:
            out[t]["uw_bias"] = "Neutral"

        # Put/Call volume dominance (use volume if exists, else count rows)
        if "volume" in sub.columns and "opt_type" in sub.columns:
            callv = float(sub.loc[sub["opt_type"] == "call", "volume"].fillna(0).sum())
            putv = float(sub.loc[sub["opt_type"] == "put", "volume"].fillna(0).sum())
        else:
            callv = float((sub["opt_type"] == "call").sum()) if "opt_type" in sub.columns else 0.0
            putv = float((sub["opt_type"] == "put").sum()) if "opt_type" in sub.columns else 0.0

        if callv + putv > 0:
            if putv > callv * 1.2:
                out[t]["put_call_vol"] = "Put-heavy"
            elif callv > putv * 1.2:
                out[t]["put_call_vol"] = "Call-heavy"
            else:
                out[t]["put_call_vol"] = "Balanced"

        # IV now (best effort from implied_volatility)
        ivp = None
        if "implied_volatility" in sub.columns:
            # take most recent non-null
            iv_raw = sub["implied_volatility"].dropna()
            if not iv_raw.empty:
                ivp = norm_iv_percent(float(iv_raw.iloc[0]))
        out[t]["iv_now"] = ivp

        # IV spike (compare last 20 IV values for that ticker in flow alerts)
        iv_spike = None
        if "implied_volatility" in sub.columns:
            ivs = sub["implied_volatility"].dropna().astype(float)
            if len(ivs) >= 8:
                # ivs are likely decimals; normalize to decimals for compare
                # if values look like >5, divide by 100
                ivs_dec = ivs.copy()
                if ivs_dec.median() > 5:
                    ivs_dec = ivs_dec / 100.0
                cur = float(ivs_dec.iloc[0])
                base = float(ivs_dec.iloc[:20].median()) if len(ivs_dec) >= 10 else float(ivs_dec.median())
                if base > 0:
                    iv_spike = (cur >= base * 1.25)
        out[t]["iv_spike"] = iv_spike

        # Gamma bias proxy (using gamma * size, calls positive, puts negative)
        gamma_bias = "N/A"
        if "gamma" in sub.columns and "opt_type" in sub.columns:
            g = sub["gamma"].fillna(0).astype(float)
            sz = sub["size"].fillna(1).astype(float) if "size" in sub.columns else 1.0
            sign = np.where(sub["opt_type"] == "call", 1.0, -1.0)
            gx = float((g * sz * sign).sum())
            if gx > 0:
                gamma_bias = "Positive"
            elif gx < 0:
                gamma_bias = "Negative"
            else:
                gamma_bias = "Neutral"
        out[t]["gamma_bias"] = gamma_bias

        # Contract pressure proxy (bid/ask vol if available; else premium)
        if "ask_vol" in sub.columns and "bid_vol" in sub.columns:
            av = float(pd.to_numeric(sub["ask_vol"], errors="coerce").fillna(0).sum())
            bv = float(pd.to_numeric(sub["bid_vol"], errors="coerce").fillna(0).sum())
            if av > bv * 1.2:
                out[t]["contract_pressure"] = "Ask-side"
            elif bv > av * 1.2:
                out[t]["contract_pressure"] = "Bid-side"
            else:
                out[t]["contract_pressure"] = "Mixed"
        elif "premium" in sub.columns:
            out[t]["contract_pressure"] = f"${float(sub['premium'].fillna(0).sum()):,.0f}"

        # unusual flag if premium huge (institutional)
        prem_sum = float(sub["premium"].fillna(0).sum()) if "premium" in sub.columns else 0.0
        out[t]["uw_unusual"] = "YES" if prem_sum >= (min_premium * 2.0) else "NO"

    return out

# =========================
#   UI — SIDEBAR SETTINGS
# =========================
st.title(APP_TITLE)
last_update = dt.datetime.now(CST_TZ).strftime("%Y-%m-%d %H:%M:%S CST")
st.caption(f"Last update (CST): **{last_update}**")

with st.sidebar:
    st.header("Settings")

    st.write("Type any tickers (comma-separated).")
    tickers_text = st.text_input("Tickers", value=",".join(DEFAULT_TICKERS), help="Example: SPY,TSLA,NVDA,META")

    quick = st.multiselect("Quick pick (optional)", options=DEFAULT_QUICK, default=[], help="Adds to your typed list.")

    def parse_tickers(txt: str, extra: List[str]) -> List[str]:
        base = []
        for part in (txt or "").split(","):
            t = part.strip().upper()
            if t:
                base.append(t)
        for t in extra:
            if t and t.upper() not in base:
                base.append(t.upper())
        # de-dupe preserve order
        seen = set()
        out = []
        for t in base:
            if t not in seen:
                out.append(t)
                seen.add(t)
        return out[:25]  # safety cap

    tickers = parse_tickers(tickers_text, quick)
    if not tickers:
        tickers = DEFAULT_TICKERS

    news_lookback = st.number_input("News lookback (minutes)", min_value=15, max_value=240, value=60, step=5)
    price_lookback = st.number_input("Price lookback (minutes)", min_value=60, max_value=1440, value=240, step=30)
    refresh_sec = st.slider("Auto-refresh (seconds)", min_value=10, max_value=120, value=30, step=5)
    st.divider()

    st.subheader("Institutional mode")
    inst_threshold = st.slider("Signals only if confidence ≥", min_value=50, max_value=95, value=75, step=1)

    st.divider()
    st.subheader("UW Flow filters")
    min_premium = st.number_input("Min premium ($)", min_value=100000, max_value=5000000, value=1000000, step=50000)
    max_dte = st.number_input("Max DTE (days)", min_value=1, max_value=30, value=3, step=1)
    require_vol_gt_oi = st.checkbox("Require Volume > OI (if available)", value=True)
    exclude_itm = st.checkbox("Exclude ITM (if fields available)", value=True)

    st.divider()
    st.subheader("Weights (sum doesn't have to be 1)")
    weights = {
        "rsi": st.slider("RSI weight", 0.0, 0.3, 0.15, 0.01),
        "macd": st.slider("MACD weight", 0.0, 0.3, 0.15, 0.01),
        "vwap": st.slider("VWAP weight", 0.0, 0.3, 0.15, 0.01),
        "ema": st.slider("EMA stack (9/20/50) weight", 0.0, 0.3, 0.18, 0.01),
        "vol": st.slider("Volume ratio weight", 0.0, 0.3, 0.12, 0.01),
        "uw": st.slider("UW flow weight", 0.0, 0.4, 0.20, 0.01),
        "iv": st.slider("IV spike weight", 0.0, 0.2, 0.05, 0.01),
        "teny": st.slider("10Y yield weight", 0.0, 0.2, 0.05, 0.01),
    }

    ten_y_on = st.checkbox("Enable 10Y yield filter (FRED)", value=True)

    st.divider()
    st.subheader("Keys status (green/red)")
    def key_badge(name: str, ok: bool):
        if ok:
            st.success(name)
        else:
            st.error(f"{name} (missing)")

    key_badge("UW_TOKEN", bool(UW_TOKEN))
    key_badge("EODHD_API_KEY", bool(EODHD_API_KEY))
    # Polygon removed on purpose
    st.info("Polygon removed (was causing http_403).")
    if ten_y_on:
        key_badge("FRED_API_KEY (10Y live)", bool(FRED_API_KEY))
    else:
        st.info("10Y filter disabled.")

# Auto-refresh
st.markdown(
    f"""
    <script>
      setTimeout(function() {{
        window.location.reload();
      }}, {int(refresh_sec)*1000});
    </script>
    """,
    unsafe_allow_html=True
)

# =========================
#   MAIN DATA PULL
# =========================
# EODHD intraday per ticker
bars_map: Dict[str, pd.DataFrame] = {}
bars_status: Dict[str, EndpointStatus] = {}
for t in tickers:
    df_bars, st_b = eodhd_intraday_bars(t, interval="5m", lookback_minutes=int(price_lookback))
    bars_map[t] = df_bars
    bars_status[t] = st_b

# EODHD news
news_df, news_status = eodhd_news(tickers, lookback_minutes=int(news_lookback))

# UW flow alerts
flow_df, uw_flow_status = uw_flow_alerts(limit=400)

# FRED 10Y
ten_y_val, ten_y_status = fred_10y_yield() if ten_y_on else (None, EndpointStatus(False, "disabled", ""))

# UW features
uw_feat = uw_features_for_tickers(
    flow_df=flow_df,
    tickers=tickers,
    min_premium=float(min_premium),
    max_dte=int(max_dte),
    require_volume_gt_oi=bool(require_vol_gt_oi),
    exclude_itm=bool(exclude_itm),
)

# =========================
#   BUILD SIGNAL TABLE
# =========================
rows = []
for t in tickers:
    df = bars_map.get(t, pd.DataFrame())

    # defaults
    rsi_v = macd_h = None
    vwap_above = None
    ema_state = None
    vol_ratio = None
    bars_n = 0
    last_bar = None
    bars_status_code = bars_status[t].code

    if df is not None and not df.empty:
        bars_n = int(len(df))
        last_bar = df["datetime"].iloc[-1].astimezone(CST_TZ).strftime("%Y-%m-%d %H:%M:%S") if hasattr(df["datetime"].iloc[-1], "astimezone") else str(df["datetime"].iloc[-1])
        close = df["close"].astype(float)
        rsi_v = float(rsi(close, 14).iloc[-1]) if len(close) >= 20 else None
        macd_h = float(macd_hist(close).iloc[-1]) if len(close) >= 35 else None
        vwap_above = vwap_bias(df)
        ema_state = ema_stack_state(close)
        vol_ratio = compute_volume_ratio(df, window=20)

    # UW
    uwb = uw_feat[t]["uw_bias"]
    pcv = uw_feat[t]["put_call_vol"]
    iv_now = uw_feat[t]["iv_now"]
    iv_spike = uw_feat[t]["iv_spike"]
    gamma_bias = uw_feat[t]["gamma_bias"]

    # Score
    conf, direction, signal = score_direction(
        rsi_val=rsi_v,
        macd_h=macd_h,
        vwap_above=vwap_above,
        ema_stack=ema_state,
        vol_ratio=vol_ratio,
        uw_bias=uwb,
        put_call_vol=pcv,
        iv_spike=iv_spike,
        ten_y=ten_y_val,
        ten_y_filter_on=ten_y_on,
        weights=weights,
    )

    institutional = "YES" if conf >= int(inst_threshold) and signal != "WAIT" else "NO"

    rows.append({
        "Ticker": t,
        "Confidence": conf,
        "Direction": direction,
        "Signal": signal,
        "Institutional": institutional,

        "RSI": None if rsi_v is None else round(rsi_v, 2),
        "MACD_hist": None if macd_h is None else round(macd_h, 4),
        "VWAP_above": None if vwap_above is None else ("Above" if vwap_above else "Below"),
        "EMA_stack": ema_state if ema_state else None,
        "Vol_ratio": None if vol_ratio is None else round(vol_ratio, 2),

        "UW_unusual": uw_feat[t]["uw_unusual"],
        "UW_bias": uwb,
        "Put/Call_vol": pcv,

        "IV_now_%": None if iv_now is None else round(iv_now, 2),
        "IV_spike": "YES" if iv_spike else ("NO" if iv_spike is False else None),
        "Gamma_bias": gamma_bias,

        "10Y": None if ten_y_val is None else round(float(ten_y_val), 2),

        "Bars": bars_n,
        "Last_bar(CST)": last_bar,
        "Bars_status": bars_status_code,
        "News_status": news_status.code,
        "UW_flow_status": uw_flow_status.code,
    })

signals_df = pd.DataFrame(rows)

# =========================
#   LAYOUT
# =========================
left, right = st.columns([1.35, 1.0], gap="large")

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("This is embedded. True filtering (DTE/ITM/premium rules) is best done inside the screener itself.")

    # Use your existing web embed (doesn't require API)
    # If you have a specific saved screener URL, replace it below.
    UW_WEB_URL = "https://unusualwhales.com/options-screener"
    st.components.v1.iframe(UW_WEB_URL, height=760, scrolling=True)

with right:
    st.subheader("Endpoints status")
    # show endpoint statuses as green/yellow/red
    def badge(name: str, status: EndpointStatus):
        if status.ok:
            st.success(f"{name} ✅ ({status.code})")
        else:
            # yellow for "expected/partial" cases
            if status.code in ("ticker_data_empty", "ok") or status.code.startswith("http_4"):
                st.warning(f"{name} ⚠️ ({status.code}) — {status.detail}")
            else:
                st.error(f"{name} ❌ ({status.code}) — {status.detail}")

    badge("EODHD intraday bars", EndpointStatus(bars_status[tickers[0]].ok, bars_status[tickers[0]].code, bars_status[tickers[0]].detail) if tickers else EndpointStatus(False,"no_tickers",""))
    badge("UW flow-alerts", uw_flow_status)
    badge("EODHD news", news_status)
    if ten_y_on:
        badge("FRED 10Y yield", ten_y_status)

    st.divider()
    st.subheader("Live Score / Signals (EODHD intraday + EODHD headlines + UW flow)")

    # show signals table (no crashes on missing columns)
    display_cols = [
        "Ticker","Confidence","Direction","Signal","Institutional",
        "RSI","MACD_hist","VWAP_above","EMA_stack","Vol_ratio",
        "UW_unusual","UW_bias","Put/Call_vol","IV_now_%","IV_spike","Gamma_bias","10Y",
        "Bars","Last_bar(CST)","Bars_status","News_status","UW_flow_status"
    ]
    st.dataframe(signals_df.reindex(columns=display_cols), use_container_width=True, height=260)

    st.divider()
    st.subheader(f"Institutional Alerts (≥ threshold only)")
    inst_df = signals_df[(signals_df["Institutional"] == "YES")].copy()
    if inst_df.empty:
        st.info("No institutional signals right now.")
    else:
        st.dataframe(inst_df.reindex(columns=display_cols), use_container_width=True, height=200)

    st.divider()
    st.subheader("Unusual Flow Alerts (UW API) — filtered")

    # Show filtered flow for selected tickers only
    # We keep it simple: show top rows by timestamp with key columns
    if flow_df is None or flow_df.empty:
        st.info("No UW alerts returned.")
    else:
        f2 = flow_df[flow_df["ticker_norm"].isin([t.upper() for t in tickers])].copy()
        # Apply same premium filter for display
        if "premium" in f2.columns:
            f2 = f2[f2["premium"].fillna(0) >= float(min_premium)]
        # Useful display columns
        flow_cols = []
        for c in ["ts","ticker_norm","opt_type","strike","expiry","premium","volume","open_interest","implied_volatility","gamma","tags"]:
            if c in f2.columns:
                flow_cols.append(c)
        if not flow_cols:
            st.info("UW returned alerts, but expected fields are not present on this plan.")
        else:
            f2 = f2.sort_values("ts", ascending=False).head(50)
            # Format IV as %
            if "implied_volatility" in f2.columns:
                f2["iv_%"] = f2["implied_volatility"].apply(norm_iv_percent)
                if "iv_%" not in flow_cols:
                    flow_cols.append("iv_%")
            st.dataframe(f2.reindex(columns=flow_cols), use_container_width=True, height=240)

    st.divider()
    st.subheader(f"News — last {int(news_lookback)} minutes (EODHD)")
    # Fix the KeyError: always reindex columns even if missing
    if news_df is None or news_df.empty:
        st.info("No news in this lookback window (or EODHD returned none).")
    else:
        news_out = news_df.copy()
        # Build a safe schema
        news_out["published_cst"] = news_out.get("published", pd.NaT)
        if pd.api.types.is_datetime64_any_dtype(news_out["published_cst"]):
            news_out["published_cst"] = news_out["published_cst"].dt.strftime("%Y-%m-%d %H:%M:%S")
        show = news_out.reindex(columns=["ticker", "published_cst", "source", "title", "url"])
        st.dataframe(show.head(40), use_container_width=True, height=260)
        st.caption("Tip: Click URL column links (or copy/paste).")

    with st.expander("What None/N/A means (plain English)"):
        st.write(
            "- **Bars_status = ticker_data_empty** → EODHD didn’t return 5-minute bars for that ticker/time window (common after-hours).\n"
            "- **RSI/MACD/VWAP/EMA/Vol_ratio = N/A** → We don’t have enough intraday bars to calculate them.\n"
            "- **UW_flow_status = ok** → Flow-alerts endpoint is working.\n"
            "- **News_status = ok** but empty → No headlines inside your lookback window."
        )

# =========================
#   EXTRA: Helpful note if UW endpoints differ
# =========================
if uw_flow_status.ok is False and uw_flow_status.code.startswith("http_4"):
    st.warning(
        "UW returned an HTTP error. This is usually either:\n"
        "1) the endpoint path changed, or\n"
        "2) your plan doesn’t include flow-alerts.\n\n"
        "You can still use the embedded UW screener even if API access is limited."
    )
