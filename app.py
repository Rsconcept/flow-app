import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Institutional Options Signals", layout="wide")

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "AMD", "NVDA"]
INSTITUTIONAL_THRESHOLD = 75

EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")).strip()
UW_BEARER = st.secrets.get("UW_BEARER", os.getenv("UW_BEARER", "")).strip()

UTC = timezone.utc


# =========================================================
# UTILS
# =========================================================
def utc_now():
    return datetime.now(UTC)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# =========================================================
# INDICATORS (pandas / numpy)
# =========================================================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / (dn.ewm(alpha=1/n, adjust=False).mean() + 1e-9)
    return 100 - (100 / (1 + rs))


def macd_hist(close: pd.Series, fast=12, slow=26, sig=9) -> pd.Series:
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    return line - signal


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    # Typical price * volume / cumulative volume
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    return (pv.cumsum() / (df["volume"].cumsum() + 1e-9))


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    return mid, upper, lower


def stoch(df: pd.DataFrame, n: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    low_min = df["low"].rolling(n).min()
    high_max = df["high"].rolling(n).max()
    k = 100 * (df["close"] - low_min) / ((high_max - low_min) + 1e-9)
    k = k.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    Returns: (supertrend_line, direction)
    direction: +1 bullish, -1 bearish
    """
    _atr = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upperband = hl2 + multiplier * _atr
    lowerband = hl2 - multiplier * _atr

    st_line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i == 0:
            st_line.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = 1
            continue

        prev_st = st_line.iloc[i-1]
        prev_dir = direction.iloc[i-1]

        # finalize bands
        ub = upperband.iloc[i]
        lb = lowerband.iloc[i]
        prev_close = df["close"].iloc[i-1]

        if ub < upperband.iloc[i-1] or prev_close > upperband.iloc[i-1]:
            ub_final = ub
        else:
            ub_final = upperband.iloc[i-1]

        if lb > lowerband.iloc[i-1] or prev_close < lowerband.iloc[i-1]:
            lb_final = lb
        else:
            lb_final = lowerband.iloc[i-1]

        close_i = df["close"].iloc[i]

        if prev_dir == 1:
            if close_i < lb_final:
                direction.iloc[i] = -1
                st_line.iloc[i] = ub_final
            else:
                direction.iloc[i] = 1
                st_line.iloc[i] = lb_final
        else:
            if close_i > ub_final:
                direction.iloc[i] = 1
                st_line.iloc[i] = lb_final
            else:
                direction.iloc[i] = -1
                st_line.iloc[i] = ub_final

    return st_line, direction


# =========================================================
# DATA: EODHD intraday (1m or 5m)
# =========================================================
@st.cache_data(ttl=20)
def eodhd_intraday(symbol: str, minutes_back: int, api_key: str) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()

    end = utc_now()
    start = end - timedelta(minutes=minutes_back)

    def fetch(interval: str):
        url = f"https://eodhd.com/api/intraday/{symbol}.US"
        params = {
            "api_token": api_key,
            "fmt": "json",
            "interval": interval,
            "from": start.strftime("%Y-%m-%d %H:%M:%S"),
            "to": end.strftime("%Y-%m-%d %H:%M:%S"),
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data)

    try:
        df = fetch("1m")
    except Exception:
        try:
            df = fetch("5m")
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close", "high", "low"])
    if "volume" not in df.columns:
        df["volume"] = 0

    return df


def to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    if df_1m.empty:
        return df_1m
    # If already ~5m, resample still safe.
    o = df_1m["open"].resample("5min").first()
    h = df_1m["high"].resample("5min").max()
    l = df_1m["low"].resample("5min").min()
    c = df_1m["close"].resample("5min").last()
    v = df_1m["volume"].resample("5min").sum()
    out = pd.concat([o, h, l, c, v], axis=1).dropna()
    out.columns = ["open", "high", "low", "close", "volume"]
    return out


# =========================================================
# DATA: Unusual Whales options-volume (THIS is your endpoint)
# =========================================================
@st.cache_data(ttl=30)
def uw_options_volume(ticker: str, bearer: str) -> dict:
    if not bearer:
        return {}
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/options-volume"
    headers = {"Accept": "application/json, text/plain", "Authorization": f"Bearer {bearer}"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    js = r.json()
    data = js.get("data", [])
    if not data:
        return {}
    # Take most recent record
    return data[0]


def uw_flow_bias(vol_js: dict) -> dict:
    """
    Returns a stable, institutional "flow confirmation":
    - dominant side (CALL/PUT)
    - unusual trigger (True/False)
    """
    if not vol_js:
        return {"unusual": False, "dominant": None, "strength": 0.0, "call_premium": 0, "put_premium": 0}

    call_prem = safe_float(vol_js.get("call_premium"), 0.0)
    put_prem = safe_float(vol_js.get("put_premium"), 0.0)
    bullish_prem = safe_float(vol_js.get("bullish_premium"), 0.0)
    bearish_prem = safe_float(vol_js.get("bearish_premium"), 0.0)

    # Use premium dominance
    total = call_prem + put_prem
    if total <= 0:
        return {"unusual": False, "dominant": None, "strength": 0.0, "call_premium": call_prem, "put_premium": put_prem}

    dom = "CALL" if call_prem > put_prem else "PUT"
    imbalance = abs(call_prem - put_prem) / total  # 0..1

    # Institutional trigger:
    # - imbalance >= 0.35 and premium >= 50k (tweakable)
    unusual = (imbalance >= 0.35) and (total >= 50000)

    # Strength 0..1
    strength = clamp(imbalance, 0.0, 1.0)

    # If bullish/bearish premium present, reinforce direction
    if bullish_prem and bearish_prem and (bullish_prem + bearish_prem) > 0:
        prem_dom = "CALL" if bullish_prem > bearish_prem else "PUT"
        if prem_dom == dom:
            strength = clamp(strength + 0.15, 0.0, 1.0)

    return {
        "unusual": unusual,
        "dominant": dom,
        "strength": strength,
        "call_premium": call_prem,
        "put_premium": put_prem,
    }


# =========================================================
# SCORING: CALLScore + PUTScore (0..100)
# =========================================================
def compute_call_put_scores(df5: pd.DataFrame, flow: dict, weights: dict) -> dict:
    """
    Uses: VWAP, EMA stack, RSI, MACD, Bollinger, Volume spike, ATR, Supertrend, Stoch, + UW flow bias
    Returns call_score, put_score and debug fields.
    """
    if df5.empty or len(df5) < 60:
        return {"call": 0.0, "put": 0.0, "signal": "NO TRADE", "conf": 0.0, "debug": {"reason": "not enough bars"}}

    close = df5["close"].astype(float)
    volume = df5["volume"].astype(float)

    # core indicators
    vwap_line = vwap(df5)
    e9 = ema(close, 9)
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    r = rsi(close, 14)
    mh = macd_hist(close)
    bb_mid, bb_up, bb_lo = bollinger(close, 20, 2.0)
    a = atr(df5, 14)
    st_line, st_dir = supertrend(df5, 10, 3.0)
    k, d = stoch(df5, 14, 3, 3)

    # last values
    c = float(close.iloc[-1])
    vwap_last = float(vwap_line.iloc[-1])
    r_last = float(r.iloc[-1])
    mh_last = float(mh.iloc[-1])
    e9_last, e20_last, e50_last = float(e9.iloc[-1]), float(e20.iloc[-1]), float(e50.iloc[-1])
    bb_u, bb_l = float(bb_up.iloc[-1]), float(bb_lo.iloc[-1])
    st_bull = (int(st_dir.iloc[-1]) == 1)
    k_last = float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else 50.0
    d_last = float(d.iloc[-1]) if not np.isnan(d.iloc[-1]) else 50.0

    # volume spike
    vol_avg = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1] / (vol_avg + 1e-9))

    # --- Convert each feature into call/put points 0..1
    def as01(x):  # clamp 0..1
        return float(clamp(x, 0.0, 1.0))

    # 1) VWAP bias
    vwap_call = as01(0.5 + (c - vwap_last) / (abs(vwap_last) * 0.003 + 1e-9))  # ~0.3% scale
    vwap_put  = 1.0 - vwap_call

    # 2) EMA stack trend
    if e9_last > e20_last > e50_last:
        trend_call, trend_put = 1.0, 0.0
    elif e9_last < e20_last < e50_last:
        trend_call, trend_put = 0.0, 1.0
    else:
        trend_call, trend_put = 0.5, 0.5

    # 3) RSI institutional bands
    rsi_call = as01((r_last - 45) / 25)   # 45->0, 70->1
    rsi_put  = as01((55 - r_last) / 25)   # 55->0, 30->1

    # 4) MACD hist
    macd_call = as01(0.5 + mh_last * 20)  # scale
    macd_put  = as01(0.5 - mh_last * 20)

    # 5) Bollinger position (trend vs extreme)
    # If price is riding upper band => calls, riding lower => puts, extreme outside => reduce confidence slightly later
    if c > bb_u:
        bb_call, bb_put = 0.65, 0.35
    elif c < bb_l:
        bb_call, bb_put = 0.35, 0.65
    else:
        # inside bands: slight bias to side based on middle
        bb_call = as01(0.5 + (c - bb_mid.iloc[-1]) / (abs(bb_mid.iloc[-1]) * 0.004 + 1e-9))
        bb_put  = 1.0 - bb_call

    # 6) Volume (activity, doesn’t decide direction alone)
    act = as01((vol_ratio - 1.0) / 2.0)  # 1x->0, 3x->1
    vol_call = 0.5 + 0.2 * act
    vol_put  = 0.5 + 0.2 * act

    # 7) ATR risk filter (too wild = reduce confidence)
    atr_pct = float(a.iloc[-1] / (c + 1e-9))
    # if ATR% > 0.8% on 5m, it's wild; penalize later
    atr_penalty = as01((atr_pct - 0.004) / 0.006)  # start penalty around 0.4%, max near 1.0%

    # 8) Supertrend
    st_call = 1.0 if st_bull else 0.0
    st_put  = 0.0 if st_bull else 1.0

    # 9) Stoch timing (small)
    # Bullish if K crosses above D under 20; bearish if crosses below D above 80
    k_prev = float(k.iloc[-2]) if len(k) >= 2 and not np.isnan(k.iloc[-2]) else k_last
    d_prev = float(d.iloc[-2]) if len(d) >= 2 and not np.isnan(d.iloc[-2]) else d_last
    cross_up = (k_prev <= d_prev) and (k_last > d_last) and (k_last < 25)
    cross_dn = (k_prev >= d_prev) and (k_last < d_last) and (k_last > 75)
    stoch_call = 1.0 if cross_up else 0.5
    stoch_put  = 1.0 if cross_dn else 0.5

    # 10) UW flow confirmation (your new endpoint)
    flow_dom = flow.get("dominant")
    flow_strength = float(flow.get("strength", 0.0))
    if flow.get("unusual") and flow_dom == "CALL":
        uw_call, uw_put = 0.5 + 0.5 * flow_strength, 0.5 - 0.5 * flow_strength
    elif flow.get("unusual") and flow_dom == "PUT":
        uw_call, uw_put = 0.5 - 0.5 * flow_strength, 0.5 + 0.5 * flow_strength
    else:
        uw_call, uw_put = 0.5, 0.5

    # Weighted blend -> 0..100
    W = weights
    w_sum = sum(W.values()) + 1e-9

    call01 = (
        vwap_call * W["vwap"] +
        trend_call * W["ema"] +
        rsi_call * W["rsi"] +
        macd_call * W["macd"] +
        bb_call * W["bb"] +
        vol_call * W["vol"] +
        st_call * W["supertrend"] +
        stoch_call * W["stoch"] +
        uw_call * W["uw"]
    ) / w_sum

    put01 = (
        vwap_put * W["vwap"] +
        trend_put * W["ema"] +
        rsi_put * W["rsi"] +
        macd_put * W["macd"] +
        bb_put * W["bb"] +
        vol_put * W["vol"] +
        st_put * W["supertrend"] +
        stoch_put * W["stoch"] +
        uw_put * W["uw"]
    ) / w_sum

    # apply ATR penalty to confidence (institutional risk control)
    call_score = clamp(call01 * 100 * (1.0 - 0.25 * atr_penalty), 0, 100)
    put_score  = clamp(put01 * 100 * (1.0 - 0.25 * atr_penalty), 0, 100)

    # Decision (institutional)
    if call_score >= INSTITUTIONAL_THRESHOLD and call_score > put_score:
        signal = "BUY CALLS"
        conf = call_score
    elif put_score >= INSTITUTIONAL_THRESHOLD and put_score > call_score:
        signal = "BUY PUTS"
        conf = put_score
    else:
        signal = "NO TRADE"
        conf = max(call_score, put_score)

    debug = {
        "VWAP": round(vwap_last, 4),
        "Close": round(c, 4),
        "RSI": round(r_last, 2),
        "MACD_hist": round(mh_last, 5),
        "VolRatio": round(vol_ratio, 2),
        "ATR%": round(atr_pct * 100, 2),
        "ST_dir": "UP" if st_bull else "DOWN",
        "UW_dom": flow_dom or "-",
        "UW_unusual": flow.get("unusual", False),
        "UW_callPrem": round(flow.get("call_premium", 0.0), 0),
        "UW_putPrem": round(flow.get("put_premium", 0.0), 0),
    }

    return {"call": call_score, "put": put_score, "signal": signal, "conf": conf, "debug": debug}


# =========================================================
# UI
# =========================================================
st.title("🏦 Institutional Options Signals (5m) — CALLS / PUTS ONLY")

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY", "QQQ", "DIA", "IWM"])
    lookback_min = st.number_input("Price lookback (minutes)", 120, 1200, 420, 30)
    refresh_sec = st.slider("Auto-refresh (seconds)", 15, 300, 60, 15)

    st.divider()
    st.subheader("Keys status")
    st.write("EODHD:", "✅" if EODHD_API_KEY else "❌")
    st.write("UW Bearer:", "✅" if UW_BEARER else "❌")

    st.divider()
    st.subheader("Weights (institutional defaults)")
    # these are stable weights; UW is confirmation, not overruling
    weights = {
        "vwap": st.slider("VWAP", 0.0, 0.30, 0.15, 0.01),
        "ema": st.slider("EMA stack (9/20/50)", 0.0, 0.30, 0.18, 0.01),
        "rsi": st.slider("RSI", 0.0, 0.20, 0.10, 0.01),
        "macd": st.slider("MACD", 0.0, 0.20, 0.10, 0.01),
        "bb": st.slider("Bollinger Bands", 0.0, 0.20, 0.08, 0.01),
        "vol": st.slider("Volume activity", 0.0, 0.20, 0.07, 0.01),
        "supertrend": st.slider("Supertrend", 0.0, 0.30, 0.14, 0.01),
        "stoch": st.slider("Stochastic timing", 0.0, 0.15, 0.03, 0.01),
        "uw": st.slider("UW options-volume confirmation", 0.0, 0.35, 0.15, 0.01),
    }

st_autorefresh(interval=int(refresh_sec * 1000), key="refresh")

if not tickers:
    st.info("Pick at least one ticker.")
    st.stop()

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Unusual Whales Screener (web view)")
    st.components.v1.iframe(UW_SCREENER_URL, height=850, scrolling=True)

with col2:
    st.subheader(f"Signals (Institutional: only ≥ {INSTITUTIONAL_THRESHOLD} confidence)")
    st.write(f"Updated (UTC): **{utc_now().strftime('%Y-%m-%d %H:%M:%S')}**")

    rows = []
    alerts = []

    for t in tickers:
        df1 = eodhd_intraday(t, int(lookback_min), EODHD_API_KEY)
        df5 = to_5m(df1)

        uw_js = {}
        flow = {"unusual": False, "dominant": None, "strength": 0.0, "call_premium": 0, "put_premium": 0}
        if UW_BEARER:
            try:
                uw_js = uw_options_volume(t, UW_BEARER)
                flow = uw_flow_bias(uw_js)
            except Exception:
                pass

        out = compute_call_put_scores(df5, flow, weights)

        rows.append({
            "Ticker": t,
            "Signal": out["signal"],
            "Confidence": round(out["conf"], 1),
            "CallScore": round(out["call"], 1),
            "PutScore": round(out["put"], 1),
            "UW Unusual": "YES" if flow.get("unusual") else "NO",
            "UW Dom": flow.get("dominant") or "-",
            "UW CallPrem": round(flow.get("call_premium", 0.0), 0),
            "UW PutPrem": round(flow.get("put_premium", 0.0), 0),
        })

        if out["signal"] != "NO TRADE":
            alerts.append(f"{t}: {out['signal']} | Confidence {round(out['conf'],1)} | UW={('YES' if flow.get('unusual') else 'NO')} Dom={flow.get('dominant') or '-'}")

    df_out = pd.DataFrame(rows).sort_values(["Confidence"], ascending=False)
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("High-Conviction Alerts")
    if alerts:
        for a in alerts[:20]:
            st.success(a)
    else:
        st.info("No institutional signals right now (needs ≥ 75 confidence).")

    st.divider()
    st.subheader("Debug (why it decided)")
    pick = st.selectbox("Inspect ticker", tickers, index=0)
    df1 = eodhd_intraday(pick, int(lookback_min), EODHD_API_KEY)
    df5 = to_5m(df1)
    try:
        uw_js = uw_options_volume(pick, UW_BEARER) if UW_BEARER else {}
        flow = uw_flow_bias(uw_js)
    except Exception:
        flow = {"unusual": False, "dominant": None, "strength": 0.0, "call_premium": 0, "put_premium": 0}

    dbg = compute_call_put_scores(df5, flow, weights)["debug"]
    st.json(dbg)

