import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

st.set_page_config(page_title="Institutional Options Signals (5m) — Calls/Puts Only", layout="wide")

CST = ZoneInfo("America/Chicago")

DEFAULT_QUICK = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "NVDA", "AMD"]

UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF"
    "&limit=250&max_dte=3"
    "&min_premium=1000000"
    "&min_volume_oi_ratio=1"
    "&order=premium&order_direction=desc"
)

EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")).strip()
UW_TOKEN      = st.secrets.get("UW_TOKEN", os.getenv("UW_TOKEN", "")).strip()

UW_FLOW_ALERTS_URL = st.secrets.get(
    "UW_FLOW_ALERTS_URL",
    os.getenv("UW_FLOW_ALERTS_URL", "https://api.unusualwhales.com/api/option-trade/flow-alerts")
).strip().replace("flow_alerts", "flow-alerts")


# ---------------- helpers ----------------
def now_cst():
    return datetime.now(tz=CST)

def fmt_cst(dt):
    return dt.astimezone(CST).strftime("%Y-%m-%d %H:%M:%S CST")

def safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default

def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))

def http_get_json(url, params=None, headers=None, timeout=20):
    r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------- indicators ----------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def macd_hist(series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line

def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    return pv.cumsum() / df["volume"].cumsum().replace(0, pd.NA)

def calc_volume_ratio(df: pd.DataFrame, lookback: int = 20):
    if df.empty or "volume" not in df.columns:
        return None
    v = df["volume"]
    if len(v) < 5:
        return None
    avg = v.rolling(lookback).mean().iloc[-1]
    cur = v.iloc[-1]
    if pd.isna(avg) or avg == 0:
        return None
    return float(cur / avg)


# ---------------- EODHD ----------------
@st.cache_data(ttl=30)
def eodhd_intraday(symbol_us: str, interval="5m", lookback_minutes=240):
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"

    url = "https://eodhd.com/api/intraday/" + symbol_us
    params = {"api_token": EODHD_API_KEY, "fmt": "json", "interval": interval}

    try:
        data = http_get_json(url, params=params)
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame(), "ticker_data_empty"

        df = pd.DataFrame(data)
        if "datetime" not in df.columns:
            return pd.DataFrame(), "bad_schema"

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(CST)
        df = df.dropna(subset=["datetime"]).sort_values("datetime")

        cutoff = now_cst() - timedelta(minutes=int(lookback_minutes))
        df = df[df["datetime"] >= cutoff].copy()

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"]).copy()

        if df.empty:
            return pd.DataFrame(), "ticker_data_empty"

        # Need enough bars to compute indicators reliably
        if len(df) < 30:
            return df, "insufficient_bars"

        return df, "ok"

    except requests.HTTPError as e:
        return pd.DataFrame(), f"http_{e.response.status_code}"
    except Exception:
        return pd.DataFrame(), "error"

@st.cache_data(ttl=60)
def eodhd_news(symbol_us: str, lookback_minutes=60, limit=30):
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"

    url = "https://eodhd.com/api/news"
    params = {"api_token": EODHD_API_KEY, "s": symbol_us, "limit": int(limit), "offset": 0, "fmt": "json"}

    try:
        data = http_get_json(url, params=params)
        if not isinstance(data, list):
            return pd.DataFrame(), "bad_schema"

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(), "no_headlines"

        if "date" in df.columns:
            df["published"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(CST)
        else:
            df["published"] = pd.NaT

        cutoff = now_cst() - timedelta(minutes=int(lookback_minutes))
        df = df[df["published"].isna() | (df["published"] >= cutoff)].copy()

        df["Title"] = df.get("title", "")
        df["Source"] = df.get("source", "")
        df["URL"] = df.get("link", df.get("url", ""))

        df = df[["published", "Source", "Title", "URL"]].sort_values("published", ascending=False)
        return df, "ok"

    except requests.HTTPError as e:
        return pd.DataFrame(), f"http_{e.response.status_code}"
    except Exception:
        return pd.DataFrame(), "error"

def naive_news_sentiment(title: str) -> float:
    if not isinstance(title, str):
        return 0.0
    t = title.lower()
    pos = ["beats", "beat", "surge", "rally", "up", "growth", "record", "strong", "bull", "upgrade", "raises"]
    neg = ["miss", "misses", "drop", "down", "lawsuit", "fraud", "weak", "bear", "downgrade", "cuts", "warn", "loss"]
    score = 0
    for w in pos:
        if w in t:
            score += 1
    for w in neg:
        if w in t:
            score -= 1
    return max(-1.0, min(1.0, score / 5.0))


# ---------------- UW endpoints ----------------
@st.cache_data(ttl=60)
def uw_options_volume_bias(ticker: str):
    if not UW_TOKEN:
        return None, "missing_token"

    url = f"https://api.unusualwhales.com/api/stock/{ticker}/options-volume"
    headers = {"Accept": "application/json, text/plain", "Authorization": f"Bearer {UW_TOKEN}"}

    try:
        data = http_get_json(url, headers=headers)
        rows = data.get("data", [])
        if not rows:
            return None, "no_data"
        row = rows[0]

        bullish = safe_float(row.get("bullish_premium"), 0.0)
        bearish = safe_float(row.get("bearish_premium"), 0.0)
        prem_total = max(bullish + bearish, 1.0)
        prem_bias = (bullish - bearish) / prem_total  # -1..+1

        call_vol = safe_float(row.get("call_volume"), None)
        put_vol  = safe_float(row.get("put_volume"), None)
        pc = None
        if call_vol and call_vol > 0 and put_vol is not None:
            pc = float(put_vol / call_vol)

        return {"prem_bias": float(prem_bias), "put_call_vol": pc}, "ok"
    except requests.HTTPError as e:
        return None, f"http_{e.response.status_code}"
    except Exception:
        return None, "error"

@st.cache_data(ttl=30)
def uw_flow_alerts(limit: int = 200):
    if not UW_TOKEN:
        return [], "missing_token"

    headers = {"Accept": "application/json, text/plain", "Authorization": f"Bearer {UW_TOKEN}"}
    params = {"limit": int(limit)}

    try:
        data = http_get_json(UW_FLOW_ALERTS_URL, params=params, headers=headers)
        rows = data.get("data", [])
        if not isinstance(rows, list):
            return [], "bad_schema"
        return rows, "ok"
    except requests.HTTPError as e:
        return [], f"http_{e.response.status_code}"
    except Exception:
        return [], "error"


# ---------------- scoring ----------------
def score_ticker(df_5m: pd.DataFrame, *, uw_vol=None, news_df=None, weights=None):
    # If we have NO bars or too few bars -> explain it clearly.
    if df_5m is None or df_5m.empty:
        return {"confidence": 50, "direction": "—", "signal": "WAIT", "reason": "No intraday bars returned for this window."}

    if len(df_5m) < 30:
        last_dt = df_5m["datetime"].iloc[-1] if "datetime" in df_5m.columns else None
        last_dt_s = fmt_cst(last_dt) if last_dt is not None else "unknown"
        return {
            "confidence": 50, "direction": "—", "signal": "WAIT",
            "reason": f"Not enough 5m bars to compute indicators (bars={len(df_5m)}; last={last_dt_s})."
        }

    close = df_5m["close"]
    df = df_5m.copy()

    df["EMA9"] = ema(close, 9)
    df["EMA20"] = ema(close, 20)
    df["EMA50"] = ema(close, 50)
    df["RSI"] = rsi(close, 14)
    df["MACD_H"] = macd_hist(close)
    df["VWAP"] = vwap(df)

    last = df.iloc[-1]
    rsi_v = float(last["RSI"])
    macd_h = float(last["MACD_H"])
    vwap_above = bool(last["close"] > last["VWAP"])
    ema_bull = bool(last["EMA9"] > last["EMA20"] > last["EMA50"])
    ema_bear = bool(last["EMA9"] < last["EMA20"] < last["EMA50"])
    vol_ratio = calc_volume_ratio(df, 20)

    # News sentiment
    ns = 0.0
    if news_df is not None and not news_df.empty:
        scores = [naive_news_sentiment(t) for t in news_df.head(10)["Title"].tolist()]
        if scores:
            ns = float(pd.Series(scores).mean())

    # UW options volume bias
    uw_bias = 0.0
    put_call_vol = None
    if uw_vol and isinstance(uw_vol, dict):
        uw_bias = float(uw_vol.get("prem_bias", 0.0))
        put_call_vol = uw_vol.get("put_call_vol", None)

    # Components -1..+1
    rsi_comp = max(-1.0, min(1.0, (rsi_v - 50.0) / 25.0))
    mh = df["MACD_H"].dropna()
    mh_std = float(mh.tail(100).std()) if len(mh) > 20 else 0.0
    macd_comp = 0.0 if mh_std == 0 else max(-1.0, min(1.0, macd_h / (2.0 * mh_std)))

    vwap_comp = 1.0 if vwap_above else -1.0
    ema_comp = 1.0 if ema_bull else (-1.0 if ema_bear else 0.0)

    vol_comp = 0.0
    if vol_ratio is not None:
        vol_comp = max(-1.0, min(1.0, (vol_ratio - 1.0) / 1.5))

    uw_comp = max(-1.0, min(1.0, uw_bias))
    news_comp = max(-1.0, min(1.0, ns))

    w = weights or {"vwap":0.20,"ema":0.20,"rsi":0.15,"macd":0.15,"vol":0.10,"uw":0.15,"news":0.05}

    raw = (
        w["vwap"]*vwap_comp +
        w["ema"]*ema_comp +
        w["rsi"]*rsi_comp +
        w["macd"]*macd_comp +
        w["vol"]*vol_comp +
        w["uw"]*uw_comp +
        w["news"]*news_comp
    )

    max_abs = sum(abs(x) for x in w.values())
    norm = 0.0 if max_abs == 0 else max(-1.0, min(1.0, raw / max_abs))

    confidence = int(round(50 + 50 * abs(norm)))
    direction = "BULLISH" if norm > 0.05 else ("BEARISH" if norm < -0.05 else "—")

    # Calls/Puts ONLY
    if direction == "BULLISH" and confidence >= 60:
        signal = "BUY CALLS"
    elif direction == "BEARISH" and confidence >= 60:
        signal = "BUY PUTS"
    else:
        signal = "WAIT"

    return {
        "confidence": confidence,
        "direction": direction,
        "signal": signal,
        "RSI": round(rsi_v, 1),
        "MACD_hist": round(macd_h, 4),
        "VWAP": "Above" if vwap_above else "Below",
        "EMA_stack": "Bull" if ema_bull else ("Bear" if ema_bear else "Neutral"),
        "Vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else "N/A",
        "UW_bias": round(uw_comp, 2),
        "Put/Call vol": round(put_call_vol, 2) if isinstance(put_call_vol, (int, float)) else "N/A",
        "reason": "ok"
    }


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")

    st.caption("Type any tickers (comma-separated). Example: SPY,TSLA,NVDA")
    raw = st.text_input("Tickers", value="TSLA")
    quick = st.multiselect("Quick pick (optional)", DEFAULT_QUICK, default=[])

    typed = [t.strip().upper() for t in raw.split(",") if t.strip()]
    tickers = []
    for t in typed + [q.upper() for q in quick]:
        if t and t not in tickers:
            tickers.append(t)

    st.divider()

    news_lookback = st.number_input("News lookback (minutes)", min_value=1, max_value=240, value=60, step=1)
    price_lookback = st.number_input("Price lookback (minutes)", min_value=60, max_value=1440, value=240, step=30)

    st.divider()
    refresh_seconds = st.slider("Auto-refresh (seconds)", min_value=10, max_value=300, value=30, step=5)

    st.divider()
    inst_min = st.slider("Institutional mode: signals only if confidence ≥", min_value=50, max_value=95, value=75, step=1)

    st.divider()
    st.subheader("Keys status (green/red)")

    def pill(label, ok: bool, detail=""):
        if ok:
            st.success(f"{label} ✅")
        else:
            st.error(f"{label} ❌ {detail}".strip())

    pill("EODHD_API_KEY", bool(EODHD_API_KEY), "(missing)")
    pill("UW_TOKEN (Bearer)", bool(UW_TOKEN), "(missing)")
    pill("UW_FLOW_ALERTS_URL", bool(UW_FLOW_ALERTS_URL), "(missing)")


# ---------------- autorefresh ----------------
if HAS_AUTOREFRESH:
    st_autorefresh(interval=int(refresh_seconds * 1000), key="autorefresh")


# ---------------- main UI ----------------
st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")
st.write(f"Last update (CST): **{fmt_cst(now_cst())}**")

if not tickers:
    st.warning("Type at least one ticker in the sidebar (example: TSLA).")
    st.stop()

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("This is embedded. True filtering (DTE/ITM/premium rules) is best done inside the screener itself.")
    st.components.v1.iframe(UW_SCREENER_URL, height=850, scrolling=True)

with right:
    st.subheader("Endpoints status")
    flow_rows, flow_status = uw_flow_alerts(limit=200)

    c1, c2 = st.columns(2)
    with c1:
        st.success("EODHD ✅" if EODHD_API_KEY else "EODHD ❌")
    with c2:
        if flow_status == "ok":
            st.success("UW flow-alerts ✅")
        else:
            st.warning(f"UW flow-alerts ⚠️ ({flow_status})")

    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW options-volume)")

    rows = []
    news_frames = []

    for t in tickers:
        symbol_us = f"{t}.US"

        bars, bars_status = eodhd_intraday(symbol_us, interval="5m", lookback_minutes=int(price_lookback))
        bars_count = int(len(bars)) if isinstance(bars, pd.DataFrame) else 0
        last_bar = fmt_cst(bars["datetime"].iloc[-1]) if bars_count and "datetime" in bars.columns else "N/A"

        ndf, news_status = eodhd_news(symbol_us, lookback_minutes=int(news_lookback), limit=50)
        if news_status == "ok" and not ndf.empty:
            ndf2 = ndf.copy()
            ndf2.insert(0, "Ticker", t)
            news_frames.append(ndf2)

        uwv, uwv_status = uw_options_volume_bias(t)

        res = score_ticker(
            bars,
            uw_vol=uwv if uwv_status == "ok" else None,
            news_df=ndf if news_status == "ok" else None,
        )

        def na(x):
            return "N/A" if (x is None or (isinstance(x, float) and pd.isna(x))) else x

        rows.append({
            "Ticker": t,
            "Confidence": res.get("confidence", 50),
            "Direction": res.get("direction", "—"),
            "Signal": res.get("signal", "WAIT"),
            "RSI": na(res.get("RSI")),
            "MACD_hist": na(res.get("MACD_hist")),
            "VWAP": na(res.get("VWAP")),
            "EMA_stack": na(res.get("EMA_stack")),
            "Vol_ratio": na(res.get("Vol_ratio")),
            "UW_bias": na(res.get("UW_bias")),
            "Put/Call vol": na(res.get("Put/Call vol")),
            "Bars": bars_count,
            "Last bar (CST)": last_bar,
            "EODHD bars status": bars_status,
            "EODHD news status": news_status,
            "UW opt-vol status": uwv_status,
            "Reason": res.get("reason", "ok"),
        })

    df_out = pd.DataFrame(rows)
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    st.markdown("### Institutional Alerts (≥ threshold only)")
    inst = df_out[(df_out["Confidence"] >= int(inst_min)) & (df_out["Signal"].isin(["BUY CALLS", "BUY PUTS"]))]
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        for _, r in inst.sort_values("Confidence", ascending=False).iterrows():
            st.success(f"{r['Ticker']}: {r['Signal']} | {r['Direction']} | Confidence={r['Confidence']}")

    st.markdown(f"### News — last {int(news_lookback)} minutes (EODHD)")
    if news_frames:
        news_all = pd.concat(news_frames, ignore_index=True)
        news_all["published_cst"] = news_all["published"].dt.strftime("%Y-%m-%d %H:%M:%S CST")
        show = news_all[["Ticker", "published_cst", "Source", "Title", "URL"]].copy()
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("No news in this lookback window (or EODHD returned none).")

    with st.expander("What 'None/N/A' means (plain English)"):
        st.write("If RSI/MACD/VWAP/EMA/Vol_ratio show N/A, it means EODHD did not return enough 5-minute bars in your lookback window.")
        st.write("Common reasons: after-hours, holiday/weekend, too small lookback, or ticker not returning bars from EODHD.")
