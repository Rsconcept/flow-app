import time
import math
import requests
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================
# App setup
# =========================
st.set_page_config(page_title="Institutional Options Signals (5m) — CALLS / PUTS ONLY", layout="wide")

CST = ZoneInfo("America/Chicago")
NOW_CST = dt.datetime.now(CST)

UW_BASE = "https://api.unusualwhales.com/api"
EOD_BASE = "https://eodhd.com/api"
FRED_BASE = "https://api.stlouisfed.org/fred"

UW_TOKEN = st.secrets.get("UW_TOKEN", "").strip()
EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", "").strip()
FRED_API_KEY = st.secrets.get("FRED_API_KEY", "").strip()

# Optional override
UW_FLOW_ALERTS_URL = st.secrets.get("UW_FLOW_ALERTS_URL", "").strip()


# =========================
# UI helpers
# =========================
def ok_box(label: str):
    st.markdown(
        f"""
        <div style="padding:10px 12px;border-radius:10px;background:#123a2b;border:1px solid #1f6b4d;color:#bff5da;font-weight:700">
        ✅ {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

def bad_box(label: str, detail: str = ""):
    st.markdown(
        f"""
        <div style="padding:10px 12px;border-radius:10px;background:#3b1212;border:1px solid #7a2b2b;color:#ffd3d3;font-weight:700">
        ❌ {label}<div style="font-weight:500;margin-top:6px;color:#ffb3b3">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def warn_box(label: str, detail: str = ""):
    st.markdown(
        f"""
        <div style="padding:10px 12px;border-radius:10px;background:#3b3512;border:1px solid #7a6b2b;color:#fff2b3;font-weight:700">
        ⚠️ {label}<div style="font-weight:500;margin-top:6px;color:#ffeaa0">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Helpers
# =========================
def http_get(url, headers=None, params=None, timeout=20):
    return requests.get(url, headers=headers, params=params, timeout=timeout)

def to_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x).replace(",", ""))
    except Exception:
        return default

def safe_upper_ticker(s: str) -> str:
    return "".join([c for c in s.upper().strip() if c.isalnum() or c in [":", ".", "-", "_"]])


# =========================
# Technical indicators
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, math.nan)
    return 100 - (100 / (1 + rs))

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line

def vwap(df: pd.DataFrame) -> pd.Series:
    typical = df["close"]
    pv = (typical * df["volume"]).cumsum()
    vv = df["volume"].cumsum().replace(0, math.nan)
    return pv / vv


# =========================
# Data pulls (EODHD replaces Polygon)
# =========================
def eodhd_intraday_bars(ticker: str, interval: str = "1m"):
    """
    EODHD intraday endpoint:
    GET /intraday/{ticker}.US?interval=1m&fmt=json&api_token=KEY
    """
    if not EODHD_API_KEY:
        return None, "missing_key"

    url = f"{EOD_BASE}/intraday/{ticker}.US"
    params = {"interval": interval, "fmt": "json", "api_token": EODHD_API_KEY}
    r = http_get(url, params=params, timeout=25)
    if r.status_code != 200:
        return None, f"http_{r.status_code}"

    js = r.json()
    if not isinstance(js, list) or len(js) == 0:
        return pd.DataFrame(), "empty"

    df = pd.DataFrame(js)
    # Typical columns: datetime, open, high, low, close, volume
    if "datetime" not in df.columns:
        return pd.DataFrame(), "bad_json"

    # EODHD datetime is usually "YYYY-MM-DD HH:MM:SS"
    df["dt_utc"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    # If it parsed as naive, force as UTC
    if df["dt_utc"].isna().all():
        df["dt_utc"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["dt_utc"] = df["dt_utc"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")

    df["dt_cst"] = df["dt_utc"].dt.tz_convert(CST)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("dt_cst")
    return df, "ok"

def eodhd_news(ticker: str, lookback_minutes: int):
    if not EODHD_API_KEY:
        return pd.DataFrame(), "missing_key"

    to_dt = dt.datetime.now(dt.timezone.utc)
    from_dt = to_dt - dt.timedelta(minutes=lookback_minutes)

    url = f"{EOD_BASE}/news"
    params = {
        "s": ticker,
        "from": from_dt.strftime("%Y-%m-%d"),
        "to": to_dt.strftime("%Y-%m-%d"),
        "limit": 50,
        "offset": 0,
        "api_token": EODHD_API_KEY,
        "fmt": "json",
    }
    r = http_get(url, params=params, timeout=20)
    if r.status_code != 200:
        return pd.DataFrame(), f"http_{r.status_code}"

    js = r.json()
    if not isinstance(js, list):
        return pd.DataFrame(), "bad_json"

    df = pd.DataFrame(js)
    if df.empty:
        return df, "none"

    if "date" in df.columns:
        df["published_utc"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["published_cst"] = df["published_utc"].dt.tz_convert(CST)
    else:
        df["published_cst"] = pd.NaT

    df = df.rename(columns={"s": "ticker", "link": "url"})
    keep = [c for c in ["ticker", "published_cst", "source", "title", "url"] if c in df.columns]
    df = df[keep]

    if df["published_cst"].notna().any():
        cutoff = dt.datetime.now(CST) - dt.timedelta(minutes=lookback_minutes)
        df = df[df["published_cst"] >= cutoff]

    return df.head(25), "ok"

def eodhd_iv_now(ticker: str):
    if not EODHD_API_KEY:
        return None, "missing_key"

    url = f"{EOD_BASE}/options/{ticker}.US"
    params = {"api_token": EODHD_API_KEY, "fmt": "json"}
    r = http_get(url, params=params, timeout=25)
    if r.status_code != 200:
        return None, f"http_{r.status_code}"

    js = r.json()
    data = js.get("data", js) if isinstance(js, dict) else js
    if not data:
        return None, "empty"

    rows = []
    if isinstance(data, dict):
        for exp, contracts in data.items():
            if isinstance(contracts, list):
                for c in contracts:
                    c2 = dict(c)
                    c2["expiration"] = exp
                    rows.append(c2)
    elif isinstance(data, list):
        rows = data

    if not rows:
        return None, "no_rows"

    df = pd.DataFrame(rows)
    if "impliedVolatility" not in df.columns:
        return None, "no_iv_field"

    df["iv"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    df = df[df["iv"].notna() & (df["iv"] > 0)]
    if df.empty:
        return None, "iv_empty"

    if "expiration" in df.columns:
        df["exp_dt"] = pd.to_datetime(df["expiration"], errors="coerce")
        df = df[df["exp_dt"].notna()]
        if not df.empty:
            closest = df["exp_dt"].min()
            df = df[df["exp_dt"] == closest]

    return float(df["iv"].median() * 100.0), "ok"

def fred_10y():
    if not FRED_API_KEY:
        return None, "missing_key"
    url = f"{FRED_BASE}/series/observations"
    params = {
        "series_id": "DGS10",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,
    }
    r = http_get(url, params=params, timeout=20)
    if r.status_code != 200:
        return None, f"http_{r.status_code}"
    js = r.json()
    obs = js.get("observations", [])
    for o in obs:
        v = o.get("value")
        if v and v != ".":
            return float(v), "ok"
    return None, "empty"


# =========================
# Unusual Whales pulls
# =========================
def uw_headers():
    return {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}",
    }

def uw_flow_alerts(limit=100):
    if not UW_TOKEN:
        return pd.DataFrame(), "missing_key"

    if UW_FLOW_ALERTS_URL:
        r = http_get(UW_FLOW_ALERTS_URL, headers=uw_headers(), timeout=25)
    else:
        url = f"{UW_BASE}/option-trades/flow-alerts"
        r = http_get(url, headers=uw_headers(), params={"limit": limit}, timeout=25)

    if r.status_code != 200:
        return pd.DataFrame(), f"http_{r.status_code}"

    js = r.json()
    data = js.get("data", js)
    if not data:
        return pd.DataFrame(), "empty"

    return pd.DataFrame(data), "ok"

def uw_ticker_options_flow(ticker: str):
    """
    Correct endpoint:
    GET /stock/{ticker}/options-flow
    """
    if not UW_TOKEN:
        return None, "missing_key"
    url = f"{UW_BASE}/stock/{ticker}/options-flow"
    r = http_get(url, headers=uw_headers(), timeout=25)
    if r.status_code != 200:
        return None, f"http_{r.status_code}"
    js = r.json()
    data = js.get("data", [])
    if not data:
        return None, "empty"
    row = data[0] if isinstance(data, list) else data
    return row, "ok"


# =========================
# Sidebar
# =========================
st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")
st.caption(f"Last update (CST): {NOW_CST.strftime('%Y-%m-%d %H:%M:%S')} CST")

with st.sidebar:
    st.header("Settings")

    tickers_raw = st.text_input(
        "Type any tickers (comma-separated). Example: SPY,TSLA,NVDA",
        value="SPY,TSLA",
    )
    tickers = [safe_upper_ticker(x) for x in tickers_raw.split(",") if safe_upper_ticker(x)]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        tickers = ["SPY"]

    news_lookback = st.slider("News lookback (minutes)", 5, 240, 60, 5)
    price_lookback = st.slider("Price lookback (minutes)", 30, 600, 240, 30)

    st.divider()
    threshold = st.slider("Institutional mode: signals only if confidence ≥", 50, 95, 75, 1)

    st.divider()
    st.subheader("Weights")
    w_rsi = st.slider("RSI weight", 0.0, 0.4, 0.15, 0.01)
    w_macd = st.slider("MACD weight", 0.0, 0.4, 0.15, 0.01)
    w_vwap = st.slider("VWAP weight", 0.0, 0.4, 0.15, 0.01)
    w_ema = st.slider("EMA stack (9/20/50) weight", 0.0, 0.4, 0.18, 0.01)
    w_volr = st.slider("Volume ratio weight", 0.0, 0.4, 0.12, 0.01)
    w_uw = st.slider("UW flow weight", 0.0, 0.6, 0.20, 0.01)
    w_10y = st.slider("10Y yield (optional) weight", 0.0, 0.2, 0.05, 0.01)

    st.divider()
    refresh_s = st.slider("Auto-refresh (seconds)", 10, 120, 30, 5)

    st.divider()
    st.subheader("Keys status (green/red)")
    ok_box("UW_TOKEN") if UW_TOKEN else bad_box("UW_TOKEN", "Missing in Secrets")
    ok_box("EODHD_API_KEY") if EODHD_API_KEY else bad_box("EODHD_API_KEY", "Missing in Secrets")
    ok_box("FRED_API_KEY (10Y live)") if FRED_API_KEY else warn_box("FRED_API_KEY (10Y live)", "Optional")


# =========================
# Main layout
# =========================
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("This is embedded. True filtering is best done inside the screener itself.")
    components.iframe("https://unusualwhales.com/options-screener", height=720, scrolling=True)

# Shared pulls
teny, teny_status = fred_10y()
flow_df, flow_status = uw_flow_alerts(limit=100)

# =========================
# Build signals
# =========================
rows = []

for tkr in tickers:
    bars_df, bars_status = eodhd_intraday_bars(tkr, interval="1m")

    news_df, news_status = eodhd_news(tkr, news_lookback)
    iv_now, iv_status = eodhd_iv_now(tkr)
    uw_flow_row, uw_tickerflow_status = uw_ticker_options_flow(tkr)

    rsi_val = macd_h = vwap_above = vol_ratio = None
    ema_stack = None
    bars_count = 0
    last_bar_cst = None

    if isinstance(bars_df, pd.DataFrame) and not bars_df.empty:
        cutoff = dt.datetime.now(CST) - dt.timedelta(minutes=price_lookback)
        bars_df2 = bars_df[bars_df["dt_cst"] >= cutoff].copy()
        if not bars_df2.empty:
            bars_count = len(bars_df2)
            last_bar_cst = bars_df2["dt_cst"].iloc[-1]

            close = bars_df2["close"].astype(float)
            vol = bars_df2["volume"].astype(float).fillna(0)

            rsi_series = rsi(close, 14)
            macd_series = macd_hist(close)
            vwap_series = vwap(pd.DataFrame({"close": close, "volume": vol}))

            rsi_val = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else None
            macd_h = float(macd_series.iloc[-1]) if pd.notna(macd_series.iloc[-1]) else None

            vwap_last = float(vwap_series.iloc[-1]) if pd.notna(vwap_series.iloc[-1]) else None
            vwap_above = (float(close.iloc[-1]) > vwap_last) if (vwap_last is not None) else None

            e9 = ema(close, 9).iloc[-1]
            e20 = ema(close, 20).iloc[-1]
            e50 = ema(close, 50).iloc[-1]
            if pd.notna(e9) and pd.notna(e20) and pd.notna(e50):
                if e9 > e20 > e50:
                    ema_stack = "bullish"
                elif e9 < e20 < e50:
                    ema_stack = "bearish"
                else:
                    ema_stack = "neutral"

            avg_vol = float(vol.mean()) if len(vol) > 0 else None
            vol_ratio = (float(vol.iloc[-1]) / avg_vol) if (avg_vol and avg_vol > 0) else None

    # UW unusual tag / bias from flow alerts table
    uw_unusual = "NO"
    uw_bias = "N/A"
    put_call = "N/A"

    if isinstance(flow_df, pd.DataFrame) and not flow_df.empty:
        if "underlying_symbol" in flow_df.columns:
            sub = flow_df[flow_df["underlying_symbol"] == tkr].copy()
        elif "ticker" in flow_df.columns:
            sub = flow_df[flow_df["ticker"] == tkr].copy()
        else:
            sub = flow_df.head(0)

        if not sub.empty:
            uw_unusual = "YES"
            if "tags" in sub.columns:
                tags = sub["tags"].dropna().astype(str).str.lower().tolist()
                if any("bull" in x for x in tags):
                    uw_bias = "Bullish"
                elif any("bear" in x for x in tags):
                    uw_bias = "Bearish"
                else:
                    uw_bias = "Mixed"
            if "option_type" in sub.columns:
                calls = (sub["option_type"].astype(str).str.lower() == "call").sum()
                puts = (sub["option_type"].astype(str).str.lower() == "put").sum()
                if calls + puts > 0:
                    put_call = f"{puts}/{calls}"

    # IV spike detection (simple session history)
    iv_spike = "N/A"
    if "iv_hist" not in st.session_state:
        st.session_state["iv_hist"] = {}
    hist = st.session_state["iv_hist"].get(tkr, [])
    if iv_now is not None:
        hist = (hist + [iv_now])[-30:]
        st.session_state["iv_hist"][tkr] = hist
        if len(hist) >= 10:
            mean = sum(hist[:-1]) / max(1, len(hist[:-1]))
            sd = (sum((x - mean) ** 2 for x in hist[:-1]) / max(1, len(hist[:-1]) - 1)) ** 0.5
            iv_spike = "YES" if (sd > 0 and iv_now > mean + 2 * sd) else "NO"
        else:
            iv_spike = "NO"

    # Gamma bias proxy from UW options-flow
    gamma_bias = "N/A"
    if isinstance(uw_flow_row, dict):
        call_prem = to_float(uw_flow_row.get("call_premium"))
        put_prem = to_float(uw_flow_row.get("put_premium"))
        if call_prem is not None and put_prem is not None:
            gamma_bias = "Positive" if call_prem >= put_prem else "Negative"

    tenybias = "N/A"
    if teny is not None:
        tenybias = "Bearish" if teny >= 4.5 else "Bullish"

    # Confidence scoring
    parts = []
    rsi_score = 0.0
    if rsi_val is not None:
        if rsi_val <= 35:
            rsi_score = +1.0
        elif rsi_val >= 65:
            rsi_score = -1.0
    parts.append((rsi_score, w_rsi))

    macd_score = 0.0
    if macd_h is not None:
        macd_score = +1.0 if macd_h > 0 else (-1.0 if macd_h < 0 else 0.0)
    parts.append((macd_score, w_macd))

    vwap_score = 0.0
    if vwap_above is True:
        vwap_score = +1.0
    elif vwap_above is False:
        vwap_score = -1.0
    parts.append((vwap_score, w_vwap))

    ema_score = 0.0
    if ema_stack == "bullish":
        ema_score = +1.0
    elif ema_stack == "bearish":
        ema_score = -1.0
    parts.append((ema_score, w_ema))

    volr_score = 0.0
    if vol_ratio is not None:
        if vol_ratio >= 1.8:
            volr_score = +0.5
        elif vol_ratio <= 0.6:
            volr_score = -0.2
    parts.append((volr_score, w_volr))

    uw_score = 0.0
    if uw_unusual == "YES":
        if uw_bias == "Bullish":
            uw_score = +1.0
        elif uw_bias == "Bearish":
            uw_score = -1.0
        else:
            uw_score = +0.2
    parts.append((uw_score, w_uw))

    y_score = 0.0
    if teny is not None:
        y_score = -1.0 if tenybias == "Bearish" else +0.4
    parts.append((y_score, w_10y))

    total_w = sum(w for _, w in parts)
    raw = sum(val * w for val, w in parts) / total_w if total_w > 0 else 0.0
    confidence = int(round((raw + 1) * 50))
    confidence = max(0, min(100, confidence))

    if confidence >= 55:
        direction = "BULLISH"
        signal = "BUY CALLS"
    elif confidence <= 45:
        direction = "BEARISH"
        signal = "BUY PUTS"
    else:
        direction = "NEUTRAL"
        signal = "WAIT"

    inst_ok = confidence >= threshold

    rows.append({
        "Ticker": tkr,
        "Confidence": confidence,
        "Direction": direction,
        "Signal": signal,
        "Institutional": "YES" if inst_ok else "NO",
        "RSI": None if rsi_val is None else round(rsi_val, 1),
        "MACD_hist": None if macd_h is None else round(macd_h, 4),
        "VWAP_above": vwap_above,
        "EMA_stack": ema_stack,
        "Vol_ratio": None if vol_ratio is None else round(vol_ratio, 2),
        "UW_unusual": uw_unusual,
        "UW_bias": uw_bias,
        "Put/Call": put_call,
        "IV_now": None if iv_now is None else round(iv_now, 2),
        "IV_spike": iv_spike,
        "Gamma_bias": gamma_bias,
        "10Y": None if teny is None else round(teny, 2),
        "Bars": bars_count,
        "Last_bar(CST)": None if last_bar_cst is None else last_bar_cst.strftime("%Y-%m-%d %H:%M:%S"),
        "Bars_status": bars_status,
        "News_status": news_status,
        "UW_flow_status": flow_status,
        "UW_tickerflow_status": uw_tickerflow_status,
    })

score_df = pd.DataFrame(rows)

with right:
    st.subheader("Endpoints status")

    # EODHD intraday
    statuses = set(score_df["Bars_status"].tolist())
    if "ok" in statuses:
        ok_box("EODHD intraday bars (ok)")
    else:
        common = score_df["Bars_status"].value_counts().index[0]
        warn_box("EODHD intraday bars", f"({common}) — indicators will be N/A if no bars")

    if flow_status == "ok":
        ok_box("UW flow-alerts (ok)")
    else:
        warn_box("UW flow-alerts", f"({flow_status})")

    tf_statuses = set(score_df["UW_tickerflow_status"].tolist())
    if "ok" in tf_statuses:
        ok_box("UW ticker options-flow (ok)")
    else:
        common = score_df["UW_tickerflow_status"].value_counts().index[0]
        warn_box("UW ticker options-flow", f"({common})")

    ns = set(score_df["News_status"].tolist())
    ok_box("EODHD news (ok)") if ("ok" in ns or "none" in ns) else warn_box("EODHD news", str(ns))

    if FRED_API_KEY and teny_status == "ok":
        ok_box("FRED 10Y yield (ok)")
    elif FRED_API_KEY:
        warn_box("FRED 10Y yield", f"({teny_status})")
    else:
        warn_box("FRED 10Y yield", "optional")

    st.divider()
    st.subheader("Live Score / Signals (EODHD intraday + EODHD headlines + UW flow)")
    st.dataframe(score_df, use_container_width=True, height=260)

    st.subheader(f"Institutional Alerts (≥ {threshold} only)")
    inst = score_df[score_df["Institutional"] == "YES"].copy()
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        st.dataframe(inst[["Ticker","Confidence","Direction","Signal","UW_bias","IV_spike","Gamma_bias","10Y"]], use_container_width=True)

    st.subheader("Unusual Flow Alerts (UW API) — filtered")
    if flow_status != "ok" or flow_df.empty:
        st.warning(f"UW flow alerts not available ({flow_status}).")
    else:
        show_cols = [c for c in ["underlying_symbol","option_chain_id","option_type","strike","expiry","premium","volume","open_interest","executed_at","tags"] if c in flow_df.columns]
        df2 = flow_df.copy()
        if "executed_at" in df2.columns:
            df2["executed_at"] = pd.to_datetime(df2["executed_at"], errors="coerce", utc=True).dt.tz_convert(CST)
        if "underlying_symbol" in df2.columns:
            df2 = df2[df2["underlying_symbol"].isin(tickers)]
        st.dataframe(df2[show_cols].head(50), use_container_width=True, height=240)

    st.subheader(f"News — last {news_lookback} minutes (EODHD)")
    all_news = []
    for tkr in tickers:
        df, status = eodhd_news(tkr, news_lookback)
        if status == "ok" and not df.empty:
            all_news.append(df)
    if not all_news:
        st.info("No news in this lookback window (or provider returned none).")
    else:
        news_all = pd.concat(all_news, ignore_index=True)
        st.dataframe(news_all[["ticker","published_cst","source","title","url"]].head(40), use_container_width=True, height=220)

# Auto-refresh
st.caption(f"Auto-refresh is set to {refresh_s}s.")
time.sleep(refresh_s)
st.rerun()
