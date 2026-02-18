import os
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from streamlit_autorefresh import st_autorefresh


# =========================
#   PAGE CONFIG
# =========================
st.set_page_config(page_title="Institutional Options Signals (5m) — CALLS / PUTS ONLY", layout="wide")

CST = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")


# =========================
#   SECRETS / KEYS
# =========================
def _secret(name: str) -> str:
    return str(st.secrets.get(name, os.getenv(name, "")) or "").strip()

EODHD_API_KEY      = _secret("EODHD_API_KEY")
UW_TOKEN           = _secret("UW_TOKEN")
UW_FLOW_ALERTS_URL = _secret("UW_FLOW_ALERTS_URL")  # should be https://api.unusualwhales.com/api/option-trade/flow-alerts
FINVIZ_AUTH        = _secret("FINVIZ_AUTH")         # optional (not required in this build)
POLYGON_API_KEY    = _secret("POLYGON_API_KEY")     # optional (not required in this build)

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "AMD", "NVDA"]


# =========================
#   SAFE REQUEST
# =========================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "streamlit-app/1.0"})

def safe_get(url, *, params=None, headers=None, timeout=20):
    r = SESSION.get(url, params=params, headers=headers, timeout=timeout)
    return r


def now_cst():
    return datetime.now(tz=CST)

def fmt_cst(dt: datetime) -> str:
    return dt.astimezone(CST).strftime("%Y-%m-%d %H:%M:%S %Z")


# =========================
#   INDICATORS
# =========================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line

def vwap(df: pd.DataFrame) -> pd.Series:
    # VWAP using typical price * volume cumulative
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.cumsum()
    cum_v = df["volume"].cumsum().replace(0, 1e-12)
    return cum_pv / cum_v


# =========================
#   EODHD: intraday + news
# =========================
def ensure_us_symbol(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    # If user already typed ".US" or something, keep it.
    if "." in t:
        return t
    return f"{t}.US"

@st.cache_data(ttl=25, show_spinner=False)
def eodhd_intraday(symbol_us: str, interval: str = "5m", lookback_minutes: int = 240):
    """
    Returns DataFrame with columns: datetime, open, high, low, close, volume (CST-localized).
    """
    if not EODHD_API_KEY:
        return None, "missing_key"

    # EODHD intraday endpoint
    url = f"https://eodhd.com/api/intraday/{symbol_us}"
    # We use 'from' to reduce payload. Buffer a bit.
    start_utc = datetime.now(tz=UTC) - timedelta(minutes=lookback_minutes + 30)
    params = {
        "api_token": EODHD_API_KEY,
        "interval": interval,
        "fmt": "json",
        "from": start_utc.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        r = safe_get(url, params=params, timeout=25)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return None, "empty"

        df = pd.DataFrame(data)
        # EODHD returns "datetime" in UTC-like string
        if "datetime" not in df.columns:
            return None, "bad_schema"

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
        df["datetime"] = df["datetime"].dt.tz_convert(CST)

        # Ensure numeric
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "volume"])

        # Trim to lookback
        cutoff = now_cst() - timedelta(minutes=lookback_minutes)
        df = df[df["datetime"] >= cutoff]
        if len(df) < 30:
            # Not enough bars for indicators
            return df, "few_bars"

        return df, "ok"
    except Exception:
        return None, "exception"


@st.cache_data(ttl=45, show_spinner=False)
def eodhd_news(symbol_us: str, lookback_minutes: int = 60):
    """
    EODHD news endpoint: https://eodhd.com/api/news?s=AAPL.US&from=YYYY-MM-DD&api_token=...
    """
    if not EODHD_API_KEY:
        return None, "missing_key"

    from_dt = (now_cst() - timedelta(minutes=lookback_minutes)).date().isoformat()
    url = "https://eodhd.com/api/news"
    params = {
        "api_token": EODHD_API_KEY,
        "s": symbol_us,
        "from": from_dt,
        "limit": 50,
        "fmt": "json",
    }

    try:
        r = safe_get(url, params=params, timeout=25)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        data = r.json()
        if not isinstance(data, list):
            return None, "bad_schema"
        if len(data) == 0:
            return [], "empty"

        df = pd.DataFrame(data)
        # Normalize time
        if "date" in df.columns:
            df["published"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(CST)
        elif "published_at" in df.columns:
            df["published"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.tz_convert(CST)
        else:
            df["published"] = pd.NaT

        df = df.sort_values("published", ascending=False)

        # Filter to lookback minutes
        cutoff = now_cst() - timedelta(minutes=lookback_minutes)
        df = df[df["published"].notna() & (df["published"] >= cutoff)]

        out = []
        for _, row in df.iterrows():
            title = str(row.get("title", "") or "")
            src = str(row.get("source", "") or "")
            url_ = str(row.get("link", "") or row.get("url", "") or "")
            published = row.get("published")
            out.append({
                "Ticker": symbol_us.replace(".US", ""),
                "Published (CST)": published.strftime("%Y-%m-%d %H:%M:%S") if isinstance(published, pd.Timestamp) else "",
                "Source": src,
                "Title": title,
                "URL": url_
            })

        return out, "ok"
    except Exception:
        return None, "exception"


# =========================
#   UNUSUAL WHALES: options-volume + flow-alerts
# =========================
@st.cache_data(ttl=25, show_spinner=False)
def uw_options_volume_bias(ticker: str):
    """
    Correct endpoint:
    https://api.unusualwhales.com/api/stock/{ticker}/options-volume
    Returns bias info or None.
    """
    if not UW_TOKEN:
        return None, "missing_key"

    url = f"https://api.unusualwhales.com/api/stock/{ticker}/options-volume"
    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}",
    }
    try:
        r = safe_get(url, headers=headers, timeout=25)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        data = r.json()
        rows = data.get("data", [])
        if not rows:
            return None, "empty"

        row = rows[0]
        # Put/call premium + volumes
        call_vol = float(row.get("call_volume") or 0)
        put_vol = float(row.get("put_volume") or 0)
        call_prem = float(row.get("call_premium") or 0)
        put_prem = float(row.get("put_premium") or 0)

        # Bias (simple)
        vol_bias = "CALL" if call_vol > put_vol else ("PUT" if put_vol > call_vol else "NEUTRAL")
        prem_bias = "CALL" if call_prem > put_prem else ("PUT" if put_prem > call_prem else "NEUTRAL")

        return {
            "call_volume": call_vol,
            "put_volume": put_vol,
            "call_premium": call_prem,
            "put_premium": put_prem,
            "vol_bias": vol_bias,
            "prem_bias": prem_bias,
        }, "ok"
    except Exception:
        return None, "exception"


@st.cache_data(ttl=15, show_spinner=False)
def uw_flow_alerts(limit: int = 300):
    """
    Correct endpoint (from docs):
    https://api.unusualwhales.com/api/option-trade/flow-alerts

    IMPORTANT: must use Authorization: Bearer <token>

    Returns list of flow alerts or None.
    """
    if not UW_TOKEN:
        return None, "missing_key"

    # If user did not put URL, fall back to correct default:
    base = UW_FLOW_ALERTS_URL or "https://api.unusualwhales.com/api/option-trade/flow-alerts"
    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}",
    }
    params = {"limit": int(limit)}

    try:
        r = safe_get(base, params=params, headers=headers, timeout=25)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"

        data = r.json()
        items = data.get("data", data if isinstance(data, list) else [])
        if not isinstance(items, list):
            return None, "bad_schema"

        return items, "ok"
    except Exception:
        return None, "exception"


def filter_flow_alerts(items, *, min_premium=1_000_000, max_dte=3, require_vol_gt_oi=True, exclude_itm=True):
    """
    Best-effort filtering based on fields that may exist.
    We do not crash if schema differs.
    """
    out = []
    for it in items or []:
        try:
            prem = float(it.get("premium") or 0)
            dte = it.get("dte")
            if dte is None:
                # sometimes you can infer from expiry
                expiry = it.get("expiry")
                if expiry:
                    exp = datetime.fromisoformat(str(expiry)).replace(tzinfo=None)
                    dte = (exp.date() - now_cst().date()).days
            dte = int(dte) if dte is not None else 999

            oi = float(it.get("open_interest") or 0)
            vol = float(it.get("volume") or 0)
            # itm detection often not provided; tags sometimes include it
            tags = it.get("tags") or []
            tags_str = " ".join(tags).lower() if isinstance(tags, list) else str(tags).lower()
            is_itm = ("itm" in tags_str) or ("in_the_money" in tags_str)

            if prem < min_premium:
                continue
            if dte > max_dte:
                continue
            if require_vol_gt_oi and not (vol > oi and oi > 0):
                continue
            if exclude_itm and is_itm:
                continue

            # Stock/ETF only: best-effort via issue/industry fields if present
            out.append(it)
        except Exception:
            continue
    return out


# =========================
#   10Y YIELD (optional)
# =========================
@st.cache_data(ttl=120, show_spinner=False)
def fetch_10y_yield_optional():
    """
    Optional best-effort 10Y yield (won't break app if unavailable).
    Uses a public Stooq CSV as a lightweight source (no key).
    """
    try:
        # Stooq symbol often works like "us10y" (not guaranteed).
        url = "https://stooq.com/q/l/?s=us10y&i=d"
        r = safe_get(url, timeout=15)
        if r.status_code != 200:
            return None, "http"
        text = r.text.strip().splitlines()
        if len(text) < 2:
            return None, "empty"
        # CSV: Symbol,Date,Open,High,Low,Close,Volume
        parts = text[1].split(",")
        if len(parts) < 6:
            return None, "bad"
        close = float(parts[5])
        return close, "ok"
    except Exception:
        return None, "exception"


# =========================
#   SCORING / SIGNALS
# =========================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def score_to_confidence(bull_score_0_1: float) -> int:
    # Convert 0..1 to 0..100
    return int(round(100 * clamp01(bull_score_0_1)))

def build_signal_row(ticker: str, df: pd.DataFrame, uw_bias: dict | None, flow_for_ticker: list, y10: float | None,
                     weights: dict, institutional_cutoff: int):
    """
    Produces a CALLS/PUTS-only signal with confidence 0-100.
    """

    # If we can't compute indicators, return stable row
    if df is None or len(df) < 30:
        return {
            "Ticker": ticker,
            "Confidence": 50,
            "Direction": "—",
            "Signal": "WAIT",
            "UW Unusual": "NO",
            "UW Bias": (uw_bias.get("prem_bias") if uw_bias else "N/A"),
            "IV spike": "N/A",
            "Gamma bias": "N/A",
            "RSI": None,
            "MACD_hist": None,
            "VWAP": None,
            "EMA stack": None,
            "Vol_ratio": None,
            "10Y": (f"{y10:.2f}" if isinstance(y10, (int, float)) else "N/A"),
        }

    close = df["close"].copy()
    vol = df["volume"].copy()

    rsi14 = rsi(close, 14).iloc[-1]
    macdh = macd_hist(close).iloc[-1]

    vwap_series = vwap(df)
    vwap_last = float(vwap_series.iloc[-1])
    price_last = float(close.iloc[-1])

    ema9 = float(ema(close, 9).iloc[-1])
    ema20 = float(ema(close, 20).iloc[-1])
    ema50 = float(ema(close, 50).iloc[-1])

    # Volume ratio: last bar vs 20-bar average
    vol_avg = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
    vol_ratio = (float(vol.iloc[-1]) / (vol_avg if vol_avg > 0 else 1e-9))

    # EMA stack label
    if ema9 > ema20 > ema50:
        ema_stack = "Bull"
    elif ema9 < ema20 < ema50:
        ema_stack = "Bear"
    else:
        ema_stack = "Mixed"

    # VWAP above/below
    vwap_pos = "Above" if price_last >= vwap_last else "Below"

    # --- UW flow: determine unusual + IV spike + gamma proxy ---
    unusual = "NO"
    iv_spike = "None"
    gamma_bias = "Neutral"

    # If we have filtered flow alerts for this ticker, treat as "unusual"
    if flow_for_ticker:
        unusual = "YES"

        # IV spike: compare max IV in recent alerts vs median
        ivs = []
        gammas = []
        net_bull = 0
        for it in flow_for_ticker:
            try:
                iv = float(it.get("implied_volatility") or 0)
                if iv > 0:
                    ivs.append(iv)
                g = float(it.get("gamma") or 0)
                if g != 0:
                    gammas.append(g)

                # Use tags to bias bullish/bearish
                tags = it.get("tags") or []
                tags_str = " ".join(tags).lower() if isinstance(tags, list) else str(tags).lower()
                if "bullish" in tags_str:
                    net_bull += 1
                if "bearish" in tags_str:
                    net_bull -= 1
            except Exception:
                continue

        if ivs:
            med = sorted(ivs)[len(ivs)//2]
            mx = max(ivs)
            # spike if max is 25% above median (tunable)
            if med > 0 and mx / med >= 1.25:
                iv_spike = "YES"
            else:
                iv_spike = "no"

        # gamma proxy: sign of avg gamma + tag net
        if gammas:
            avg_g = sum(gammas)/len(gammas)
            if avg_g > 0:
                gamma_bias = "Positive (proxy)"
            elif avg_g < 0:
                gamma_bias = "Negative (proxy)"

        if net_bull > 0:
            gamma_bias = "Positive (proxy)"
        elif net_bull < 0:
            gamma_bias = "Negative (proxy)"

    # --- UW options-volume bias ---
    uw_bias_label = "N/A"
    if uw_bias:
        uw_bias_label = f"{uw_bias.get('prem_bias','N/A')} prem / {uw_bias.get('vol_bias','N/A')} vol"

    # --- Optional 10Y yield filter: if yield spikes higher, slightly bearish risk-on intraday ---
    y10_penalty = 0.0
    if isinstance(y10, (int, float)):
        # crude: treat >= 4.5 as slightly bearish (tunable); you can adjust later
        if y10 >= 4.5:
            y10_penalty = 0.05 * weights["y10"]

    # --- Convert indicators into 0..1 bull signals ---
    # RSI bull: 0 at 30-, 1 at 70+
    rsi_bull = clamp01((rsi14 - 30.0) / 40.0)

    # MACD bull: positive hist => bull, scaled
    macd_bull = 0.5 + clamp01(macdh * 50.0) - 0.5  # small scale
    macd_bull = clamp01(macd_bull)

    # VWAP bull
    vwap_bull = 1.0 if price_last >= vwap_last else 0.0

    # EMA stack bull
    ema_bull = 1.0 if ema_stack == "Bull" else (0.0 if ema_stack == "Bear" else 0.5)

    # Volume bull (spike helps momentum)
    vol_bull = clamp01((vol_ratio - 1.0) / 2.0)  # ratio 1 => 0, ratio 3 => 1

    # UW flow bull
    uw_flow_bull = 0.5
    if unusual == "YES":
        # If gamma bias positive, bull; negative, bear
        if "Positive" in gamma_bias:
            uw_flow_bull = 0.75
        elif "Negative" in gamma_bias:
            uw_flow_bull = 0.25
        else:
            uw_flow_bull = 0.6

    # News bull is not "sentiment model" here; we just use presence of news as small factor
    # (you can add real sentiment later)
    news_bull = 0.5  # neutral placeholder

    # Weighted bull score (0..1)
    bull = (
        weights["rsi"] * rsi_bull +
        weights["macd"] * macd_bull +
        weights["vwap"] * vwap_bull +
        weights["ema"] * ema_bull +
        weights["vol"] * vol_bull +
        weights["uw"] * uw_flow_bull +
        weights["news"] * news_bull
    )

    # Apply 10Y penalty (bearish tilt when high)
    bull = clamp01(bull - y10_penalty)

    confidence = score_to_confidence(bull)

    # Direction (CALLS vs PUTS only)
    direction = "CALLS" if bull >= 0.5 else "PUTS"

    # Institutional mode: only fire BUY signal if confidence >= cutoff, otherwise WAIT
    if confidence >= institutional_cutoff:
        signal = f"BUY {direction}"
    else:
        signal = "WAIT"

    # Make sure it NEVER returns "SELL" (options-only mode per your request)
    row = {
        "Ticker": ticker,
        "Confidence": confidence,
        "Direction": direction,
        "Signal": signal,
        "UW Unusual": unusual,
        "UW Bias": uw_bias_label,
        "IV spike": iv_spike,
        "Gamma bias": gamma_bias,
        "RSI": round(float(rsi14), 2) if pd.notna(rsi14) else None,
        "MACD_hist": round(float(macdh), 5) if pd.notna(macdh) else None,
        "VWAP": vwap_pos,
        "EMA stack": ema_stack,
        "Vol_ratio": round(float(vol_ratio), 2) if pd.notna(vol_ratio) else None,
        "10Y": (round(float(y10), 2) if isinstance(y10, (int, float)) else "N/A"),
    }
    return row


# =========================
#   SIDEBAR: tickers (ANY ticker)
# =========================
st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")
st.caption(f"Last update (CST): {fmt_cst(now_cst())}")

with st.sidebar:
    st.header("Settings")

    # ANY ticker input
    typed = st.text_input(
        "Type tickers (comma-separated). Example: SPY,TSLA,NVDA",
        value=",".join(DEFAULT_TICKERS[:4]),
    )
    typed_list = [t.strip().upper() for t in typed.split(",") if t.strip()]

    # Also let user pick quickly from defaults (optional convenience)
    picked = st.multiselect("Quick pick (optional)", DEFAULT_TICKERS, default=[])
    tickers = []
    for t in (typed_list + picked):
        if t and t not in tickers:
            tickers.append(t)

    st.divider()
    news_lookback = st.number_input("News lookback (minutes)", min_value=5, max_value=360, value=60, step=5)
    price_lookback = st.number_input("Price lookback (minutes)", min_value=60, max_value=600, value=240, step=30)

    st.divider()
    st.subheader("Refresh")
    refresh_sec = st.slider("Auto-refresh (seconds)", min_value=10, max_value=120, value=30, step=5)

    st.divider()
    st.subheader("Institutional mode")
    institutional_cutoff = st.slider("Signals only if confidence ≥", min_value=50, max_value=95, value=75, step=1)

    st.divider()
    st.subheader("Weights (sum doesn't have to be 1)")
    w_rsi  = st.slider("RSI weight", 0.0, 0.50, 0.15, 0.01)
    w_macd = st.slider("MACD weight", 0.0, 0.50, 0.15, 0.01)
    w_vwap = st.slider("VWAP weight", 0.0, 0.50, 0.15, 0.01)
    w_ema  = st.slider("EMA stack (9/20/50) weight", 0.0, 0.50, 0.18, 0.01)
    w_vol  = st.slider("Volume ratio weight", 0.0, 0.50, 0.12, 0.01)
    w_uw   = st.slider("UW flow weight", 0.0, 0.80, 0.20, 0.01)
    w_news = st.slider("News weight (placeholder)", 0.0, 0.30, 0.05, 0.01)
    w_y10  = st.slider("10Y yield (optional) weight", 0.0, 0.30, 0.05, 0.01)

    weights = {"rsi": w_rsi, "macd": w_macd, "vwap": w_vwap, "ema": w_ema, "vol": w_vol, "uw": w_uw, "news": w_news, "y10": w_y10}

    st.divider()
    st.subheader("Keys status (green/red)")

    def key_badge(label, ok: bool):
        if ok:
            st.success(label)
        else:
            st.error(label)

    key_badge("EODHD_API_KEY", bool(EODHD_API_KEY))
    key_badge("UW_TOKEN (Bearer)", bool(UW_TOKEN))
    key_badge("UW_FLOW_ALERTS_URL", bool(UW_FLOW_ALERTS_URL))

    st.divider()
    st.subheader("Endpoints status")
    # We'll fill these after we run checks
    endpoint_box = st.empty()


# Auto refresh
st_autorefresh(interval=int(refresh_sec) * 1000, key="auto_refresh")


# =========================
#   LEFT: UW Screener (web)
# =========================
# Your screener: filters (premium/DTE/ITM/volume>OI) are best enforced in screener UI itself.
UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF"
    "&limit=250&max_dte=3"
    "&min_premium=1000000"
    "&min_volume_oi_ratio=1"
    "&order=premium&order_direction=desc"
    "&watchlist_name=GPT%20Filter%20"
)

col_left, col_right = st.columns([1.25, 1.0], gap="large")

with col_left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("This is embedded. True filtering is best done inside the screener itself.")
    components.iframe(UW_SCREENER_URL, height=820, scrolling=True)


# =========================
#   DATA PULLS (right side)
# =========================
# Endpoint checks
eodhd_ok = True
uwflow_ok = True
uwvol_ok = True
news_ok = True

# Optional 10Y
y10, y10_status = fetch_10y_yield_optional()
# We treat 10Y failures as "ok" (optional)
y10_label = f"{y10:.2f}" if isinstance(y10, (int, float)) else "N/A"

# UW flow alerts (global pull once)
flow_items, flow_status = uw_flow_alerts(limit=400)
if flow_status != "ok" and flow_status != "missing_key":
    uwflow_ok = False

# Apply filters you requested
filtered_flow = filter_flow_alerts(
    flow_items if isinstance(flow_items, list) else [],
    min_premium=1_000_000,
    max_dte=3,
    require_vol_gt_oi=True,
    exclude_itm=True
)

# Create per ticker flow map, within recent window
flow_cutoff = datetime.now(tz=UTC) - timedelta(minutes=int(price_lookback))
flow_by_ticker = {}
for it in filtered_flow:
    sym = str(it.get("underlying_symbol") or it.get("ticker") or "").upper().strip()
    if not sym:
        continue
    # executed_at is ISO in UTC
    ex = it.get("executed_at")
    try:
        exdt = pd.to_datetime(ex, utc=True, errors="coerce")
        if pd.isna(exdt):
            continue
        if exdt.to_pydatetime().replace(tzinfo=UTC) < flow_cutoff:
            continue
    except Exception:
        pass
    flow_by_ticker.setdefault(sym, []).append(it)


# =========================
#   BUILD TABLE
# =========================
rows = []
debug_rows = []

if not tickers:
    st.warning("Type at least one ticker in the sidebar (example: SPY,TSLA,NVDA).")

for t in tickers:
    sym_us = ensure_us_symbol(t)

    # EODHD intraday candles
    df, df_status = eodhd_intraday(sym_us, interval="5m", lookback_minutes=int(price_lookback))
    if df_status not in ("ok", "few_bars", "empty") and df_status != "missing_key":
        eodhd_ok = False

    # EODHD news
    news_items, news_status = eodhd_news(sym_us, lookback_minutes=int(news_lookback))
    if news_status not in ("ok", "empty") and news_status != "missing_key":
        news_ok = False

    # UW options volume bias
    uw_bias, uw_bias_status = uw_options_volume_bias(t)
    if uw_bias_status not in ("ok", "empty") and uw_bias_status != "missing_key":
        uwvol_ok = False

    flow_for_t = flow_by_ticker.get(t, [])

    row = build_signal_row(
        t, df, uw_bias, flow_for_t, y10,
        weights=weights, institutional_cutoff=int(institutional_cutoff)
    )
    rows.append(row)

    debug_rows.append({
        "Ticker": t,
        "intraday_status": df_status,
        "news_status": news_status,
        "uw_options_volume_status": uw_bias_status,
        "uw_flow_status": flow_status,
        "flow_hits_in_window": len(flow_for_t),
        "bars": (len(df) if isinstance(df, pd.DataFrame) else 0),
        "last_bar(CST)": (df["datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S") if isinstance(df, pd.DataFrame) and len(df) else ""),
    })


# =========================
#   SHOW ENDPOINT STATUS (sidebar)
# =========================
with st.sidebar:
    lines = []
    lines.append(("EODHD intraday", eodhd_ok))
    lines.append(("EODHD news", news_ok))
    lines.append(("UW options-volume", uwvol_ok))
    lines.append(("UW flow-alerts", uwflow_ok))
    # 10Y is optional; show gray if N/A
    # We don't mark failure as red.
    endpoint_box.empty()
    for label, ok in lines:
        if ok:
            st.success(label)
        else:
            st.error(label)

    st.info(f"10Y yield (optional): {y10_label}")


# =========================
#   RIGHT PANEL UI
# =========================
with col_right:
    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW flow)")
    st.caption(f"Last update (CST): {fmt_cst(now_cst())}")

    out_df = pd.DataFrame(rows)

    # Order columns (clean)
    col_order = [
        "Ticker", "Confidence", "Direction", "Signal",
        "UW Unusual", "UW Bias",
        "RSI", "MACD_hist", "VWAP", "EMA stack", "Vol_ratio",
        "IV spike", "Gamma bias", "10Y"
    ]
    out_df = out_df[[c for c in col_order if c in out_df.columns]]

    st.dataframe(out_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Institutional Alerts (≥75 only)")
    inst = out_df[(out_df["Confidence"] >= int(institutional_cutoff)) & (out_df["Signal"] != "WAIT")]
    if inst.empty:
        st.info("No institutional signals right now.")
    else:
        for _, r in inst.sort_values("Confidence", ascending=False).iterrows():
            st.success(f"{r['Ticker']}: {r['Signal']} | Confidence={r['Confidence']} | UW={r['UW Unusual']} | IV={r['IV spike']} | Gamma={r['Gamma bias']}")

    st.divider()
    st.subheader("Unusual Flow Alerts (UW API)")
    st.caption("Rules applied: premium ≥ $1M, DTE ≤ 3, Volume > OI, exclude ITM. (Best-effort based on fields available.)")

    if flow_status == "missing_key":
        st.warning("UW_TOKEN missing in Secrets.")
    elif flow_status != "ok":
        st.error(f"UW flow fetch failed: {flow_status} (check UW_FLOW_ALERTS_URL + token)")
    else:
        # Show a small table of recent filtered flow for selected tickers only
        show = []
        for t in tickers:
            for it in (flow_by_ticker.get(t, []) or [])[:10]:
                show.append({
                    "Ticker": t,
                    "executed_at": str(it.get("executed_at", "")),
                    "option": str(it.get("option_chain_id", ""))[:28],
                    "type": str(it.get("option_type", "")),
                    "premium": it.get("premium"),
                    "dte": it.get("dte"),
                    "iv": it.get("implied_volatility"),
                    "gamma": it.get("gamma"),
                    "tags": ", ".join(it.get("tags") or []) if isinstance(it.get("tags"), list) else str(it.get("tags") or "")
                })
        if not show:
            st.info("No UW flow alerts matching your rules in the current window (normal after-hours sometimes).")
        else:
            show_df = pd.DataFrame(show)
            st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(f"News — last {int(news_lookback)} minutes (EODHD)")
    all_news = []
    for t in tickers:
        sym_us = ensure_us_symbol(t)
        items, status = eodhd_news(sym_us, lookback_minutes=int(news_lookback))
        if isinstance(items, list) and items:
            all_news.extend(items)

    if not all_news:
        st.info("No news in this lookback window (or EODHD returned none).")
    else:
        news_df = pd.DataFrame(all_news).sort_values("Published (CST)", ascending=False)
        st.dataframe(news_df[["Ticker", "Published (CST)", "Source", "Title", "URL"]], use_container_width=True, hide_index=True)
        st.caption("Tip: Click URL column links (or copy/paste).")

    with st.expander("Debug (why something might show N/A)"):
        st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)
