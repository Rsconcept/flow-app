import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from streamlit_autorefresh import st_autorefresh

# =========================
# SETTINGS / CONSTANTS
# =========================
st.set_page_config(page_title="Institutional Options Signals", layout="wide")

CST = ZoneInfo("America/Chicago")

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "NVDA", "AMD"]

# Your UW screener link (web view)
UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF"
    "&limit=250&max_dte=3"
    "&min_premium=1000000"
    "&min_volume_oi_ratio=1"
    "&order=premium&order_direction=desc"
)

# Secrets (Streamlit -> Settings -> Secrets)
EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")).strip()
UW_TOKEN = st.secrets.get("UW_TOKEN", os.getenv("UW_TOKEN", "")).strip()

# UW Flow Alerts endpoint (this is the one you want)
UW_FLOW_ALERTS_URL = "https://api.unusualwhales.com/api/option-trade/flow-alerts"

# =========================
# TIME HELPERS
# =========================
def now_cst_dt():
    return datetime.now(CST)

def now_cst_str():
    return now_cst_dt().strftime("%Y-%m-%d %H:%M:%S")

def minutes_ago_cst(minutes: int):
    return now_cst_dt() - timedelta(minutes=int(minutes))

# =========================
# SAFE HELPERS
# =========================
def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def ok_bad(label, ok: bool, detail: str = ""):
    if ok:
        st.success(f"{label} ✅ {detail}".strip())
    else:
        st.error(f"{label} ❌ {detail}".strip())

# =========================
# EODHD DATA
# =========================
def eodhd_intraday_5m(ticker: str, bars: int = 200):
    # IMPORTANT: EODHD needs .US for US stocks/ETFs
    url = f"https://eodhd.com/api/intraday/{ticker}.US"
    params = {
        "api_token": EODHD_API_KEY,
        "interval": "5m",
        "fmt": "json",
        "outputsize": bars
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df.set_index("datetime", inplace=True)
    return df

def eodhd_news(ticker: str, lookback_minutes: int = 60, limit: int = 20):
    # EODHD news endpoint
    # https://eodhd.com/financial-apis/news-api/
    url = "https://eodhd.com/api/news"
    start = minutes_ago_cst(lookback_minutes).astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")
    end = now_cst_dt().astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")

    params = {
        "s": f"{ticker}.US",
        "from": start,
        "to": end,
        "limit": limit,
        "api_token": EODHD_API_KEY,
        "fmt": "json"
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return []
    return data

def eodhd_10y_yield():
    # If this symbol doesn't work on your plan, we fail gracefully.
    url = "https://eodhd.com/api/real-time/US10Y.BOND"
    params = {"api_token": EODHD_API_KEY, "fmt": "json"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    return safe_float(js.get("close"))

# =========================
# UNUSUAL WHALES DATA
# =========================
def uw_headers():
    return {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {UW_TOKEN}"
    }

def uw_options_volume_bias(ticker: str):
    # Correct endpoint you provided:
    # https://api.unusualwhales.com/api/stock/{ticker}/options-volume
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/options-volume"
    r = requests.get(url, headers=uw_headers(), timeout=20)
    r.raise_for_status()
    js = r.json()
    if not js or "data" not in js or not js["data"]:
        return None
    return js["data"][0]

def uw_flow_alerts(limit: int = 200):
    # Flow alerts endpoint (institutional triggers)
    # Docs: PublicApi.OptionTradeController.flow_alerts
    # Using the URL you provided
    params = {"limit": limit}
    r = requests.get(UW_FLOW_ALERTS_URL, headers=uw_headers(), params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    # Some UW endpoints wrap in {"data":[...]} others return list; handle both
    if isinstance(js, dict) and "data" in js:
        return js["data"]
    if isinstance(js, list):
        return js
    return []

def normalize_flow_alerts(alerts: list):
    # Robust normalization (because UW fields vary by plan/version)
    rows = []
    for a in alerts:
        ticker = a.get("symbol") or a.get("underlying_symbol") or a.get("ticker") or ""
        premium = safe_float(a.get("premium") or a.get("total_premium") or a.get("notional") or 0)
        option_type = (a.get("option_type") or a.get("type") or "").lower()
        side = (a.get("side") or "").lower()
        expiry = a.get("expiry") or a.get("expiration") or None
        strike = safe_float(a.get("strike") or np.nan)
        underlying_price = safe_float(a.get("underlying_price") or a.get("stock_price") or np.nan)
        executed_at = a.get("executed_at") or a.get("timestamp") or a.get("created_at") or ""
        tags = a.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]

        rows.append({
            "ticker": ticker,
            "premium": premium,
            "type": option_type,
            "side": side,
            "expiry": expiry,
            "strike": strike,
            "underlying_price": underlying_price,
            "executed_at": executed_at,
            "tags": ", ".join(tags[:6]) if tags else ""
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Sort by biggest premium
    df = df.sort_values("premium", ascending=False)
    return df

# =========================
# TECHNICAL INDICATORS
# =========================
def calc_indicators(df: pd.DataFrame):
    df = df.copy()

    # EMA stack
    df["EMA9"] = df["close"].ewm(span=9).mean()
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()

    # VWAP (cumulative)
    df["VWAP"] = (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Volume ratio (20-bar)
    df["Vol_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

    # IV spike proxy via ATR expansion (simple + stable)
    tr = (df["high"] - df["low"]).abs()
    df["ATR"] = tr.rolling(14).mean()
    df["IV_spike"] = df["ATR"] > (df["ATR"].rolling(20).mean() * 1.5)

    return df

# =========================
# SCORE ENGINE (0–100 + CALL/PUT)
# =========================
def score_engine(df: pd.DataFrame, ticker: str, y10: float | None, uw_bias: dict | None):
    last = df.iloc[-1]

    score = 50.0

    # Trend via EMA stack
    if last["EMA9"] > last["EMA20"] > last["EMA50"]:
        score += 12
        trend_dir = "bull"
    elif last["EMA9"] < last["EMA20"] < last["EMA50"]:
        score -= 12
        trend_dir = "bear"
    else:
        trend_dir = "neutral"

    # VWAP bias
    if last["close"] > last["VWAP"]:
        score += 10
    else:
        score -= 10

    # RSI bias
    if last["RSI"] >= 55:
        score += 8
    elif last["RSI"] <= 45:
        score -= 8

    # MACD histogram
    if last["MACD_hist"] > 0:
        score += 8
    else:
        score -= 8

    # Volume ratio (institutional participation)
    if last["Vol_ratio"] >= 1.5:
        score += 6
    elif last["Vol_ratio"] <= 0.8:
        score -= 3

    # IV spike proxy
    if bool(last["IV_spike"]):
        score += 5

    # 10Y yield filter (risk-off sensitivity)
    if y10 is not None:
        # Higher yields = pressure on growth / risk assets
        if y10 >= 4.2:
            score -= 6
        elif y10 <= 4.0:
            score += 6

    # UW options volume premium bias (your requested endpoint)
    gamma_bias = "Neutral"
    uw_bias_str = "Neutral"
    if uw_bias:
        bull = safe_float(uw_bias.get("bullish_premium"))
        bear = safe_float(uw_bias.get("bearish_premium"))
        if np.isfinite(bull) and np.isfinite(bear):
            if bull > bear:
                score += 12
                uw_bias_str = "Bullish"
                gamma_bias = "Positive Gamma (proxy)"
            elif bear > bull:
                score -= 12
                uw_bias_str = "Bearish"
                gamma_bias = "Negative Gamma (proxy)"

    score = float(np.clip(score, 0, 100))

    # Calls / puts only decision
    if score >= 75:
        signal = "BUY CALLS"
        direction = "CALL"
    elif score <= 25:
        signal = "BUY PUTS"
        direction = "PUT"
    else:
        signal = "WAIT"
        direction = "—"

    return {
        "Ticker": ticker,
        "Confidence": round(score, 1),
        "Direction": direction,
        "Signal": signal,
        "UW Bias": uw_bias_str,
        "Gamma bias": gamma_bias,
        "RSI": round(float(last["RSI"]), 1) if np.isfinite(last["RSI"]) else None,
        "MACD_hist": round(float(last["MACD_hist"]), 4) if np.isfinite(last["MACD_hist"]) else None,
        "VWAP": "Above" if last["close"] > last["VWAP"] else "Below",
        "EMA stack": trend_dir,
        "Vol_ratio": round(float(last["Vol_ratio"]), 2) if np.isfinite(last["Vol_ratio"]) else None,
        "IV spike": bool(last["IV_spike"]),
    }

# =========================
# UI
# =========================
st.title("🏛 Institutional Options Signals (5m) — CALLS / PUTS ONLY")
st.caption(f"Last update (CST): {now_cst_str()}")

# Auto refresh
refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 10, 300, 30, step=5)
st_autorefresh(interval=refresh_seconds * 1000, key="auto_refresh")

# Inputs
with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY", "TSLA"])
    news_lookback = st.number_input("News lookback (minutes)", min_value=5, max_value=240, value=60, step=5)

    st.divider()
    st.subheader("Keys status (green/red)")

    # Keys present?
    ok_bad("EODHD_API_KEY", bool(EODHD_API_KEY), "(missing)" if not EODHD_API_KEY else "")
    ok_bad("UW_TOKEN (Bearer)", bool(UW_TOKEN), "(missing)" if not UW_TOKEN else "")

    st.caption("If a key is red, go Streamlit → App → Settings → Secrets.")

# ======= API / Endpoint health checks (REAL checks) =======
with st.sidebar:
    st.subheader("Endpoints status")

    # EODHD intraday check (uses SPY as canary)
    intraday_ok = False
    intraday_err = ""
    if EODHD_API_KEY:
        try:
            _df = eodhd_intraday_5m("SPY", bars=10)
            intraday_ok = _df is not None and len(_df) > 5
        except Exception as e:
            intraday_err = str(e)[:80]
    ok_bad("EODHD intraday", intraday_ok, intraday_err)

    # EODHD news check
    news_ok = False
    news_err = ""
    if EODHD_API_KEY:
        try:
            _n = eodhd_news("SPY", lookback_minutes=60, limit=3)
            news_ok = isinstance(_n, list)
        except Exception as e:
            news_err = str(e)[:80]
    ok_bad("EODHD news", news_ok, news_err)

    # 10Y check
    y10_ok = False
    y10_err = ""
    y10_val = None
    if EODHD_API_KEY:
        try:
            y10_val = eodhd_10y_yield()
            y10_ok = y10_val is not None and np.isfinite(y10_val)
        except Exception as e:
            y10_err = str(e)[:80]
    ok_bad("10Y yield", y10_ok, f"{y10_val:.2f}" if y10_ok else y10_err)

    # UW options-volume check
    uw_vol_ok = False
    uw_vol_err = ""
    if UW_TOKEN:
        try:
            _v = uw_options_volume_bias("SPY")
            uw_vol_ok = _v is not None
        except Exception as e:
            uw_vol_err = str(e)[:80]
    ok_bad("UW options-volume", uw_vol_ok, uw_vol_err)

    # UW flow-alerts check
    uw_flow_ok = False
    uw_flow_err = ""
    if UW_TOKEN:
        try:
            _a = uw_flow_alerts(limit=5)
            uw_flow_ok = isinstance(_a, list)
        except Exception as e:
            uw_flow_err = str(e)[:80]
    ok_bad("UW flow-alerts", uw_flow_ok, uw_flow_err)

# ======= Layout =======
left, right = st.columns([1.25, 1])

# LEFT: UW Screener web view (option flow)
with left:
    st.subheader("Unusual Whales Screener (web view)")
    st.caption("This is embedded. Use UW UI for filters: $1M+ premium, DTE<=3, stocks/ETFs only, exclude ITM, volume>OI.")
    st.components.v1.iframe(UW_SCREENER_URL, height=860, scrolling=True)

# RIGHT: Score table + alerts + news + flow alerts
with right:
    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW flow)")

    if not tickers:
        st.info("Pick at least 1 ticker in the left sidebar.")
        st.stop()

    # Build table
    rows = []
    debug_rows = []

    for t in tickers:
        try:
            df = eodhd_intraday_5m(t, bars=200)
            if df is None or len(df) < 60:
                debug_rows.append({"Ticker": t, "Reason": "intraday_not_enough_bars", "Bars": 0 if df is None else len(df)})
                continue

            df = calc_indicators(df)

            uw_bias = None
            try:
                uw_bias = uw_options_volume_bias(t) if UW_TOKEN else None
            except:
                uw_bias = None

            row = score_engine(df, t, y10_val if y10_ok else None, uw_bias)
            rows.append(row)

            debug_rows.append({"Ticker": t, "Reason": "ok", "Bars": len(df)})

        except Exception as e:
            debug_rows.append({"Ticker": t, "Reason": f"error: {str(e)[:60]}", "Bars": 0})

    score_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "Ticker","Confidence","Direction","Signal","UW Bias","Gamma bias","RSI","MACD_hist","VWAP","EMA stack","Vol_ratio","IV spike"
    ])

    st.dataframe(score_df, use_container_width=True, hide_index=True)

    # Institutional alerts >= 75
    st.subheader("Alerts (institutional)")
    inst = score_df[score_df["Confidence"] >= 75] if not score_df.empty else score_df
    if inst.empty:
        st.info("No institutional signals (confidence < 75 for all tickers).")
    else:
        for _, r in inst.sort_values("Confidence", ascending=False).iterrows():
            st.success(f"{r['Ticker']}: {r['Signal']} | Conf={r['Confidence']} | UW Bias={r['UW Bias']} | VWAP={r['VWAP']} | IV_spike={r['IV spike']}")

    # UW Flow Alerts (live trigger feed)
    st.subheader("Unusual Flow Alerts (UW API)")
    st.caption("This is the *API feed*. If UW endpoint returns nothing after-hours, this will be quiet.")

    if not UW_TOKEN:
        st.warning("UW_TOKEN missing in Secrets.")
    else:
        try:
            alerts_raw = uw_flow_alerts(limit=200)
            alerts_df = normalize_flow_alerts(alerts_raw)

            # Filter your rules (best-effort; fields vary by plan)
            # - Premium >= $1,000,000
            # - Stocks/ETFs only: (not always labeled; we filter by ticker list)
            # - DTE <= 3: needs expiry (optional)
            # - Exclude ITM: needs strike & underlying (optional)
            if not alerts_df.empty:
                alerts_df = alerts_df[alerts_df["premium"] >= 1_000_000]

                # Only show chosen tickers
                alerts_df = alerts_df[alerts_df["ticker"].isin(tickers)]

                # Add DTE if expiry present
                def dte(exp):
                    try:
                        if not exp:
                            return np.nan
                        exp_dt = pd.to_datetime(exp).tz_localize(None)
                        return (exp_dt.date() - now_cst_dt().date()).days
                    except:
                        return np.nan

                alerts_df["DTE"] = alerts_df["expiry"].apply(dte)

                # Apply DTE<=3 only if we have DTE
                alerts_df = alerts_df[(alerts_df["DTE"].isna()) | (alerts_df["DTE"] <= 3)]

                # Exclude ITM if we have enough fields
                def is_itm(row):
                    try:
                        if np.isnan(row["strike"]) or np.isnan(row["underlying_price"]):
                            return False
                        if row["type"] == "call":
                            return row["underlying_price"] > row["strike"]
                        if row["type"] == "put":
                            return row["underlying_price"] < row["strike"]
                        return False
                    except:
                        return False

                alerts_df = alerts_df[~alerts_df.apply(is_itm, axis=1)]

                st.dataframe(
                    alerts_df[["ticker","premium","type","side","DTE","strike","underlying_price","executed_at","tags"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No flow alerts returned right now (normal after-hours, or your plan/endpoint may limit this feed).")

        except Exception as e:
            st.error(f"UW flow alerts error: {e}")

    # News feed (EODHD)
    st.subheader(f"News — last {int(news_lookback)} minutes (EODHD)")
    if not EODHD_API_KEY:
        st.warning("EODHD_API_KEY missing in Secrets.")
    else:
        news_frames = []
        for t in tickers:
            try:
                items = eodhd_news(t, lookback_minutes=int(news_lookback), limit=20)
                for it in items:
                    news_frames.append({
                        "Ticker": t,
                        "published": it.get("date") or it.get("published") or "",
                        "Source": it.get("source") or "",
                        "Title": it.get("title") or "",
                        "URL": it.get("link") or it.get("url") or ""
                    })
            except:
                pass

        if not news_frames:
            st.info("No news in this lookback window (or feed returned none).")
        else:
            news_df = pd.DataFrame(news_frames)
            st.dataframe(news_df[["Ticker","published","Source","Title","URL"]], use_container_width=True, hide_index=True)

    # Debug (why None)
    with st.expander("Debug (why indicators might be None)"):
        st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)
