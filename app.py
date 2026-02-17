import os, time
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="5m Score + UW Flow + News", layout="wide")

UTC = timezone.utc

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "AMD", "NVDA"]

# --- Secrets ---
POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY", os.getenv("POLYGON_API_KEY", "")).strip()
UW_API_KEY = st.secrets.get("UNUSUAL_WHALES_API_KEY", os.getenv("UNUSUAL_WHALES_API_KEY", "")).strip()

# You must add this in Secrets once you locate it:
UW_FLOW_ALERTS_URL = st.secrets.get("UW_FLOW_ALERTS_URL", os.getenv("UW_FLOW_ALERTS_URL", "")).strip()

# Contract flow (you already have this)
UW_CONTRACT_FLOW_URL = st.secrets.get(
    "UW_CONTRACT_FLOW_URL",
    os.getenv("UW_CONTRACT_FLOW_URL", "https://api.unusualwhales.com/api/option-contract/{id}/flow")
).strip()

# Your UW screener webpage (iframe only)
UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF&limit=250&max_dte=7"
    "&min_diff=-0.1&min_oi_change_perc=-10&min_premium=500000"
    "&min_volume_oi_ratio=1&min_volume_ticker_vol_ratio=0.03"
    "&order=premium&order_direction=desc&watchlist_name=GPT%20Filter%20"
)

# -----------------------------
# Indicators
# -----------------------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / (dn.ewm(alpha=1/n, adjust=False).mean() + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist

def utc_now():
    return datetime.now(UTC)

# -----------------------------
# Polygon: price (1m bars) -> resample to 5m
# -----------------------------
@st.cache_data(ttl=20)
def polygon_1m_bars(ticker: str, minutes_back: int):
    if not POLYGON_API_KEY:
        raise RuntimeError("Missing POLYGON_API_KEY in Secrets")

    end = utc_now()
    start = end - timedelta(minutes=minutes_back)
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    params = {"adjusted":"true", "sort":"asc", "limit":50000, "apiKey": POLYGON_API_KEY}
    r = requests.get(url, params=params, timeout=20)

    if r.status_code == 429:
        raise RuntimeError("Polygon 429 rate limit. Increase refresh seconds or reduce tickers.")
    r.raise_for_status()

    data = r.json().get("results", []) or []
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["dt"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
    return df[["dt","open","high","low","close","volume"]].set_index("dt")

def to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    if df_1m.empty:
        return df_1m
    o = df_1m["open"].resample("5min").first()
    h = df_1m["high"].resample("5min").max()
    l = df_1m["low"].resample("5min").min()
    c = df_1m["close"].resample("5min").last()
    v = df_1m["volume"].resample("5min").sum()
    out = pd.concat([o,h,l,c,v], axis=1).dropna()
    out.columns = ["open","high","low","close","volume"]
    return out

# -----------------------------
# Polygon: News (cached)
# -----------------------------
@st.cache_data(ttl=120)
def polygon_news(ticker: str, minutes_back: int):
    if not POLYGON_API_KEY:
        return []
    since = utc_now() - timedelta(minutes=minutes_back)
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker,
        "published_utc.gte": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 10,
        "order":"desc",
        "sort":"published_utc",
        "apiKey": POLYGON_API_KEY
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code == 429:
        return [{"_rate_limited": True}]
    r.raise_for_status()
    return r.json().get("results", []) or []

POS = {"beat","beats","surge","soar","upgrade","upgraded","strong","bull","bullish","profit","approval","approved","record"}
NEG = {"miss","misses","drop","plunge","downgrade","lawsuit","probe","fraud","weak","bear","bearish","loss","halt","recall"}

def headline_sentiment(title: str) -> float:
    if not title:
        return 0.0
    t = title.lower()
    p = sum(1 for w in POS if w in t)
    n = sum(1 for w in NEG if w in t)
    if p+n == 0:
        return 0.0
    return (p-n)/(p+n)

# -----------------------------
# Unusual Whales: Flow Alerts (list) + Contract Flow drilldown
# -----------------------------
def uw_headers():
    if not UW_API_KEY:
        raise RuntimeError("Missing UNUSUAL_WHALES_API_KEY in Secrets")
    return {"Authorization": f"Bearer {UW_API_KEY}", "Accept": "application/json, text/plain"}

@st.cache_data(ttl=20)
def uw_flow_alerts():
    """
    This must be a LIST endpoint (flow alerts / option trades alerts).
    Put its full URL in UW_FLOW_ALERTS_URL secret.
    """
    if not UW_FLOW_ALERTS_URL:
        return None
    r = requests.get(UW_FLOW_ALERTS_URL, headers=uw_headers(), timeout=20)
    if r.status_code == 429:
        return {"_rate_limited": True}
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60)
def uw_contract_flow(contract_id: str):
    url = UW_CONTRACT_FLOW_URL.replace("{id}", contract_id)
    r = requests.get(url, headers=uw_headers(), timeout=20)
    if r.status_code == 429:
        return {"_rate_limited": True}
    r.raise_for_status()
    return r.json()

# -----------------------------
# Scoring (0-100) + BUY/SELL
# -----------------------------
def compute_score(df5: pd.DataFrame, news_items: list, flow_hit: bool,
                  w_rsi=0.25, w_macd=0.25, w_trend=0.20, w_vol=0.20, w_news=0.10):
    if df5.empty or len(df5) < 30:
        return {"score": 0, "bias":"Neutral", "signal":"WAIT", "rsi":None, "macd_hist":None, "vol_spike":None, "news":0.0}

    close = df5["close"].astype(float)
    vol = df5["volume"].astype(float)

    r = float(rsi(close, 14).iloc[-1])
    _, _, hist = macd(close)
    mh = float(hist.iloc[-1])

    ema20 = float(ema(close, 20).iloc[-1])
    ema50 = float(ema(close, 50).iloc[-1]) if len(close) >= 50 else float(ema(close, 30).iloc[-1])
    trend_ok = ema20 > ema50

    vol_med = float(np.median(vol.values)) if len(vol) else 0.0
    vol_last = float(vol.iloc[-1]) if len(vol) else 0.0
    vol_spike = (vol_last / vol_med) if vol_med > 0 else 1.0

    # news sentiment
    news_sent = 0.0
    if news_items and not (isinstance(news_items, list) and news_items and news_items[0].get("_rate_limited")):
        s = [headline_sentiment(x.get("title","")) for x in news_items]
        news_sent = float(np.mean(s)) if s else 0.0

    # normalize components to 0..1 bullish
    if r <= 30: rsi_comp = 1.0
    elif r >= 70: rsi_comp = 0.0
    else: rsi_comp = (70 - r) / 40.0

    macd_comp = 0.5 + max(-0.5, min(0.5, mh * 5))
    trend_comp = 0.75 if trend_ok else 0.25
    vol_comp = max(0.0, min(1.0, (vol_spike - 1.0) / 2.0))  # 3x => 1.0
    news_comp = 0.5 + max(-0.5, min(0.5, news_sent))

    total = w_rsi + w_macd + w_trend + w_vol + w_news
    bull = (w_rsi*rsi_comp + w_macd*macd_comp + w_trend*trend_comp + w_vol*vol_comp + w_news*news_comp) / (total or 1.0)

    # small boost if UW flow triggered
    if flow_hit:
        bull = min(1.0, bull + 0.08)

    score = int(round(bull * 100))
    bias = "Bullish" if score >= 60 else ("Bearish" if score <= 40 else "Neutral")

    # signals (tuned for 5m trading)
    if score >= 75 and trend_ok and mh > 0:
        signal = "BUY"
    elif score <= 25 and (not trend_ok) and mh < 0:
        signal = "SELL"
    else:
        signal = "WAIT"

    return {
        "score": score,
        "bias": bias,
        "signal": signal,
        "rsi": round(r, 1),
        "macd_hist": round(mh, 4),
        "vol_spike": round(vol_spike, 2),
        "news": round(news_sent, 2),
    }

# -----------------------------
# UI
# -----------------------------
st.title("📈 5m Live Score + 🐋 Unusual Whales Flow + 🗞️ News")

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY","QQQ","DIA","IWM"])
    price_minutes = st.number_input("Price lookback (minutes)", 60, 1000, 390, 15)  # ~1 trading day
    news_minutes = st.number_input("News lookback (minutes)", 5, 240, 60, 5)
    refresh = st.slider("Auto-refresh (seconds)", 30, 600, 120, 30)

    st.divider()
    st.subheader("Status")
    st.write("Polygon key:", "✅" if POLYGON_API_KEY else "❌")
    st.write("UW key:", "✅" if UW_API_KEY else "❌")
    st.write("UW Flow Alerts URL:", "✅" if UW_FLOW_ALERTS_URL else "❌ (add UW_FLOW_ALERTS_URL in Secrets)")

st_autorefresh(interval=int(refresh*1000), key="refresh")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Unusual Whales Screener (web view)")
    st.components.v1.iframe(UW_SCREENER_URL, height=850, scrolling=True)

with col2:
    st.subheader("Scores / Signals (5m)")
    st.write(f"Updated (UTC): **{utc_now().strftime('%Y-%m-%d %H:%M:%S')}**")

    # Fetch flow alerts list once
    flow_json = uw_flow_alerts()
    flow_rate_limited = isinstance(flow_json, dict) and flow_json.get("_rate_limited")

    # Build a quick “ticker hit” map from flow alerts json (structure varies by endpoint)
    flow_hits = {t: False for t in tickers}
    contract_ids_by_ticker = {t: [] for t in tickers}

    if flow_json and not flow_rate_limited:
        # Try to find records in common keys
        records = None
        if isinstance(flow_json, dict):
            records = flow_json.get("data") or flow_json.get("results") or flow_json.get("alerts")
        elif isinstance(flow_json, list):
            records = flow_json

        if isinstance(records, list):
            for r in records:
                sym = r.get("underlying_symbol") or r.get("ticker") or r.get("symbol")
                cid = r.get("id") or r.get("option_contract_id") or r.get("contract_id")
                if sym in flow_hits:
                    flow_hits[sym] = True
                    if cid:
                        contract_ids_by_ticker[sym].append(cid)

    rows = []
    for t in tickers:
        try:
            df1 = polygon_1m_bars(t, int(price_minutes))
            df5 = to_5m(df1)

            news = polygon_news(t, int(news_minutes))
            flow_hit = bool(flow_hits.get(t, False))

            out = compute_score(df5, news, flow_hit)

            rows.append({
                "Ticker": t,
                "Score": out["score"],
                "Bias": out["bias"],
                "Signal": out["signal"],
                "UW Flow Trigger": "YES" if flow_hit else ("429" if flow_rate_limited else "NO"),
                "RSI14": out["rsi"],
                "MACD_hist": out["macd_hist"],
                "VolSpike": out["vol_spike"],
                "NewsSent": out["news"],
            })

            time.sleep(0.15)  # gentle rate limiting
        except Exception as e:
            rows.append({"Ticker": t, "Score":"ERR", "Bias":"", "Signal":"", "UW Flow Trigger":"", "RSI14":"", "MACD_hist":"", "VolSpike":"", "NewsSent":"", "Error": str(e)})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("UW Contract Flow Drilldown (optional)")
    st.caption("Pick a ticker → then pick a contract id → see the last trades for that contract.")
    t_pick = st.selectbox("Ticker", tickers if tickers else ["SPY"])
    cands = contract_ids_by_ticker.get(t_pick, [])
    if not UW_FLOW_ALERTS_URL:
        st.info("Add UW_FLOW_ALERTS_URL in Secrets to populate contract IDs automatically.")
    elif not cands:
        st.info("No contract IDs found from your flow-alerts endpoint response yet.")
    else:
        cid = st.selectbox("Contract ID", cands[:25])
        if st.button("Load contract flow"):
            try:
                cf = uw_contract_flow(cid)
                st.json(cf)
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("News (clickable)")
    for t in tickers:
        items = polygon_news(t, int(news_minutes))
        if isinstance(items, list) and items and items[0].get("_rate_limited"):
            st.warning(f"{t}: Polygon news rate-limited (429). Increase refresh seconds.")
            continue
        for it in (items or [])[:5]:
            title = it.get("title","(no title)")
            url = it.get("article_url","")
            if url:
                st.markdown(f"- **{t}** — [{title}]({url})")

