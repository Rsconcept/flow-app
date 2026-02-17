import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Flow + News (5m)", layout="wide")

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "AMD", "NVDA"]

# Your Unusual Whales screener link
UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF&limit=250&max_dte=7"
    "&min_diff=-0.1&min_oi_change_perc=-10&min_premium=500000"
    "&min_volume_oi_ratio=1&min_volume_ticker_vol_ratio=0.03"
    "&order=premium&order_direction=desc&watchlist_name=GPT%20Filter%20"
)

# Read Polygon key from Streamlit Secrets first, then env var
POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY", os.getenv("POLYGON_API_KEY", "")).strip()


# -----------------------------
# HELPERS
# -----------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@st.cache_data(ttl=120)
def get_polygon_news(ticker: str, api_key: str):
    """
    Pull latest news from Polygon.
    Cached to reduce rate limits.
    """
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker,
        "limit": 50,
        "order": "desc",
        "sort": "published_utc",
        "apiKey": api_key,
    }

    r = requests.get(url, params=params, timeout=20)
    # Let caller handle status codes with messages
    return r.status_code, r.json() if r.content else {}


def parse_published_utc(ts: str):
    """
    Polygon returns timestamps like '2026-02-17T20:25:55Z'
    Convert to datetime in UTC.
    """
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


# -----------------------------
# UI
# -----------------------------
st.title("📈 Option Flow (Unusual Whales) + 🗞️ News (Polygon) — last X minutes")

# Auto refresh every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="refresh_5min")

with st.sidebar:
    st.header("Settings")

    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["SPY", "QQQ", "DIA", "IWM"])
    minutes = st.number_input("News lookback (minutes)", min_value=1, max_value=60, value=5, step=1)

    st.divider()
    st.subheader("Polygon API Key")

    if POLYGON_API_KEY:
        st.success("Polygon key loaded ✅")
    else:
        st.warning("Polygon key missing. Add it in Streamlit → App settings → Secrets (POLYGON_API_KEY).")

    st.caption("App auto-refreshes every 5 minutes.")

col1, col2 = st.columns([1.15, 1])

with col1:
    st.subheader("Unusual Whales — your screener")
    st.write("This is your exact screener link embedded below.")
    st.components.v1.iframe(UW_SCREENER_URL, height=900, scrolling=True)

with col2:
    st.subheader(f"Polygon News — last {int(minutes)} minutes")
    st.write(f"Last update (UTC): **{utc_now().strftime('%Y-%m-%d %H:%M:%S')}**")

    if not tickers:
        st.info("Pick at least 1 ticker in the sidebar.")
        st.stop()

    if not POLYGON_API_KEY:
        st.info("No news because Polygon API key is missing.")
        st.stop()

    cutoff = utc_now() - timedelta(minutes=int(minutes))

    all_frames = []
    errors = []

    for t in tickers:
        try:
            status, payload = get_polygon_news(t, POLYGON_API_KEY)

            if status == 401:
                errors.append(f"{t}: 401 Unauthorized (bad/blocked Polygon key)")
                continue

            if status == 429:
                errors.append(f"{t}: 429 Too Many Requests (Polygon rate limit). Try increasing cache TTL or reducing tickers.")
                continue

            # Polygon returns {"results": [...]}
            items = (payload or {}).get("results", []) or []
            rows = []

            for it in items:
                published = parse_published_utc(it.get("published_utc", ""))
                if not published:
                    continue
                if published < cutoff:
                    # because results are newest-first, we can break once older
                    break

                rows.append(
                    {
                        "Ticker": t,
                        "Published (UTC)": published.strftime("%Y-%m-%d %H:%M:%S"),
                        "Title": it.get("title", ""),
                        "Source": (it.get("publisher") or {}).get("name", ""),
                        "URL": it.get("article_url", ""),
                    }
                )

            if rows:
                all_frames.append(pd.DataFrame(rows))

        except Exception as e:
            errors.append(f"{t}: {e}")

    if errors:
        st.error("Some tickers failed to load news:")
        for msg in errors:
            st.write("-", msg)

    if not all_frames:
        st.info
