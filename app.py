import os
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import streamlit as st

from streamlit_autorefresh import st_autorefresh


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Flow + News (5m)", layout="wide")

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TSLA", "AMD", "NVDA"]

# Your Unusual Whales screener link (from your message)
UW_SCREENER_URL = (
    "https://unusualwhales.com/options-screener"
    "?close_greater_avg=true&exclude_ex_div_ticker=true&exclude_itm=true"
    "&issue_types[]=Common%20Stock&issue_types[]=ETF&limit=250&max_dte=7"
    "&min_diff=-0.1&min_oi_change_perc=-10&min_premium=500000"
    "&min_volume_oi_ratio=1&min_volume_ticker_vol_ratio=0.03"
    "&order=premium&order_direction=desc&watchlist_name=GPT%20Filter%20"
)

POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY", os.getenv("POLYGON_API_KEY", "")).strip()


# -----------------------------
# HELPERS
# -----------------------------
def utc_now():
    return datetime.now(timezone.utc)


@st.cache_data(ttl=60)
def get_polygon_news(ticker: str, api_key: str):
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker,
        "limit": 20,
        "order": "desc",
        "sort": "published_utc",
        "apiKey": api_key,
    }

    try:
        r = requests.get(url, params=params, timeout=20)

        if r.status_code == 429:
            st.warning("Polygon rate limit hit. Waiting before retry.")
            return []

        r.raise_for_status()
        data = r.json()
        return data.get("results", []) or []

    except Exception as e:
        st.error(f"Polygon error: {e}")
        return []



# -----------------------------
# UI
# -----------------------------
st.title("📈 Option Flow (Unusual Whales) + 🗞️ News (Polygon) — last 5 minutes")

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
        st.warning("Polygon key missing. Add it in Streamlit → Settings → Secrets (POLYGON_API_KEY).")

    st.caption("App auto-refreshes every 5 minutes.")

col1, col2 = st.columns([1.15, 1])

with col1:
    st.subheader("Unusual Whales — your screener")
    st.write("This is your exact screener link embedded below.")
    st.components.v1.iframe(UW_SCREENER_URL, height=900, scrolling=True)

with col2:
    st.subheader(f"Polygon News — last {minutes} minutes")
    st.write(f"Last update (UTC): **{utc_now().strftime('%Y-%m-%d %H:%M:%S')}**")

    if not tickers:
        st.info("Pick at least 1 ticker in the sidebar.")
    else:
        all_frames = []
        errors = []

        for t in tickers:
            try:
                items = polygon_fetch_news(t, minutes=int(minutes), limit=50)
                df = normalize_news(items, t)
                if not df.empty:
                    all_frames.append(df)
            except Exception as e:
                errors.append(f"{t}: {e}")

        if errors:
            st.error("Some tickers failed to load news:")
            for msg in errors:
                st.write("-", msg)

        if not all_frames:
            st.info("No news in the last window (or Polygon key missing).")
        else:
            news_df = pd.concat(all_frames, ignore_index=True)
            # Make URLs clickable
            st.dataframe(
                news_df,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()
            st.subheader("Clickable links")
            for _, row in news_df.iterrows():
                title = row["Title"] or "(no title)"
                url = row["URL"] or ""
                if url:
                    st.markdown(f"- **{row['Ticker']}** — [{title}]({url})")
