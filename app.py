 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/app.py b/app.py
index 7554aebd2fecfbe2e856d6dd17cb230f77b66c0a..f2942ccdaeb176bbbbbfd2e394bbbac2094d6036 100644
--- a/app.py
+++ b/app.py
@@ -1,797 +1,540 @@
-import os
 import math
-import time
 from datetime import datetime, timedelta
+from io import StringIO
 from zoneinfo import ZoneInfo
 
-import requests
 import pandas as pd
+import requests
 import streamlit as st
 from streamlit_autorefresh import st_autorefresh
 
 
-# ============================================================
-# CONFIG
-# ============================================================
-st.set_page_config(page_title="Institutional Options Signals (5m) — CALLS/PUTS ONLY", layout="wide")
+st.set_page_config(page_title="Institutional Options Signals", layout="wide")
+
+TZ = ZoneInfo("America/Chicago")
 
-TZ = ZoneInfo("America/Chicago")  # CST/CDT auto-handled
-NOW_CST = lambda: datetime.now(tz=TZ)
 
-DEFAULT_QUICK = ["SPY", "QQQ", "IWM", "DIA", "TSLA", "NVDA", "AMD"]
+def now_cst() -> datetime:
+    return datetime.now(tz=TZ)
 
-# Unusual Whales screener URL (web view) — your rules:
-# $1M premium min, DTE<=3, stocks+ETF only, volume>OI, exclude ITM
-UW_SCREENER_URL = (
-    "https://unusualwhales.com/options-screener"
-    "?exclude_itm=true"
-    "&issue_types[]=Common%20Stock&issue_types[]=ETF"
-    "&min_premium=1000000"
-    "&max_dte=3"
-    "&min_volume_oi_ratio=1"
-    "&order=premium&order_direction=desc"
-)
 
-# ============================================================
-# SECRETS
-# ============================================================
 def get_secret(name: str, default: str = "") -> str:
     try:
-        v = st.secrets.get(name, default)
+        value = st.secrets.get(name, default)
     except Exception:
-        v = default
-    if v is None:
-        v = default
-    return str(v).strip()
+        value = default
+    return str(value or "").strip()
+
 
 EODHD_API_KEY = get_secret("EODHD_API_KEY")
-UW_TOKEN = get_secret("UW_TOKEN")  # Bearer token
-UW_FLOW_ALERTS_URL = get_secret("UW_FLOW_ALERTS_URL")  # must be FULL URL
-FINVIZ_AUTH = get_secret("FINVIZ_AUTH")  # optional (not used here)
-
-
-# ============================================================
-# SMALL HELPERS
-# ============================================================
-def safe_upper_ticker(t: str) -> str:
-    t = (t or "").strip().upper()
-    # allow letters, numbers, dot, dash (for some tickers)
-    cleaned = "".join(ch for ch in t if ch.isalnum() or ch in [".", "-"])
-    return cleaned
-
-def as_eodhd_symbol(ticker: str) -> str:
-    """
-    EODHD uses e.g. AAPL.US
-    If user already typed a suffix (like .US), keep it.
-    """
-    t = safe_upper_ticker(ticker)
+UW_TOKEN = get_secret("UW_TOKEN")
+UW_FLOW_ALERTS_URL = get_secret("UW_FLOW_ALERTS_URL")
+
+
+def clean_ticker(ticker: str) -> str:
+    ticker = (ticker or "").upper().strip()
+    return "".join(ch for ch in ticker if ch.isalnum() or ch in [".", "-"])
+
+
+def to_eodhd_symbol(ticker: str) -> str:
+    t = clean_ticker(ticker)
     if not t:
         return ""
-    if "." in t:
-        return t
-    return f"{t}.US"
+    return t if "." in t else f"{t}.US"
 
-def clamp(x, lo, hi):
-    return max(lo, min(hi, x))
 
-def http_get(url, params=None, headers=None, timeout=20):
-    return requests.get(url, params=params, headers=headers, timeout=timeout)
+def clamp(value: float, lo: float, hi: float) -> float:
+    return max(lo, min(hi, value))
 
-def fmt_num(x, digits=2):
+
+def fmt(value, digits: int = 2) -> str:
     try:
-        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
+        f = float(value)
+        if math.isnan(f) or math.isinf(f):
             return "N/A"
-        return f"{float(x):.{digits}f}"
+        return f"{f:.{digits}f}"
     except Exception:
         return "N/A"
 
 
-# ============================================================
-# TECH INDICATORS (pandas)
-# ============================================================
+def safe_get(url: str, params=None, headers=None, timeout: int = 20):
+    return requests.get(url, params=params, headers=headers, timeout=timeout)
+
+
 def rsi(series: pd.Series, period: int = 14) -> pd.Series:
     delta = series.diff()
-    up = delta.clip(lower=0.0)
-    down = (-delta).clip(lower=0.0)
-    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
-    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
-    rs = roll_up / roll_down.replace(0, pd.NA)
-    out = 100 - (100 / (1 + rs))
-    return out.fillna(method="bfill")
-
-def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
+    gains = delta.clip(lower=0)
+    losses = (-delta).clip(lower=0)
+    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
+    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
+    rs = avg_gain / avg_loss.replace(0, pd.NA)
+    return (100 - (100 / (1 + rs))).bfill()
+
+
+def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
     ema_fast = close.ewm(span=fast, adjust=False).mean()
     ema_slow = close.ewm(span=slow, adjust=False).mean()
-    macd_line = ema_fast - ema_slow
-    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
-    hist = macd_line - signal_line
-    return hist
-
-def vwap(df: pd.DataFrame) -> pd.Series:
-    # VWAP = cumulative(sum(price*volume)) / cumulative(sum(volume))
-    # use typical price (H+L+C)/3 if available, else close
-    if all(c in df.columns for c in ["high", "low", "close", "volume"]):
-        tp = (df["high"] + df["low"] + df["close"]) / 3.0
-        vol = df["volume"].replace(0, pd.NA)
-        cum_pv = (tp * vol).cumsum()
-        cum_v = vol.cumsum()
-        out = cum_pv / cum_v
-        return out.fillna(method="bfill")
-    elif "close" in df.columns and "volume" in df.columns:
-        vol = df["volume"].replace(0, pd.NA)
-        cum_pv = (df["close"] * vol).cumsum()
-        cum_v = vol.cumsum()
-        out = cum_pv / cum_v
-        return out.fillna(method="bfill")
-    else:
+    line = ema_fast - ema_slow
+    sig = line.ewm(span=signal, adjust=False).mean()
+    return line - sig
+
+
+def calc_vwap(df: pd.DataFrame) -> pd.Series:
+    if not all(col in df.columns for col in ["high", "low", "close", "volume"]):
         return pd.Series([pd.NA] * len(df), index=df.index)
+    typical = (df["high"] + df["low"] + df["close"]) / 3
+    vol = df["volume"].replace(0, pd.NA)
+    return ((typical * vol).cumsum() / vol.cumsum()).bfill()
 
 
-# ============================================================
-# DATA SOURCES
-# ============================================================
 @st.cache_data(ttl=60)
 def eodhd_intraday(symbol: str, interval: str = "5m", lookback_minutes: int = 240):
-    """
-    Returns dataframe with columns: datetime, open, high, low, close, volume
-    """
     if not EODHD_API_KEY:
         return pd.DataFrame(), "missing_key"
-
     url = f"https://eodhd.com/api/intraday/{symbol}"
-    # request a window slightly bigger than needed
-    params = {
-        "api_token": EODHD_API_KEY,
-        "interval": interval,
-        "fmt": "json",
-    }
-
+    params = {"api_token": EODHD_API_KEY, "interval": interval, "fmt": "json"}
     try:
-        r = http_get(url, params=params, timeout=20)
+        r = safe_get(url, params=params)
         if r.status_code == 429:
             return pd.DataFrame(), "rate_limited"
         r.raise_for_status()
         data = r.json()
-        if not isinstance(data, list) or len(data) == 0:
+        if not isinstance(data, list) or not data:
             return pd.DataFrame(), "empty"
-
         df = pd.DataFrame(data)
-        # EODHD returns "datetime" like "2026-02-18 12:55:00"
-        if "datetime" in df.columns:
-            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
-            df = df.dropna(subset=["datetime"]).sort_values("datetime")
-        else:
+        if "datetime" not in df.columns:
             return pd.DataFrame(), "bad_format"
+        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
+        df = df.dropna(subset=["datetime"]).sort_values("datetime")
+        cutoff = pd.Timestamp(now_cst() - timedelta(minutes=lookback_minutes)).tz_localize(None)
+        df = df[df["datetime"] >= cutoff]
+        for col in ["open", "high", "low", "close", "volume"]:
+            if col in df.columns:
+                df[col] = pd.to_numeric(df[col], errors="coerce")
+        return df.dropna(subset=["close"]), "ok"
+    except requests.HTTPError:
+        return pd.DataFrame(), f"http_{r.status_code}"
+    except Exception as ex:
+        return pd.DataFrame(), f"error:{type(ex).__name__}"
 
-        # keep only last N minutes if possible
-        if lookback_minutes and "datetime" in df.columns:
-            cutoff = pd.Timestamp(NOW_CST() - timedelta(minutes=int(lookback_minutes))).tz_localize(None)
-            df = df[df["datetime"] >= cutoff]
-
-        # normalize numeric columns
-        for c in ["open", "high", "low", "close", "volume"]:
-            if c in df.columns:
-                df[c] = pd.to_numeric(df[c], errors="coerce")
-
-        df = df.dropna(subset=["close"])
-        return df, "ok"
-    except Exception as e:
-        return pd.DataFrame(), f"error: {type(e).__name__}"
 
 @st.cache_data(ttl=120)
 def eodhd_news(symbol: str, lookback_minutes: int = 60, limit: int = 20):
-    """
-    EODHD News endpoint:
-    https://eodhd.com/api/news?s=AAPL.US&from=YYYY-MM-DD&to=YYYY-MM-DD&api_token=...&fmt=json
-    """
     if not EODHD_API_KEY:
         return [], "missing_key"
-
+    now = now_cst()
     url = "https://eodhd.com/api/news"
-    now = NOW_CST()
-    start = now - timedelta(minutes=int(lookback_minutes))
     params = {
         "s": symbol,
-        "from": start.strftime("%Y-%m-%d"),
+        "from": (now - timedelta(minutes=lookback_minutes)).strftime("%Y-%m-%d"),
         "to": now.strftime("%Y-%m-%d"),
-        "limit": int(limit),
+        "limit": limit,
         "api_token": EODHD_API_KEY,
         "fmt": "json",
     }
     try:
-        r = http_get(url, params=params, timeout=20)
+        r = safe_get(url, params=params)
         if r.status_code == 429:
             return [], "rate_limited"
         r.raise_for_status()
-        data = r.json()
-        if not isinstance(data, list):
+        payload = r.json()
+        if not isinstance(payload, list):
             return [], "bad_format"
-
-        # Filter to last lookback_minutes using published datetime if present
+        cutoff = now.replace(tzinfo=None) - timedelta(minutes=lookback_minutes)
         items = []
-        for it in data:
-            published = it.get("date") or it.get("datetime") or it.get("published")
-            # EODHD uses "date": "2026-02-18 11:20:00"
-            dt = None
-            if published:
-                dt = pd.to_datetime(published, errors="coerce")
-            if dt is not None and not pd.isna(dt):
-                # treat as CST naive
-                if dt.to_pydatetime() >= (now.replace(tzinfo=None) - timedelta(minutes=int(lookback_minutes))):
-                    items.append(it)
-            else:
-                items.append(it)
-
+        for item in payload:
+            raw_dt = item.get("date") or item.get("datetime") or item.get("published")
+            dt = pd.to_datetime(raw_dt, errors="coerce")
+            if pd.isna(dt) or dt.to_pydatetime() >= cutoff:
+                items.append(item)
         return items, "ok"
-    except Exception as e:
-        return [], f"error: {type(e).__name__}"
-
-def simple_news_sentiment(headline: str) -> int:
-    """
-    Very simple sentiment: +1 / 0 / -1
-    (You can later replace with a better model—this is stable + fast.)
-    """
-    if not headline:
-        return 0
-    h = headline.lower()
-    pos = ["beats", "surge", "rally", "up", "wins", "strong", "record", "bull", "growth", "upgrade"]
-    neg = ["miss", "plunge", "down", "weak", "fraud", "lawsuit", "bear", "cut", "downgrade", "probe"]
-    score = 0
-    for w in pos:
-        if w in h:
-            score += 1
-    for w in neg:
-        if w in h:
-            score -= 1
-    return 1 if score > 0 else (-1 if score < 0 else 0)
+    except requests.HTTPError:
+        return [], f"http_{r.status_code}"
+    except Exception as ex:
+        return [], f"error:{type(ex).__name__}"
+
 
 @st.cache_data(ttl=120)
 def uw_options_volume_bias(ticker: str):
-    """
-    Unusual Whales:
-    GET https://api.unusualwhales.com/api/stock/{ticker}/options-volume
-    Header: Authorization: Bearer <token>
-    """
     if not UW_TOKEN:
         return None, "missing_key"
-
-    url = f"https://api.unusualwhales.com/api/stock/{safe_upper_ticker(ticker)}/options-volume"
-    headers = {"Accept": "application/json, text/plain", "Authorization": f"Bearer {UW_TOKEN}"}
+    url = f"https://api.unusualwhales.com/api/stock/{clean_ticker(ticker)}/options-volume"
+    headers = {"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json"}
     try:
-        r = http_get(url, headers=headers, timeout=20)
+        r = safe_get(url, headers=headers)
         if r.status_code == 429:
             return None, "rate_limited"
         r.raise_for_status()
-        js = r.json()
-        data = js.get("data", [])
-        if not data:
+        payload = r.json()
+        rows = payload.get("data") if isinstance(payload, dict) else None
+        if not rows:
             return None, "empty"
-        latest = data[0]  # API usually returns a list, newest first
-        # bias
-        bullish_prem = float(latest.get("bullish_premium", 0) or 0)
-        bearish_prem = float(latest.get("bearish_premium", 0) or 0)
+        latest = rows[0]
+        bull = float(latest.get("bullish_premium", 0) or 0)
+        bear = float(latest.get("bearish_premium", 0) or 0)
         call_vol = float(latest.get("call_volume", 0) or 0)
         put_vol = float(latest.get("put_volume", 0) or 0)
         call_oi = float(latest.get("call_open_interest", 0) or 0)
         put_oi = float(latest.get("put_open_interest", 0) or 0)
 
-        # "Gamma bias" proxy (not true GEX): OI imbalance
-        gamma_proxy = (call_oi - put_oi)
+        bias = "Neutral"
+        if bull > bear and call_vol >= put_vol:
+            bias = "Bullish"
+        elif bear > bull and put_vol >= call_vol:
+            bias = "Bearish"
+
+        gamma_proxy = call_oi - put_oi
         gamma_bias = "Neutral"
         if gamma_proxy > 0:
             gamma_bias = "Positive Gamma (proxy)"
         elif gamma_proxy < 0:
             gamma_bias = "Negative Gamma (proxy)"
 
-        uw_bias = "Neutral"
-        if bullish_prem > bearish_prem and call_vol >= put_vol:
-            uw_bias = "Bullish"
-        elif bearish_prem > bullish_prem and put_vol >= call_vol:
-            uw_bias = "Bearish"
-
         return {
-            "uw_bias": uw_bias,
-            "bullish_premium": bullish_prem,
-            "bearish_premium": bearish_prem,
-            "call_vol": call_vol,
-            "put_vol": put_vol,
-            "gamma_bias": gamma_bias
+            "bias": bias,
+            "bull": bull,
+            "bear": bear,
+            "gamma_bias": gamma_bias,
         }, "ok"
     except requests.HTTPError:
         return None, f"http_{r.status_code}"
-    except Exception as e:
-        return None, f"error: {type(e).__name__}"
+    except Exception as ex:
+        return None, f"error:{type(ex).__name__}"
+
 
-@st.cache_data(ttl=30)
+@st.cache_data(ttl=45)
 def uw_flow_alerts(ticker: str, limit: int = 50):
-    """
-    Uses YOUR EXACT UW_FLOW_ALERTS_URL from Secrets.
-    If that endpoint is wrong / plan-restricted, we show it as RED but we do not crash.
-    """
     if not UW_TOKEN:
         return [], "missing_key"
-
     if not UW_FLOW_ALERTS_URL:
         return [], "missing_url"
-
-    url = UW_FLOW_ALERTS_URL.strip()  # do NOT modify (no underscore/dash changes)
-    headers = {"Accept": "application/json, text/plain", "Authorization": f"Bearer {UW_TOKEN}"}
-
-    # params are best-effort; if UW ignores them, fine.
+    headers = {"Authorization": f"Bearer {UW_TOKEN}", "Accept": "application/json"}
     params = {
-        "limit": int(limit),
-        "ticker": safe_upper_ticker(ticker),
+        "ticker": clean_ticker(ticker),
+        "limit": limit,
         "min_premium": 1000000,
         "max_dte": 3,
-        "exclude_itm": True,
-        "volume_gt_oi": True,
-        "order": "desc",
+        "exclude_itm": "true",
+        "volume_gt_oi": "true",
     }
-
     try:
-        r = http_get(url, params=params, headers=headers, timeout=20)
+        r = safe_get(UW_FLOW_ALERTS_URL, params=params, headers=headers)
         if r.status_code == 429:
             return [], "rate_limited"
         if r.status_code == 404:
-            # Plan/route issue is common here
             return [], "http_404"
         r.raise_for_status()
-        js = r.json()
-
-        # UW sometimes returns {"data":[...]} or a raw list. Handle both.
-        if isinstance(js, dict) and "data" in js:
-            items = js.get("data") or []
-        elif isinstance(js, list):
-            items = js
-        else:
-            items = []
-
-        # ticker-filter again just in case
-        out = []
-        for it in items:
-            sym = (it.get("underlying_symbol") or it.get("ticker") or "").upper()
-            if sym == safe_upper_ticker(ticker):
-                out.append(it)
-
-        return out, "ok"
+        payload = r.json()
+        rows = payload.get("data") if isinstance(payload, dict) else payload
+        if not isinstance(rows, list):
+            rows = []
+        ticker_clean = clean_ticker(ticker)
+        filtered = [
+            row
+            for row in rows
+            if str(row.get("underlying_symbol") or row.get("ticker") or "").upper() == ticker_clean
+        ]
+        return filtered, "ok"
     except requests.HTTPError:
-        return [], f"http_{getattr(r, 'status_code', 'ERR')}"
-    except Exception as e:
-        return [], f"error: {type(e).__name__}"
+        return [], f"http_{r.status_code}"
+    except Exception as ex:
+        return [], f"error:{type(ex).__name__}"
+
 
 @st.cache_data(ttl=300)
 def ten_year_yield_optional():
-    """
-    Optional 10Y source (no key): Stooq CSV.
-    If it fails, return None (still stable).
-    """
     try:
-        # Stooq US10Y symbol varies; this works often:
-        # daily: https://stooq.com/q/d/l/?s=us10y&i=d
-        url = "https://stooq.com/q/d/l/?s=us10y&i=d"
-        r = http_get(url, timeout=20)
+        r = safe_get("https://stooq.com/q/d/l/?s=us10y&i=d")
         r.raise_for_status()
-        df = pd.read_csv(pd.compat.StringIO(r.text))  # may fail on some envs
-        if df.empty:
+        df = pd.read_csv(StringIO(r.text))
+        if df.empty or "Close" not in df.columns or len(df) < 2:
             return None, "empty"
-        # last two closes
-        df = df.dropna()
+        df = df.dropna(subset=["Close"])
         if len(df) < 2:
-            return None, "too_short"
+            return None, "empty"
         last = float(df["Close"].iloc[-1])
         prev = float(df["Close"].iloc[-2])
-        chg = last - prev
-        return {"last": last, "chg": chg}, "ok"
+        return {"last": last, "chg": last - prev}, "ok"
     except Exception:
-        # keep it quiet and stable
         return None, "not_available"
 
 
-# ============================================================
-# SCORING
-# ============================================================
-def score_ticker(df_bars: pd.DataFrame, news_items: list, uw_bias_obj: dict | None, ten_y: dict | None,
-                 w_rsi: float, w_macd: float, w_vwap: float, w_ema: float, w_vol: float, w_uw: float, w_news: float, w_10y: float):
-    """
-    Returns:
-      score_0_100, direction, signal (BUY CALLS/BUY PUTS/WAIT), components dict
-    """
-    # Default neutral
+def headline_sentiment(title: str) -> int:
+    pos = ["beats", "surge", "rally", "wins", "strong", "upgrade", "growth", "record"]
+    neg = ["miss", "plunge", "weak", "downgrade", "lawsuit", "fraud", "probe", "cut"]
+    t = (title or "").lower()
+    score = sum(1 for w in pos if w in t) - sum(1 for w in neg if w in t)
+    return 1 if score > 0 else (-1 if score < 0 else 0)
+
+
+def score_signal(df: pd.DataFrame, news_items: list, uw_bias: dict | None, ten_y: dict | None, weights: dict):
     comp = {
-        "RSI": None,
-        "MACD_hist": None,
-        "VWAP": None,
-        "EMA_stack": None,
-        "Vol_ratio": None,
-        "IV_spike": None,     # optional (requires alerts with IV)
-        "Gamma_bias": "N/A",
-        "UW_bias": "Neutral",
-        "UW_unusual": "NO",
-        "10Y": "N/A",
-        "News_sent": 0
+        "rsi": None,
+        "macd_hist": None,
+        "vwap": None,
+        "ema_stack": "N/A",
+        "vol_ratio": None,
+        "uw_bias": "Neutral",
+        "gamma_bias": "N/A",
+        "news": 0,
+        "teny": "N/A",
     }
+    sig = []
 
-    # ---- News sentiment
     if news_items:
-        s = 0
-        for it in news_items[:20]:
-            title = it.get("title") or it.get("Title") or ""
-            s += simple_news_sentiment(title)
-        # clamp to -3..+3
-        comp["News_sent"] = int(clamp(s, -3, 3))
-    else:
-        comp["News_sent"] = 0
+        news_score = sum(headline_sentiment(item.get("title") or item.get("Title") or "") for item in news_items[:20])
+        comp["news"] = int(clamp(news_score, -3, 3))
 
-    # ---- UW bias + Gamma proxy
-    if uw_bias_obj:
-        comp["UW_bias"] = uw_bias_obj.get("uw_bias", "Neutral")
-        comp["Gamma_bias"] = uw_bias_obj.get("gamma_bias", "N/A")
+    if uw_bias:
+        comp["uw_bias"] = uw_bias.get("bias", "Neutral")
+        comp["gamma_bias"] = uw_bias.get("gamma_bias", "N/A")
 
-    # ---- Indicators (if we have bars)
-    if df_bars is not None and not df_bars.empty and "close" in df_bars.columns:
-        df = df_bars.copy()
+    if df is not None and not df.empty and "close" in df.columns:
         close = df["close"].astype(float)
+        comp["rsi"] = float(rsi(close).iloc[-1])
+        comp["macd_hist"] = float(macd_hist(close).iloc[-1])
 
-        # RSI
-        rsi14 = rsi(close, 14)
-        comp["RSI"] = float(rsi14.iloc[-1])
-
-        # MACD hist
-        mh = macd_hist(close)
-        comp["MACD_hist"] = float(mh.iloc[-1])
-
-        # EMA stack (9/20/50)
         ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
         ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
         ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
-        # bullish if 9>20>50; bearish if 9<20<50
         if ema9 > ema20 > ema50:
-            comp["EMA_stack"] = "Bullish"
+            comp["ema_stack"] = "Bullish"
         elif ema9 < ema20 < ema50:
-            comp["EMA_stack"] = "Bearish"
-        else:
-            comp["EMA_stack"] = "Neutral"
-
-        # VWAP
-        vw = vwap(df)
-        vwap_last = vw.iloc[-1] if len(vw) else pd.NA
-        comp["VWAP"] = float(vwap_last) if pd.notna(vwap_last) else None
-
-        # Volume ratio = last volume / avg volume (last 30 bars)
-        if "volume" in df.columns and df["volume"].notna().any():
-            vol = df["volume"].fillna(0).astype(float)
-            base = vol.tail(30).mean() if len(vol) >= 10 else vol.mean()
-            comp["Vol_ratio"] = float(vol.iloc[-1] / base) if base and base > 0 else None
-        else:
-            comp["Vol_ratio"] = None
-    else:
-        # keep N/A
-        pass
-
-    # ---- 10Y
-    if ten_y and isinstance(ten_y, dict):
-        comp["10Y"] = f"{fmt_num(ten_y.get('last'), 2)} ({fmt_num(ten_y.get('chg'), 2)})"
-
-    # ========================================================
-    # Convert components to normalized signals in [-1, +1]
-    # ========================================================
-    sigs = []
-
-    # RSI: bullish if rising from oversold-ish, bearish if overbought-ish
-    if comp["RSI"] is not None:
-        r = comp["RSI"]
-        if r <= 35:
-            sigs.append(("rsi", +0.6))
-        elif r >= 65:
-            sigs.append(("rsi", -0.6))
+            comp["ema_stack"] = "Bearish"
         else:
-            sigs.append(("rsi", 0.0))
-
-    # MACD hist: positive bullish, negative bearish (scaled)
-    if comp["MACD_hist"] is not None:
-        m = comp["MACD_hist"]
-        sigs.append(("macd", clamp(m * 10.0, -1.0, 1.0)))  # scale small numbers
-
-    # VWAP: price above vwap bullish, below bearish
-    if df_bars is not None and not df_bars.empty and comp["VWAP"] is not None:
-        last_px = float(df_bars["close"].iloc[-1])
-        vw = float(comp["VWAP"])
-        sigs.append(("vwap", +0.7 if last_px > vw else (-0.7 if last_px < vw else 0.0)))
-
-    # EMA stack
-    if comp["EMA_stack"] == "Bullish":
-        sigs.append(("ema", +0.7))
-    elif comp["EMA_stack"] == "Bearish":
-        sigs.append(("ema", -0.7))
-    else:
-        sigs.append(("ema", 0.0))
-
-    # Volume ratio: spike favors continuation (direction decided by trend proxies)
-    if comp["Vol_ratio"] is not None:
-        vr = comp["Vol_ratio"]
-        # cap: if >2 = strong
-        sigs.append(("vol", clamp((vr - 1.0) / 1.5, -1.0, 1.0)))
-
-    # UW bias
-    if comp["UW_bias"] == "Bullish":
-        sigs.append(("uw", +0.8))
-    elif comp["UW_bias"] == "Bearish":
-        sigs.append(("uw", -0.8))
-    else:
-        sigs.append(("uw", 0.0))
-
-    # News sentiment
-    ns = comp["News_sent"]
-    sigs.append(("news", clamp(ns / 3.0, -1.0, 1.0)))
-
-    # 10Y: rising yields = mild bearish; falling yields = mild bullish
-    if ten_y and isinstance(ten_y, dict) and ten_y.get("chg") is not None:
+            comp["ema_stack"] = "Neutral"
+
+        vw = calc_vwap(df)
+        last_vwap = vw.iloc[-1] if len(vw) else pd.NA
+        if pd.notna(last_vwap):
+            comp["vwap"] = float(last_vwap)
+
+        if "volume" in df.columns:
+            vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
+            baseline = vol.tail(30).mean() if len(vol) >= 10 else vol.mean()
+            comp["vol_ratio"] = float(vol.iloc[-1] / baseline) if baseline > 0 else None
+
+        if comp["rsi"] is not None:
+            sig.append(("rsi", 0.6 if comp["rsi"] <= 35 else (-0.6 if comp["rsi"] >= 65 else 0.0)))
+        sig.append(("macd", clamp((comp["macd_hist"] or 0) * 10, -1, 1)))
+        if comp["vwap"] is not None:
+            last_px = float(close.iloc[-1])
+            sig.append(("vwap", 0.7 if last_px > comp["vwap"] else (-0.7 if last_px < comp["vwap"] else 0.0)))
+        sig.append(("ema", 0.7 if comp["ema_stack"] == "Bullish" else (-0.7 if comp["ema_stack"] == "Bearish" else 0.0)))
+        if comp["vol_ratio"] is not None:
+            sig.append(("vol", clamp((comp["vol_ratio"] - 1.0) / 1.5, -1, 1)))
+
+    sig.append(("uw", 0.8 if comp["uw_bias"] == "Bullish" else (-0.8 if comp["uw_bias"] == "Bearish" else 0.0)))
+    sig.append(("news", clamp(comp["news"] / 3, -1, 1)))
+
+    if ten_y and ten_y.get("chg") is not None:
         chg = float(ten_y["chg"])
-        sigs.append(("10y", clamp(-chg * 0.5, -0.5, 0.5)))  # mild
-
-    # ========================================================
-    # Weighted sum
-    # ========================================================
-    weights = {
-        "rsi": w_rsi,
-        "macd": w_macd,
-        "vwap": w_vwap,
-        "ema": w_ema,
-        "vol": w_vol,
-        "uw": w_uw,
-        "news": w_news,
-        "10y": w_10y,
-    }
-    total_w = sum(max(0.0, v) for v in weights.values()) or 1.0
-
-    agg = 0.0
-    for name, val in sigs:
-        agg += (weights.get(name, 0.0) * float(val))
-    agg_norm = agg / total_w
-    agg_norm = clamp(agg_norm, -1.0, 1.0)
-
-    confidence = int(round(50 + 50 * abs(agg_norm)))
-    direction = "BULLISH" if agg_norm > 0.12 else ("BEARISH" if agg_norm < -0.12 else "NEUTRAL")
-
-    # CALLS/PUTS only
-    if direction == "BULLISH":
-        signal = "BUY CALLS" if confidence >= 60 else "WAIT"
-    elif direction == "BEARISH":
-        signal = "BUY PUTS" if confidence >= 60 else "WAIT"
-    else:
-        signal = "WAIT"
-
-    return confidence, direction, signal, comp
-
+        comp["teny"] = f"{fmt(ten_y.get('last'))} ({fmt(chg)})"
+        sig.append(("teny", clamp(-chg * 0.5, -0.5, 0.5)))
 
-# ============================================================
-# UI — SIDEBAR
-# ============================================================
-st.title("🏛️ Institutional Options Signals (5m) — CALLS / PUTS ONLY")
-st.caption(f"Last update (CST): {NOW_CST().strftime('%Y-%m-%d %H:%M:%S %Z')}")
+    total_weight = sum(max(0.0, float(v)) for v in weights.values()) or 1.0
+    aggregate = sum(weights.get(name, 0.0) * float(value) for name, value in sig)
+    normalized = clamp(aggregate / total_weight, -1, 1)
 
-# refresh
-with st.sidebar:
-    st.header("Settings")
+    confidence = int(round(clamp(50 + 50 * abs(normalized), 0, 100)))
+    direction = "BULLISH" if normalized > 0.12 else ("BEARISH" if normalized < -0.12 else "NEUTRAL")
+    signal = "BUY CALLS" if direction == "BULLISH" else ("BUY PUTS" if direction == "BEARISH" else "WAIT")
+    return confidence, direction, signal, comp
 
-    # 1) Let user type ANY ticker
-    tickers_text = st.text_input(
-        "Type tickers (comma-separated). Example: SPY,TSLA,NVDA",
-        value="SPY,TSLA",
-        help="You can type ANY ticker here. We auto-uppercase it."
-    )
 
-    # 2) Optional quick pick list (just convenience)
-    quick_pick = st.multiselect("Quick pick (optional)", DEFAULT_QUICK, default=[])
+def endpoint_emoji(ok: bool) -> str:
+    return "🟢" if ok else "🔴"
 
-    # Combine & dedupe
-    typed = [safe_upper_ticker(x) for x in tickers_text.split(",")]
-    typed = [x for x in typed if x]
-    combined = []
-    for t in typed + quick_pick:
-        if t and t not in combined:
-            combined.append(t)
-    tickers = combined[:25]  # safety cap
 
-    st.divider()
-    news_lookback = st.number_input("News lookback (minutes)", 1, 360, 60, 1)
-    price_lookback = st.number_input("Price lookback (minutes)", 30, 780, 240, 5)
+st.title("🏛️ Institutional Options Signals (CALLS / PUTS ONLY)")
+st.caption(f"Last update (CST): {now_cst().strftime('%Y-%m-%d %H:%M:%S %Z')}")
 
-    st.divider()
-    st.subheader("Refresh")
+with st.sidebar:
+    st.header("Settings")
+    raw_tickers = st.text_input("Tickers (comma separated)", value="SPY,TSLA", help="Type ANY ticker symbols.")
     refresh_sec = st.slider("Auto-refresh (seconds)", 10, 300, 30, 5)
-    st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh")
-
-    st.divider()
-    st.subheader("Institutional mode")
-    institutional_min = st.slider("Signals only if confidence ≥", 50, 95, 75, 1)
-
-    st.divider()
-    st.subheader("Weights (sum doesn't have to be 1)")
-
-    w_rsi = st.slider("RSI weight", 0.0, 0.40, 0.15, 0.01)
-    w_macd = st.slider("MACD weight", 0.0, 0.40, 0.15, 0.01)
-    w_vwap = st.slider("VWAP weight", 0.0, 0.40, 0.15, 0.01)
-    w_ema = st.slider("EMA stack (9/20/50) weight", 0.0, 0.40, 0.18, 0.01)
-    w_vol = st.slider("Volume ratio weight", 0.0, 0.40, 0.12, 0.01)
-    w_uw = st.slider("UW flow/bias weight", 0.0, 0.60, 0.20, 0.01)
-    w_news = st.slider("News weight", 0.0, 0.30, 0.05, 0.01)
-    w_10y = st.slider("10Y yield weight (optional)", 0.0, 0.20, 0.05, 0.01)
-
-    st.divider()
-    st.subheader("Keys status (green/red)")
-    st.success("EODHD_API_KEY ✅" if EODHD_API_KEY else "EODHD_API_KEY ❌")
-    st.success("UW_TOKEN (Bearer) ✅" if UW_TOKEN else "UW_TOKEN ❌")
-    st.success("UW_FLOW_ALERTS_URL ✅" if UW_FLOW_ALERTS_URL else "UW_FLOW_ALERTS_URL ❌")
-    if FINVIZ_AUTH:
-        st.info("FINVIZ_AUTH present (not used in this build)")
-
-# ============================================================
-# LAYOUT
-# ============================================================
-left, right = st.columns([1.35, 1.0])
-
-with left:
-    st.subheader("Unusual Whales Screener (web view)")
-    st.caption("This is embedded. True filtering (DTE/ITM/premium rules) is best done inside the screener itself.")
-    st.components.v1.iframe(UW_SCREENER_URL, height=820, scrolling=True)
-
-# ============================================================
-# RIGHT PANEL: DATA + SIGNALS
-# ============================================================
-with right:
-    st.subheader("Live Score / Signals (EODHD price + EODHD headlines + UW bias)")
-    st.caption(f"Last update (CST): {NOW_CST().strftime('%Y-%m-%d %H:%M:%S %Z')}")
-
-    if not tickers:
-        st.warning("Type at least 1 ticker in the sidebar (comma-separated).")
-        st.stop()
-
-    # 10Y optional (does not break app if unavailable)
-    teny, teny_status = ten_year_yield_optional()
-
-    rows = []
-    endpoint_notes = {
-        "eodhd_intraday": [],
-        "eodhd_news": [],
-        "uw_options_volume": [],
-        "uw_flow_alerts": []
+    institutional_min = st.slider("Institutional mode minimum confidence", 50, 95, 75, 1)
+    news_lookback = st.number_input("News lookback (minutes)", min_value=1, max_value=360, value=60, step=1)
+    price_lookback = st.number_input("Price lookback (minutes)", min_value=30, max_value=780, value=240, step=5)
+
+    st.subheader("Weights")
+    w_rsi = st.slider("RSI", 0.0, 0.40, 0.15, 0.01)
+    w_macd = st.slider("MACD hist", 0.0, 0.40, 0.15, 0.01)
+    w_vwap = st.slider("VWAP", 0.0, 0.40, 0.15, 0.01)
+    w_ema = st.slider("EMA 9/20/50", 0.0, 0.40, 0.18, 0.01)
+    w_vol = st.slider("Volume ratio", 0.0, 0.40, 0.12, 0.01)
+    w_uw = st.slider("UW bias/flow", 0.0, 0.60, 0.20, 0.01)
+    w_news = st.slider("News", 0.0, 0.30, 0.05, 0.01)
+    w_teny = st.slider("10Y yield (optional)", 0.0, 0.20, 0.05, 0.01)
+
+st_autorefresh(interval=refresh_sec * 1000, key="refresh")
+
+tickers = []
+for t in raw_tickers.split(","):
+    c = clean_ticker(t)
+    if c and c not in tickers:
+        tickers.append(c)
+tickers = tickers[:30]
+
+if not tickers:
+    st.warning("Please enter at least one ticker.")
+    st.stop()
+
+weights = {
+    "rsi": w_rsi,
+    "macd": w_macd,
+    "vwap": w_vwap,
+    "ema": w_ema,
+    "vol": w_vol,
+    "uw": w_uw,
+    "news": w_news,
+    "teny": w_teny,
+}
+
+teny, ten_status = ten_year_yield_optional()
+
+if "last_known" not in st.session_state:
+    st.session_state["last_known"] = {}
+
+status_rows = []
+out_rows = []
+alerts_rows = []
+all_news_rows = []
+
+for ticker in tickers:
+    symbol = to_eodhd_symbol(ticker)  # force .US for EODHD
+    bars, bars_status = eodhd_intraday(symbol, interval="5m", lookback_minutes=int(price_lookback))
+    news_items, news_status = eodhd_news(symbol, lookback_minutes=int(news_lookback), limit=25)
+    uw_bias, uw_status = uw_options_volume_bias(ticker)
+    flow_items, flow_status = uw_flow_alerts(ticker, limit=50)
+
+    unusual = "NO"
+    flow_bias = "Neutral"
+    if flow_items:
+        for item in flow_items[:50]:
+            premium = float(item.get("premium") or 0)
+            option_type = str(item.get("option_type") or item.get("type") or "").lower()
+            tags = [str(x).lower() for x in (item.get("tags") or [])]
+            if premium >= 1_000_000:
+                unusual = "YES"
+                if any("bearish" in tag for tag in tags) or option_type == "put":
+                    flow_bias = "Bearish"
+                elif any("bullish" in tag for tag in tags) or option_type == "call":
+                    flow_bias = "Bullish"
+                break
+
+    confidence, direction, signal, comp = score_signal(bars, news_items, uw_bias, teny, weights)
+    if unusual == "YES" and flow_bias != "Neutral":
+        comp["uw_bias"] = flow_bias
+
+    if bars_status == "ok":
+        st.session_state["last_known"][ticker] = {
+            "updated": now_cst().strftime("%Y-%m-%d %H:%M:%S %Z"),
+            "rsi": comp["rsi"],
+            "macd_hist": comp["macd_hist"],
+            "vwap": comp["vwap"],
+            "ema_stack": comp["ema_stack"],
+            "vol_ratio": comp["vol_ratio"],
+            "direction": direction,
+        }
+
+    after_hours_note = "Live"
+    if bars_status != "ok":
+        last = st.session_state["last_known"].get(ticker)
+        if last:
+            after_hours_note = f"N/A / last known ({last['updated']})"
+            comp["rsi"] = last.get("rsi")
+            comp["macd_hist"] = last.get("macd_hist")
+            comp["vwap"] = last.get("vwap")
+            comp["ema_stack"] = last.get("ema_stack")
+            comp["vol_ratio"] = last.get("vol_ratio")
+            direction = last.get("direction", direction)
+        else:
+            after_hours_note = "N/A / last known unavailable"
+
+    signal_out = signal if confidence >= institutional_min else "WAIT"
+
+    row = {
+        "Ticker": ticker,
+        "Confidence": confidence,
+        "Direction": direction,
+        "Signal": signal_out,
+        "After-hours": after_hours_note,
+        "UW unusual": unusual,
+        "UW bias": comp["uw_bias"],
+        "Gamma bias": comp["gamma_bias"],
+        "RSI": fmt(comp["rsi"], 1),
+        "MACD hist": fmt(comp["macd_hist"], 4),
+        "VWAP": fmt(comp["vwap"], 2),
+        "EMA 9/20/50": comp["ema_stack"],
+        "Volume ratio": fmt(comp["vol_ratio"], 2),
+        "10Y": comp["teny"],
     }
+    out_rows.append(row)
+
+    status_rows.append(
+        {
+            "Ticker": ticker,
+            "EODHD intraday": bars_status,
+            "EODHD news": news_status,
+            "UW options-volume": uw_status,
+            "UW flow-alerts": flow_status,
+        }
+    )
 
-    # Build signals
-    for t in tickers:
-        sym = as_eodhd_symbol(t)
-
-        bars, s_bars = eodhd_intraday(sym, interval="5m", lookback_minutes=int(price_lookback))
-        news, s_news = eodhd_news(sym, lookback_minutes=int(news_lookback), limit=30)
-        uw_bias_obj, s_uw = uw_options_volume_bias(t)
-
-        # Flow alerts are OPTIONAL; we try but do not crash if broken
-        alerts, s_alerts = uw_flow_alerts(t, limit=50)
-
-        # If alerts present, set UW_unusual quickly (premium >= 1M and put/call)
-        uw_unusual = "NO"
-        uw_dir = "Neutral"
-        if alerts and isinstance(alerts, list):
-            for it in alerts[:50]:
-                prem = float(it.get("premium") or 0)
-                otype = (it.get("option_type") or it.get("type") or "").lower()
-                tags = it.get("tags") or []
-                if prem >= 1_000_000:
-                    uw_unusual = "YES"
-                    # direction from tags or option_type
-                    if isinstance(tags, list) and any("bearish" in str(x).lower() for x in tags):
-                        uw_dir = "Bearish"
-                    elif isinstance(tags, list) and any("bullish" in str(x).lower() for x in tags):
-                        uw_dir = "Bullish"
-                    elif otype == "put":
-                        uw_dir = "Bearish"
-                    elif otype == "call":
-                        uw_dir = "Bullish"
-                    break
-
-        confidence, direction, signal, comp = score_ticker(
-            bars, news, uw_bias_obj, teny,
-            w_rsi, w_macd, w_vwap, w_ema, w_vol, w_uw, w_news, w_10y
+    if signal_out in ["BUY CALLS", "BUY PUTS"]:
+        alerts_rows.append(f"{ticker}: {signal_out} | {direction} | confidence {confidence}")
+
+    for item in news_items[:8]:
+        all_news_rows.append(
+            {
+                "Ticker": ticker,
+                "Date": item.get("date", ""),
+                "Title": item.get("title", "(no title)"),
+                "Link": item.get("link") or item.get("url") or "",
+            }
         )
-        comp["UW_unusual"] = uw_unusual
-        comp["UW_bias"] = uw_dir if uw_unusual == "YES" and uw_dir != "Neutral" else comp["UW_bias"]
 
-        # Institutional filter
-        if confidence < institutional_min:
-            inst_signal = "WAIT"
-        else:
-            inst_signal = signal
-
-        rows.append({
-            "Ticker": t,
-            "Confidence": confidence,
-            "Direction": direction,
-            "Signal": inst_signal,     # filtered output
-            "UW_Unusual": comp["UW_unusual"],
-            "UW_Bias": comp["UW_bias"],
-            "Gamma_bias": comp["Gamma_bias"],
-            "RSI": fmt_num(comp["RSI"], 1),
-            "MACD_hist": fmt_num(comp["MACD_hist"], 4),
-            "VWAP": fmt_num(comp["VWAP"], 2),
-            "EMA_stack": comp["EMA_stack"] if comp["EMA_stack"] else "N/A",
-            "Vol_ratio": fmt_num(comp["Vol_ratio"], 2),
-            "News_sent": comp["News_sent"],
-            "10Y": comp["10Y"],
-        })
-
-        endpoint_notes["eodhd_intraday"].append((t, s_bars))
-        endpoint_notes["eodhd_news"].append((t, s_news))
-        endpoint_notes["uw_options_volume"].append((t, s_uw))
-        endpoint_notes["uw_flow_alerts"].append((t, s_alerts))
-
-    df_out = pd.DataFrame(rows)
-
-    st.dataframe(df_out, use_container_width=True, hide_index=True)
-
-    # Alerts section
-    st.divider()
-    st.subheader(f"Institutional Alerts (≥{institutional_min} only)")
-    inst = df_out[df_out["Confidence"] >= institutional_min]
-    inst = inst[inst["Signal"].isin(["BUY CALLS", "BUY PUTS"])]
-    if inst.empty:
-        st.info("No institutional signals right now.")
-    else:
-        for _, r in inst.iterrows():
-            st.success(f"{r['Ticker']}: {r['Signal']} — {r['Direction']} — Confidence {r['Confidence']} (UW unusual: {r['UW_Unusual']}, UW bias: {r['UW_Bias']})")
+summary_col, status_col = st.columns([1.6, 1.0])
+with summary_col:
+    st.subheader("Live score table")
+    st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)
 
-    # UW flow alerts panel
-    st.divider()
-    st.subheader("Unusual Flow Alerts (UW API)")
-    if not UW_TOKEN:
-        st.warning("UW_TOKEN missing in Secrets.")
-    elif not UW_FLOW_ALERTS_URL:
-        st.warning("UW_FLOW_ALERTS_URL missing in Secrets.")
+    st.subheader(f"Institutional alerts (confidence ≥ {institutional_min})")
+    if alerts_rows:
+        for line in alerts_rows:
+            st.success(line)
     else:
-        # show endpoint health summary
-        bad = [x for x in endpoint_notes["uw_flow_alerts"] if x[1] not in ["ok", "empty"]]
-        if bad:
-            # Common failure = 404 due to plan/route mismatch
-            st.error(
-                f"UW flow alerts failing for some tickers (example: {bad[0][0]} => {bad[0][1]}).\n\n"
-                f"URL being used (from Secrets): {UW_FLOW_ALERTS_URL}\n\n"
-                "If you see http_404: this is almost always an API route/plan access issue (not your code)."
-            )
-        else:
-            st.success("UW flow-alerts endpoint responded (ok/empty).")
-
-        st.caption("This panel won’t crash your app. It will show RED if UW blocks the endpoint.")
-        st.code(
-            "Tip: The working URL MUST exactly match the 'curl --url ...' line in UW docs.\n"
-            "If UW returns 404, your token may not have access to that endpoint, or the URL is wrong.\n"
-            "Your code uses UW_FLOW_ALERTS_URL EXACTLY as typed in Secrets."
-        )
+        st.info("No institutional CALL/PUT alerts at this moment.")
 
-    # EODHD News panel (clickable)
-    st.divider()
-    st.subheader(f"News — last {int(news_lookback)} minutes (EODHD)")
-    any_news = False
-    for t in tickers:
-        sym = as_eodhd_symbol(t)
-        items, st_news = eodhd_news(sym, lookback_minutes=int(news_lookback), limit=15)
-        if items:
-            any_news = True
-            for it in items[:8]:
-                title = it.get("title", "(no title)")
-                url = it.get("link") or it.get("url") or ""
-                dt = it.get("date") or ""
-                if url:
-                    st.markdown(f"- **{t}** — {dt} — [{title}]({url})")
-                else:
-                    st.markdown(f"- **{t}** — {dt} — {title}")
-    if not any_news:
-        st.info("No news in this lookback window (or EODHD returned none).")
-
-    # Debug / endpoint status
-    st.divider()
-    with st.expander("Debug (endpoint status per ticker)"):
-        st.write("EODHD intraday:", endpoint_notes["eodhd_intraday"])
-        st.write("EODHD news:", endpoint_notes["eodhd_news"])
-        st.write("UW options-volume:", endpoint_notes["uw_options_volume"])
-        st.write("UW flow-alerts:", endpoint_notes["uw_flow_alerts"])
-        st.write("10Y yield source:", teny_status)
+    st.subheader(f"EODHD news (last {int(news_lookback)} minutes)")
+    if all_news_rows:
+        for n in all_news_rows:
+            if n["Link"]:
+                st.markdown(f"- **{n['Ticker']}** — {n['Date']} — [{n['Title']}]({n['Link']})")
+            else:
+                st.markdown(f"- **{n['Ticker']}** — {n['Date']} — {n['Title']}")
+    else:
+        st.info("No news in this window.")
+
+with status_col:
+    st.subheader("API status")
+    eodhd_ok = all(s["EODHD intraday"] in ["ok", "empty"] and s["EODHD news"] in ["ok", "empty"] for s in status_rows) and bool(EODHD_API_KEY)
+    uw_ok = all(s["UW options-volume"] in ["ok", "empty"] and s["UW flow-alerts"] in ["ok", "empty"] for s in status_rows) and bool(UW_TOKEN)
+    flow_ok = all(s["UW flow-alerts"] in ["ok", "empty"] for s in status_rows) and bool(UW_FLOW_ALERTS_URL)
+    ten_ok = ten_status == "ok"
+
+    st.markdown(f"{endpoint_emoji(eodhd_ok)} **EODHD API**")
+    st.markdown(f"{endpoint_emoji(uw_ok)} **Unusual Whales API**")
+    st.markdown(f"{endpoint_emoji(flow_ok)} **UW flow-alerts endpoint**")
+    st.markdown(f"{endpoint_emoji(ten_ok)} **10Y yield feed (optional)**")
+
+    st.caption("Endpoint status by ticker")
+    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
+
+    st.caption("Secrets detected")
+    st.write({
+        "EODHD_API_KEY": "set" if EODHD_API_KEY else "missing",
+        "UW_TOKEN": "set" if UW_TOKEN else "missing",
+        "UW_FLOW_ALERTS_URL": "set" if UW_FLOW_ALERTS_URL else "missing",
+    })
 
EOF
)
