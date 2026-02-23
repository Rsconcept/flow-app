#!/usr/bin/env python3
"""
v3.1 Options Flow Scanner (UW + Polygon + EODHD) with:
- Live/Replay toggle (snapshot)
- Polygon intraday spot-at-time (minute bars) + fallback to prev close
- EODHD options chain IV (current IV)
- Local IV history store (free IV ramp by saving daily IV per contract)
- Clear Endpoint Status panel

requirements.txt:
streamlit
requests

Streamlit Secrets:
UW_TOKEN="..."
POLYGON_API_KEY="..."
EODHD_API_KEY="..."
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

# -------------------- Constants --------------------

CT_OFFSET = -6  # Central Time offset
MAX_PENDING_TRADES = 50

PENDING_TRADES_FILE = "pending_trades.json"
INVERSE_SIGNALS_FILE = "inverse_signals.json"
VALIDATED_TRADES_FILE = "validated_trades.json"

SNAPSHOT_FILE = "last_uw_flows.json"
IV_STORE_FILE = "iv_history_store.json"

EXCLUDED_TICKERS_DEFAULT = {
    # Indexes
    "SPX", "SPXW", "NDX", "VIX", "RUT", "DJX", "XSP", "OEX",
    # Index ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # Sector ETFs
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC",
}

# -------------------- Helpers --------------------


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Tuple[int, Optional[Any], str]:
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code == 200:
            try:
                return resp.status_code, resp.json(), ""
            except Exception:
                return resp.status_code, None, "Failed to parse JSON."
        return resp.status_code, None, f"HTTP {resp.status_code}: {resp.text[:400]}"
    except Exception as e:
        return 0, None, f"Request error: {e}"


def ensure_json_file(path: str, default_value: Any) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_value, f, indent=2)


def read_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json_file(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def today_yyyy_mm_dd() -> str:
    return date.today().strftime("%Y-%m-%d")


def pretty_money(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)


def calculate_dte(expiry_yyyy_mm_dd: str) -> int:
    try:
        exp_date = datetime.strptime(expiry_yyyy_mm_dd, "%Y-%m-%d")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return (exp_date - today).days
    except Exception:
        return 0


def to_central_time(iso_timestamp: str, ct_offset_hours: int = CT_OFFSET) -> str:
    """Convert ISO timestamp to 'YYYY-MM-DD HH:MM:SS AM CT'. If parsing fails, return raw."""
    if not iso_timestamp:
        return ""
    try:
        if "T" in iso_timestamp:
            dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(iso_timestamp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        ct = dt + timedelta(hours=ct_offset_hours)
        return ct.strftime("%Y-%m-%d %I:%M:%S %p CT")
    except Exception:
        return iso_timestamp


def parse_uw_time_to_utc(iso_timestamp: str) -> Optional[datetime]:
    """UW timestamps are usually ISO with Z. Return aware UTC dt."""
    if not iso_timestamp:
        return None
    try:
        if "T" in iso_timestamp:
            dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        dt = datetime.strptime(iso_timestamp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def contract_key(ticker: str, expiry: str, option_type: str, strike: float) -> str:
    return f"{ticker.upper()}|{expiry}|{option_type.lower()}|{float(strike):.2f}"


# -------------------- Snapshot Manager --------------------


class SnapshotManager:
    def __init__(self, path: str = SNAPSHOT_FILE):
        self.path = path

    def save(self, flows: List[Dict[str, Any]]) -> None:
        payload = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "count": len(flows),
            "flows": flows,
        }
        write_json_file(self.path, payload)

    def load(self) -> List[Dict[str, Any]]:
        payload = read_json_file(self.path, {})
        if isinstance(payload, dict) and isinstance(payload.get("flows"), list):
            return payload["flows"]
        if isinstance(payload, list):
            return payload
        return []

    def get_meta(self) -> Dict[str, Any]:
        payload = read_json_file(self.path, {})
        if isinstance(payload, dict):
            return {
                "saved_at_utc": payload.get("saved_at_utc"),
                "count": payload.get("count"),
            }
        return {}

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def raw_text(self) -> str:
        if not self.exists():
            return ""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""


# -------------------- Local IV Store --------------------


class LocalIVStore:
    """
    Stores IV history per contract in JSON:
    {
      "AAPL|2026-03-20|call|200.00": [{"date":"2026-02-10","iv":45.2}, ...],
      ...
    }
    """

    def __init__(self, path: str = IV_STORE_FILE):
        self.path = path
        ensure_json_file(self.path, default_value={})

    def load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        data = read_json_file(self.path, {})
        return data if isinstance(data, dict) else {}

    def save_all(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        write_json_file(self.path, data)

    def upsert_today(self, key: str, iv_value: float) -> None:
        if iv_value <= 0:
            return
        data = self.load_all()
        rows = data.get(key, [])
        if not isinstance(rows, list):
            rows = []

        t = today_yyyy_mm_dd()
        replaced = False
        for r in rows:
            if isinstance(r, dict) and r.get("date") == t:
                r["iv"] = float(iv_value)
                replaced = True
                break
        if not replaced:
            rows.append({"date": t, "iv": float(iv_value)})

        rows = [r for r in rows if isinstance(r, dict) and r.get("date") and safe_float(r.get("iv", 0)) > 0]
        rows.sort(key=lambda r: str(r.get("date")))
        rows = rows[-120:]  # keep last ~4 months

        data[key] = rows
        self.save_all(data)

    def get_history_map(self, key: str) -> Dict[str, float]:
        data = self.load_all()
        rows = data.get(key, [])
        out: Dict[str, float] = {}
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                d = str(r.get("date", ""))
                iv = safe_float(r.get("iv", 0))
                if d and iv > 0:
                    out[d] = iv
        return out

    def detect_ramp(
        self, key: str, lookback_days: int = 3, require_strict: bool = True
    ) -> Tuple[bool, List[Tuple[str, float]]]:
        hist = self.get_history_map(key)
        if len(hist) < lookback_days:
            return False, []
        dates_sorted = sorted(hist.keys())
        last_dates = dates_sorted[-lookback_days:]
        pts = [(d, float(hist[d])) for d in last_dates]
        vals = [v for _, v in pts]
        if require_strict:
            ok = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
        else:
            ok = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        return ok, pts

    def raw_text(self) -> str:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def reset(self) -> None:
        self.save_all({})


# -------------------- API Clients --------------------


class UnusualWhalesAPI:
    BASE_URL = "https://api.unusualwhales.com/api"

    def __init__(self, token: str):
        self.token = token.strip()
        self.headers = {
            "Accept": "application/json, text/plain",
            "Authorization": f"Bearer {self.token}" if self.token else "",
        }

    def test_connection(self) -> Tuple[bool, str]:
        if not self.token:
            return False, "Missing UW_TOKEN"
        status, data, err = http_get(
            f"{self.BASE_URL}/option-trades/flow-alerts",
            headers=self.headers,
            params={"limit": 1},
        )
        if status == 200 and data:
            return True, "UW flow-alerts (ok)"
        return False, err or "UW failed"

    def get_flows(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.token:
            return []
        status, data, _ = http_get(
            f"{self.BASE_URL}/option-trades/flow-alerts",
            headers=self.headers,
            params={"limit": limit},
        )
        if status != 200 or not data:
            return []
        return data.get("data", []) or []

    def get_ticker_flow(self, ticker: str, limit: int = 180) -> List[Dict[str, Any]]:
        if not self.token:
            return []
        ticker = ticker.strip().upper()
        status, data, _ = http_get(
            f"{self.BASE_URL}/stock/{ticker}/options-flow",
            headers=self.headers,
            params={"limit": limit},
        )
        if status == 200 and data:
            return data.get("data", []) or []
        return []

    def get_earnings(self, ticker: str) -> List[Dict[str, Any]]:
        if not self.token:
            return []
        ticker = ticker.strip().upper()
        status, data, _ = http_get(
            f"{self.BASE_URL}/stock/{ticker}/earnings-history",
            headers=self.headers,
        )
        if status == 200 and data:
            return data.get("data", []) or []
        return []


class PolygonAPI:
    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    def test_connection(self) -> Tuple[bool, str]:
        if not self.api_key:
            return False, "Missing POLYGON_API_KEY"
        status, data, err = http_get(
            f"{self.BASE_URL}/v2/aggs/ticker/AAPL/prev",
            params={"apiKey": self.api_key},
        )
        if status == 200 and data and data.get("status") == "OK":
            return True, "Polygon (ok)"
        return False, err or "Polygon failed"

    def get_previous_close(self, ticker: str) -> Tuple[float, str]:
        if not self.api_key:
            return 0.0, "Missing POLYGON_API_KEY"
        ticker = ticker.strip().upper()
        status, data, err = http_get(
            f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/prev",
            params={"apiKey": self.api_key},
        )
        if status == 200 and data:
            results = data.get("results") or []
            if results:
                return safe_float(results[0].get("c", 0.0)), "Polygon prev close"
        return 0.0, f"Polygon prev close error: {err or 'no data'}"

    def get_spot_at_time(self, ticker: str, timestamp_ct: str) -> Tuple[float, str]:
        """
        Get spot near the flow timestamp using Polygon 1-min aggregates for that day.
        Falls back to previous close if bars unavailable.
        """
        ticker = ticker.strip().upper()
        if not self.api_key:
            return 0.0, "Missing POLYGON_API_KEY"
        try:
            if not timestamp_ct or "CT" not in timestamp_ct:
                return self.get_previous_close(ticker)

            parts = timestamp_ct.replace(" CT", "").strip()
            dt_ct = datetime.strptime(parts, "%Y-%m-%d %I:%M:%S %p")
            trade_date = dt_ct.strftime("%Y-%m-%d")

            url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{trade_date}/{trade_date}"
            status, data, err = http_get(url, params={"apiKey": self.api_key, "limit": 50000})
            if status == 200 and data and data.get("results"):
                bars = data["results"]
                target_min = dt_ct.hour * 60 + dt_ct.minute

                closest = None
                best = 10_000
                for bar in bars:
                    bar_ts = safe_float(bar.get("t", 0)) / 1000.0
                    bar_dt_utc = datetime.utcfromtimestamp(bar_ts).replace(tzinfo=timezone.utc)
                    bar_ct = bar_dt_utc + timedelta(hours=CT_OFFSET)
                    bar_min = bar_ct.hour * 60 + bar_ct.minute
                    diff = abs(bar_min - target_min)
                    if diff < best:
                        best = diff
                        closest = bar

                if closest:
                    # even if diff is a bit large (early open / thin), still use closest bar
                    return safe_float(closest.get("c", 0.0)), f"Polygon intraday (Δ{best}m)"
            # fallback
            px, src = self.get_previous_close(ticker)
            return px, f"{src} (intraday missing)"
        except Exception as e:
            px, src = self.get_previous_close(ticker)
            return px, f"{src} (spot error: {e})"

    def get_price_history(self, ticker: str, days: int = 30) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        ticker = ticker.strip().upper()
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            url = (
                f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/"
                f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            )
            status, data, _ = http_get(url, params={"apiKey": self.api_key, "limit": days})
            if status == 200 and data:
                out: List[Dict[str, Any]] = []
                for r in (data.get("results") or []):
                    out.append(
                        {
                            "date": datetime.fromtimestamp(safe_float(r.get("t", 0)) / 1000.0).strftime("%Y-%m-%d"),
                            "open": safe_float(r.get("o", 0)),
                            "high": safe_float(r.get("h", 0)),
                            "low": safe_float(r.get("l", 0)),
                            "close": safe_float(r.get("c", 0)),
                            "volume": safe_float(r.get("v", 0)),
                        }
                    )
                return out
            return []
        except Exception:
            return []

    def calculate_support_resistance(self, ticker: str, strike: float) -> Dict[str, Any]:
        candles = self.get_price_history(ticker, 30)
        if not candles or len(candles) < 5:
            return {}
        highs = [safe_float(c["high"]) for c in candles]
        lows = [safe_float(c["low"]) for c in candles]

        resistance = max(highs[-10:]) if len(highs) >= 10 else max(highs)
        support = min(lows[-10:]) if len(lows) >= 10 else min(lows)

        recent_high_wick = max(highs[-5:]) if len(highs) >= 5 else max(highs)
        recent_low_wick = min(lows[-5:]) if len(lows) >= 5 else min(lows)

        wick_triggered = (abs(strike - recent_high_wick) < 0.5) or (abs(strike - recent_low_wick) < 0.5)
        return {
            "resistance": resistance,
            "support": support,
            "recent_high_wick": recent_high_wick,
            "recent_low_wick": recent_low_wick,
            "wick_triggered": wick_triggered,
        }


class EODHDAPI:
    BASE_URL = "https://eodhd.com/api"

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    def test_connection(self) -> Tuple[bool, str]:
        if not self.api_key:
            return False, "Missing EODHD_API_KEY"
        status, data, err = http_get(
            f"{self.BASE_URL}/eod/AAPL.US",
            params={"api_token": self.api_key, "fmt": "json", "from": "2025-01-01", "limit": 1},
        )
        if status == 200 and data:
            return True, "EODHD (ok)"
        return False, err or "EODHD failed"

    def get_iv_from_chain(self, ticker: str, strike: float, expiry: str, option_type: str) -> float:
        """Return current IV% for a contract if found; else 0."""
        if not self.api_key:
            return 0.0
        ticker = ticker.strip().upper()
        option_type = option_type.lower().strip()
        try:
            url = f"{self.BASE_URL}/options/{ticker}.US"
            status, data, err = http_get(url, params={"api_token": self.api_key, "fmt": "json"})
            if status != 200 or not data or not isinstance(data, dict):
                return 0.0

            for exp_key, chain in data.items():
                if expiry not in str(exp_key):
                    continue
                if not isinstance(chain, dict):
                    continue

                options_list = chain.get(option_type + "s", [])
                if isinstance(options_list, dict):
                    options_list = list(options_list.values())
                if not isinstance(options_list, list):
                    continue

                for opt in options_list:
                    if not isinstance(opt, dict):
                        continue
                    opt_strike = safe_float(opt.get("strike", 0))
                    if abs(opt_strike - strike) < 0.5:
                        iv = safe_float(opt.get("impliedVolatility", 0))
                        if iv > 0:
                            if iv < 1:
                                iv *= 100
                            return float(iv)
            return 0.0
        except Exception:
            return 0.0


# -------------------- Enrichment + Ladder --------------------


class LadderDetector:
    """
    Ladder detection:
    - Requires same expiry + type
    - Requires >= min_unique_strikes total strikes within recent_minutes (including target)
    """

    def __init__(self, uw: UnusualWhalesAPI):
        self.uw = uw

    def detect(
        self,
        ticker: str,
        target_strike: float,
        option_type: str,
        expiry: str,
        recent_minutes: int = 90,
        min_unique_strikes: int = 3,
    ) -> Tuple[bool, List[float]]:
        flows = self.uw.get_ticker_flow(ticker, limit=180)
        if not flows:
            return False, []

        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(minutes=recent_minutes)

        strikes: set[float] = {float(target_strike)}
        for f in flows:
            f_type = str(f.get("option_type", "")).lower().strip()
            f_expiry = str(f.get("expiry", "")).strip()
            if f_type != option_type or f_expiry != expiry:
                continue
            ts = parse_uw_time_to_utc(str(f.get("start_time", "")))
            if ts is None or ts < cutoff:
                continue
            s = safe_float(f.get("strike", 0))
            if s > 0:
                strikes.add(float(s))

        if len(strikes) >= min_unique_strikes:
            related = sorted([s for s in strikes if abs(s - target_strike) > 1e-9])
            return True, related
        return False, []


class DataEnricher:
    def __init__(
        self,
        uw: UnusualWhalesAPI,
        polygon: PolygonAPI,
        eodhd: EODHDAPI,
        iv_store: LocalIVStore,
        iv_lookback_days: int = 3,
        iv_require_strict: bool = True,
    ):
        self.uw = uw
        self.polygon = polygon
        self.eodhd = eodhd
        self.iv_store = iv_store
        self.iv_lookback_days = iv_lookback_days
        self.iv_require_strict = iv_require_strict

    def enrich_trade(self, raw_flow: Dict[str, Any], use_iv: bool = True) -> Dict[str, Any]:
        ticker = str(raw_flow.get("ticker", "")).upper().strip()
        strike = safe_float(raw_flow.get("strike", 0))
        option_type = str(raw_flow.get("option_type", "call")).lower().strip()
        expiry = str(raw_flow.get("expiry", "")).strip()
        timestamp = str(raw_flow.get("start_time", "")).strip()

        timestamp_ct = to_central_time(timestamp)

        # Polygon spot at time
        spot, spot_source = self.polygon.get_spot_at_time(ticker, timestamp_ct)

        # Strike distance
        if spot > 0:
            strike_dist_pct = abs(strike - spot) / spot * 100.0
            is_otm = (strike > spot) if option_type == "call" else (strike < spot)
        else:
            strike_dist_pct = 0.0
            is_otm = True

        # Premium and ask%
        total_prem = (
            safe_float(raw_flow.get("total_ask_side_prem", 0))
            + safe_float(raw_flow.get("total_bid_side_prem", 0))
            + safe_float(raw_flow.get("total_mid_side_prem", 0))
            + safe_float(raw_flow.get("total_no_side_prem", 0))
        )
        ask_prem = safe_float(raw_flow.get("total_ask_side_prem", 0))
        ask_pct = (ask_prem / total_prem * 100.0) if total_prem > 0 else 0.0

        volume = safe_int(raw_flow.get("total_size", 0))
        oi = safe_int(raw_flow.get("open_interest", 0))
        vol_oi_ratio = (volume / oi) if oi > 0 else 999.0

        denom = (spot * 100.0 * max(volume, 1)) if spot > 0 else 0.0
        premium_pct = (total_prem / denom * 100.0) if denom > 0 else 0.0

        dte = calculate_dte(expiry)

        # Polygon S/R wick
        sr = self.polygon.calculate_support_resistance(ticker, strike)

        # Earnings (UW)
        days_to_er: Optional[int] = None
        earnings = self.uw.get_earnings(ticker)
        if earnings:
            today_dt = datetime.now()
            for er in earnings:
                er_date_str = str(er.get("date", "")).strip()
                try:
                    er_date = datetime.strptime(er_date_str, "%Y-%m-%d")
                    delta = (er_date - today_dt).days
                    if 0 <= delta <= 30:
                        days_to_er = delta
                        break
                except Exception:
                    continue

        # EODHD chain IV + local store ramp
        ckey = contract_key(ticker, expiry, option_type, strike)
        current_iv = 0.0
        if use_iv:
            current_iv = self.eodhd.get_iv_from_chain(ticker, strike, expiry, option_type)
            if current_iv > 0:
                self.iv_store.upsert_today(ckey, current_iv)

        local_iv = self.iv_store.get_history_map(ckey)
        iv_ramping, ramp_pts = self.iv_store.detect_ramp(
            ckey, lookback_days=self.iv_lookback_days, require_strict=self.iv_require_strict
        )

        clean_exception = (
            ask_pct >= 70 and vol_oi_ratio > 1 and strike_dist_pct <= 7 and 2.5 <= premium_pct <= 5.0
        )

        return {
            "ticker": ticker,
            "strike": strike,
            "option_type": option_type,
            "expiry": expiry,
            "entry_timestamp": timestamp_ct,
            "spot": spot,
            "spot_source": spot_source,
            "strike_dist_pct": strike_dist_pct,
            "is_otm": is_otm,
            "total_premium": total_prem,
            "premium_pct": premium_pct,
            "volume": volume,
            "open_interest": oi,
            "vol_oi_ratio": vol_oi_ratio,
            "ask_pct": ask_pct,
            "dte": dte,
            "wick_triggered": bool(sr.get("wick_triggered", False)),
            "support": safe_float(sr.get("support", 0)),
            "resistance": safe_float(sr.get("resistance", 0)),
            "contract_key": ckey,
            "current_iv": float(current_iv),
            "iv_history_local": local_iv,
            "iv_ramping": bool(iv_ramping),
            "iv_ramp_points": ramp_pts,
            "iv_ramp_lookback_days": int(self.iv_lookback_days),
            "days_to_earnings": days_to_er,
            "clean_exception": bool(clean_exception),
            "has_sweep": bool(raw_flow.get("is_sweep", False)),
            "ladder_role": "isolated",
            "related_strikes": [],
            "category_tags": [],
            "_raw": raw_flow,
        }


# -------------------- Scoring --------------------


class V31ScoringEngine:
    def score(self, record: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        factors: List[str] = []
        penalties: List[str] = []
        record["category_tags"] = record.get("category_tags") or []

        prem_pct = safe_float(record.get("premium_pct", 0))
        if 2.5 <= prem_pct <= 5.0:
            score += 2
            factors.append(f"Premium {prem_pct:.1f}% (+2)")
        elif 1.0 <= prem_pct < 2.5:
            score += 1
            factors.append(f"Premium {prem_pct:.1f}% (+1)")
        elif prem_pct < 1.0:
            score -= 2
            penalties.append(f"Ultra-low premium {prem_pct:.2f}% (-2)")
        else:
            factors.append(f"Excessive premium {prem_pct:.1f}% (0)")

        dist = safe_float(record.get("strike_dist_pct", 0))
        if dist <= 7:
            score += 2
            factors.append(f"Strike {dist:.1f}% OTM (+2)")
        elif dist <= 15:
            factors.append(f"Strike {dist:.1f}% OTM (0)")
        else:
            score -= 2
            penalties.append(f"Strike {dist:.1f}% deep OTM (-2)")

        dte = safe_int(record.get("dte", 0))
        if 7 <= dte <= 21:
            score += 1
            factors.append(f"DTE {dte}d (+1)")
        elif dte <= 1:
            score -= 1
            penalties.append("0-1 DTE (-1)")

        ask_pct = safe_float(record.get("ask_pct", 0))
        if ask_pct >= 70:
            score += 1
            factors.append(f"Ask {ask_pct:.0f}% (+1)")
        elif 0 < ask_pct < 30:
            score -= 2
            penalties.append(f"Bid/mid heavy (Ask {ask_pct:.0f}%) (-2)")
        else:
            factors.append("Execution side unknown/neutral (0)")

        vol_oi = safe_float(record.get("vol_oi_ratio", 0))
        if vol_oi >= 2:
            score += 2
            factors.append(f"Vol/OI {vol_oi:.1f}x (+2)")
        elif vol_oi >= 1:
            score += 1
            factors.append(f"Vol/OI {vol_oi:.1f}x (+1)")

        if bool(record.get("wick_triggered", False)):
            score -= 2
            penalties.append("Wick reversal strike (-2)")

        if bool(record.get("iv_ramping", False)):
            score += 1
            factors.append(f"IV ramp (local) (+1) {record.get('iv_ramp_points', [])}")

        ladder_role = str(record.get("ladder_role", "isolated")).lower()
        if ladder_role in ("anchor", "specleg", "ladder"):
            score += 1
            factors.append("Ladder/cluster (+1)")
        else:
            score -= 1
            penalties.append("Isolated (-1)")

        if str(record.get("option_type", "")).lower() == "put":
            strike = safe_float(record.get("strike", 0))
            support = safe_float(record.get("support", 0))
            if strike > support and support > 0:
                score -= 1
                penalties.append("Put above support (-1)")

        days_to_er = record.get("days_to_earnings")
        if isinstance(days_to_er, int) and 2 <= days_to_er <= 10:
            score += 1
            factors.append(f"Catalyst {days_to_er}d (+1)")

        # Caps (unlock if local ramp true)
        if bool(record.get("iv_ramping", False)):
            max_score = 12
        elif bool(record.get("clean_exception", False)):
            max_score = 7
        else:
            max_score = 6

        final_score = min(score, max_score)

        if final_score >= 8:
            verdict = "HIGH CONVICTION"
            record["category_tags"].append("HighConviction")
        elif final_score >= 7:
            verdict = "TRADEABLE"
            record["category_tags"].append("Tradeable")
        elif final_score >= 6:
            verdict = "MODERATE"
            record["category_tags"].append("Moderate")
        elif final_score >= 5:
            verdict = "WATCHLIST"
            record["category_tags"].append("Watchlist")
        else:
            verdict = "TRAP / SKIP"
            record["category_tags"].append("Trap")

        if bool(record.get("has_sweep", False)):
            record["category_tags"].append("Sweep")
        if safe_float(record.get("vol_oi_ratio", 0)) >= 10:
            record["category_tags"].append("LonelyWhale")
        if isinstance(days_to_er, int) and days_to_er <= 10:
            record["category_tags"].append("PreER")
        if safe_float(record.get("current_iv", 0)) > 0:
            record["category_tags"].append("HasIV")
        if safe_float(record.get("spot", 0)) <= 0:
            record["category_tags"].append("NoSpot")

        record["predictive_score"] = int(final_score)
        record["max_score"] = int(max_score)
        record["score_factors"] = factors
        record["score_penalties"] = penalties
        record["verdict"] = verdict
        return record


# -------------------- Queues --------------------


class QueueManager:
    def __init__(self, pending_file: str, inverse_file: str, validated_file: str):
        self.pending_file = pending_file
        self.inverse_file = inverse_file
        self.validated_file = validated_file
        ensure_json_file(self.pending_file, default_value=[])
        ensure_json_file(self.inverse_file, default_value=[])
        ensure_json_file(self.validated_file, default_value=[])

    def load_queue(self, filepath: str) -> List[Dict[str, Any]]:
        if not os.path.exists(filepath):
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                x = json.load(f)
                return x if isinstance(x, list) else []
        except Exception:
            return []

    def save_queue(self, filepath: str, data: List[Dict[str, Any]]) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def add_pending(self, trade: Dict[str, Any]) -> None:
        q = self.load_queue(self.pending_file)
        q.append(trade)
        if len(q) > MAX_PENDING_TRADES:
            q = q[-MAX_PENDING_TRADES:]
        self.save_queue(self.pending_file, q)

    def add_inverse(self, trade: Dict[str, Any]) -> None:
        q = self.load_queue(self.inverse_file)
        q.append(trade)
        self.save_queue(self.inverse_file, q)

    def add_validated(self, trade: Dict[str, Any]) -> None:
        q = self.load_queue(self.validated_file)
        q.append(trade)
        self.save_queue(self.validated_file, q)


# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="v3.1 Options Flow Scanner", layout="wide")
st.title("v3.1 Options Flow Scanner (Polygon Spot + EODHD IV + Local IV Ramp)")
st.caption("Live/Replay • Polygon intraday spot • EODHD chain IV • local IV history • JSON queues")

snapshot = SnapshotManager(SNAPSHOT_FILE)
iv_store = LocalIVStore(IV_STORE_FILE)

with st.sidebar:
    st.header("Data Source")
    replay_mode = st.toggle("Replay Mode (use saved snapshot)", value=False)

    meta = snapshot.get_meta()
    if meta.get("saved_at_utc"):
        st.write(f"Snapshot saved: `{meta.get('saved_at_utc')}`")
        st.write(f"Snapshot count: `{meta.get('count')}`")
    else:
        st.write("Snapshot saved: —")

    st.divider()
    st.header("API Keys (Secrets recommended)")
    uw_token = st.text_input("UW_TOKEN", value=os.getenv("UW_TOKEN", ""), type="password")
    polygon_key = st.text_input("POLYGON_API_KEY", value=os.getenv("POLYGON_API_KEY", ""), type="password")
    eodhd_key = st.text_input("EODHD_API_KEY", value=os.getenv("EODHD_API_KEY", ""), type="password")

    st.divider()
    st.subheader("Snapshot Tools")
    uploaded = st.file_uploader("Upload snapshot JSON (optional)", type=["json"])
    if uploaded is not None:
        try:
            up = json.loads(uploaded.read().decode("utf-8"))
            if isinstance(up, dict) and isinstance(up.get("flows"), list):
                snapshot.save(up["flows"])
                st.success("Uploaded snapshot saved.")
            elif isinstance(up, list):
                snapshot.save(up)
                st.success("Uploaded snapshot saved.")
            else:
                st.error("Snapshot format not recognized.")
        except Exception as e:
            st.error(f"Upload error: {e}")

    if snapshot.exists():
        st.download_button(
            "Download current snapshot",
            data=snapshot.raw_text(),
            file_name="last_uw_flows.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Local IV Store")
    st.caption("IV ramp is built for free by saving one IV value per day.")
    st.download_button(
        "Download iv_history_store.json",
        data=iv_store.raw_text(),
        file_name="iv_history_store.json",
        mime="application/json",
        use_container_width=True,
    )
    if st.button("Reset IV store (danger)", type="secondary", use_container_width=True):
        iv_store.reset()
        st.warning("IV store reset.")

    st.divider()
    st.subheader("IV Ramp Settings")
    iv_lookback_days = st.slider("Ramp lookback days", 3, 10, 3, 1)
    iv_strict = st.checkbox("Require strictly increasing IV", value=True)

    st.divider()
    st.subheader("Scan Controls")
    use_iv = st.checkbox("Use EODHD chain IV + store locally", value=True)

    limit = st.slider("UW flow alerts limit (Live only)", 10, 250, 200, 10)

    min_premium = st.number_input("Min premium ($)", min_value=0, value=25_000, step=5_000)
    min_size = st.number_input("Min size (contracts)", min_value=0, value=0, step=100)
    min_vol_oi = st.number_input("Min Vol/OI", min_value=0.0, value=1.0, step=0.1)

    require_vol_gt_oi = st.checkbox("Require Vol > OI", value=False)
    exclude_indices = st.checkbox("Exclude indices + major ETFs (SPX/SPY/QQQ/etc.)", value=True)

    st.divider()
    st.subheader("Ladder Settings")
    ladder_minutes = st.slider("Ladder time window (minutes)", 15, 240, 90, 15)
    ladder_min_strikes = st.slider("Min unique strikes to call a ladder", 2, 6, 3, 1)

    st.divider()
    st.subheader("Queues")
    pending_path = st.text_input("Pending file", value=PENDING_TRADES_FILE)
    inverse_path = st.text_input("Inverse file", value=INVERSE_SIGNALS_FILE)
    validated_path = st.text_input("Validated file", value=VALIDATED_TRADES_FILE)

# Clients
uw = UnusualWhalesAPI(uw_token)
polygon = PolygonAPI(polygon_key)
eodhd = EODHDAPI(eodhd_key)

enricher = DataEnricher(
    uw=uw,
    polygon=polygon,
    eodhd=eodhd,
    iv_store=iv_store,
    iv_lookback_days=int(iv_lookback_days),
    iv_require_strict=bool(iv_strict),
)
scorer = V31ScoringEngine()
ladder = LadderDetector(uw)
queue = QueueManager(pending_path, inverse_path, validated_path)

tabs = st.tabs(["Scan", "Queues", "Connections"])


def get_source_flows() -> Tuple[List[Dict[str, Any]], str]:
    if replay_mode:
        return snapshot.load(), "Replay (snapshot)"
    return uw.get_flows(limit=limit), "Live (UW API)"


# -------------------- Scan Tab --------------------

with tabs[0]:
    st.subheader("Run Scanner")

    colA, colB = st.columns([1, 2], gap="large")
    with colA:
        run = st.button("Run scan", type="primary", use_container_width=True)

    with colB:
        st.markdown(
            """
**What this does**
- Pulls UW flow alerts (Live) or replays a saved snapshot
- Spot-at-time from **Polygon intraday minute bars** (fallback to prev close)
- Current IV from **EODHD options chain** + saves daily IV to local store
- IV ramp detection from your local store (free)
- Scores v3.1 and writes JSON queues
"""
        )

    if run:
        if not replay_mode and not uw_token:
            st.error("Live mode requires UW_TOKEN in Secrets.")
        else:
            with st.spinner("Loading flows..."):
                flows, src = get_source_flows()

            if not flows:
                st.warning(f"No flows from {src}. If using Replay Mode, run Live once to save a snapshot.")
            else:
                # save snapshot on live
                if not replay_mode:
                    try:
                        snapshot.save(flows)
                    except Exception:
                        pass

                excluded = EXCLUDED_TICKERS_DEFAULT if exclude_indices else set()
                skip_reasons = {"premium": 0, "excluded": 0, "vol_oi": 0, "min_size": 0, "min_vol_oi": 0, "bad_ticker": 0}

                results: List[Dict[str, Any]] = []
                skipped = 0

                with st.spinner("Filtering, enriching, laddering, scoring..."):
                    for f in flows:
                        ticker = str(f.get("ticker", "")).upper().strip()
                        if not ticker or len(ticker) > 8:
                            skipped += 1
                            skip_reasons["bad_ticker"] += 1
                            continue

                        total_prem = (
                            safe_float(f.get("total_ask_side_prem", 0))
                            + safe_float(f.get("total_bid_side_prem", 0))
                            + safe_float(f.get("total_mid_side_prem", 0))
                            + safe_float(f.get("total_no_side_prem", 0))
                        )
                        if total_prem < float(min_premium):
                            skipped += 1
                            skip_reasons["premium"] += 1
                            continue

                        if ticker in excluded:
                            skipped += 1
                            skip_reasons["excluded"] += 1
                            continue

                        vol = safe_int(f.get("total_size", 0))
                        oi = safe_int(f.get("open_interest", 0))

                        if vol < int(min_size):
                            skipped += 1
                            skip_reasons["min_size"] += 1
                            continue

                        if require_vol_gt_oi and (oi > 0) and (vol <= oi):
                            skipped += 1
                            skip_reasons["vol_oi"] += 1
                            continue

                        vol_oi_ratio = (vol / oi) if oi > 0 else 999.0
                        if vol_oi_ratio < float(min_vol_oi):
                            skipped += 1
                            skip_reasons["min_vol_oi"] += 1
                            continue

                        enriched = enricher.enrich_trade(f, use_iv=bool(use_iv and eodhd_key))
                        is_l, rel = ladder.detect(
                            ticker=enriched["ticker"],
                            target_strike=enriched["strike"],
                            option_type=enriched["option_type"],
                            expiry=enriched["expiry"],
                            recent_minutes=int(ladder_minutes),
                            min_unique_strikes=int(ladder_min_strikes),
                        )
                        if is_l:
                            enriched["ladder_role"] = "ladder"
                            enriched["related_strikes"] = rel

                        scored = scorer.score(enriched)
                        results.append(scored)

                        if safe_int(scored.get("predictive_score", 0)) >= 5:
                            queue.add_pending(scored)
                        if safe_int(scored.get("predictive_score", 0)) <= -3:
                            queue.add_inverse(scored)

                st.success(f"Source: {src} • Scored {len(results)} • Skipped {skipped}")
                st.write("Skip breakdown:", skip_reasons)

                if results:
                    results.sort(
                        key=lambda r: (safe_int(r.get("predictive_score", 0)), safe_float(r.get("total_premium", 0))),
                        reverse=True,
                    )

                    table = []
                    for r in results:
                        table.append(
                            {
                                "Ticker": r.get("ticker"),
                                "Type": str(r.get("option_type", "")).upper(),
                                "Strike": r.get("strike"),
                                "Expiry": r.get("expiry"),
                                "Spot": round(safe_float(r.get("spot", 0.0)), 2),
                                "SpotSrc": r.get("spot_source"),
                                "Dist%": round(safe_float(r.get("strike_dist_pct", 0.0)), 2),
                                "Premium$": round(safe_float(r.get("total_premium", 0.0))),
                                "Prem%": round(safe_float(r.get("premium_pct", 0.0)), 2),
                                "Ask%": round(safe_float(r.get("ask_pct", 0.0)), 1),
                                "Vol/OI": round(safe_float(r.get("vol_oi_ratio", 0.0)), 2),
                                "IV%": round(safe_float(r.get("current_iv", 0.0)), 2),
                                "IVRamp": bool(r.get("iv_ramping", False)),
                                "Wick": bool(r.get("wick_triggered", False)),
                                "Ladder": (r.get("ladder_role") != "isolated"),
                                "Score": r.get("predictive_score"),
                                "Max": r.get("max_score"),
                                "Verdict": r.get("verdict"),
                                "Tags": ", ".join(r.get("category_tags", [])),
                            }
                        )
                    st.dataframe(table, use_container_width=True, hide_index=True)

                    st.divider()
                    st.subheader("Details")
                    for r in results[:60]:
                        header = (
                            f"{r['ticker']} {str(r['option_type']).upper()} ${r['strike']} {r['expiry']} • "
                            f"Score {r['predictive_score']}/{r['max_score']} • {r['verdict']}"
                        )
                        with st.expander(header, expanded=False):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.write("**Entry (CT)**", r.get("entry_timestamp"))
                                st.write("**Spot**", round(safe_float(r.get("spot", 0)), 2))
                                st.write("**Spot source**", r.get("spot_source"))
                                st.write("**DTE**", r.get("dte"))
                                st.write("**Premium**", pretty_money(safe_float(r.get("total_premium", 0))))
                                st.write("**Ask%**", f"{safe_float(r.get('ask_pct', 0)):.1f}%")
                                st.write("**Vol/OI**", f"{safe_float(r.get('vol_oi_ratio', 0)):.2f}x")
                            with c2:
                                st.write("**IV (current)**", f"{safe_float(r.get('current_iv', 0)):.2f}%")
                                st.write("**IV ramp**", bool(r.get("iv_ramping", False)))
                                st.write("**IV ramp points**", r.get("iv_ramp_points"))
                                st.write("**Local IV history**")
                                st.json(r.get("iv_history_local", {}))

                            st.write("**Factors**")
                            st.write("\n".join([f"• {x}" for x in r.get("score_factors", [])]) or "—")
                            st.write("**Penalties**")
                            st.write("\n".join([f"• {x}" for x in r.get("score_penalties", [])]) or "—")

                            st.write("**Raw UW (debug)**")
                            st.json(r.get("_raw", {}))

# -------------------- Queues Tab --------------------

with tabs[1]:
    st.subheader("Queues")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("### Pending")
        pend = queue.load_queue(pending_path)
        st.write(f"{len(pend)} items")
        st.dataframe(
            [
                {
                    "Ticker": t.get("ticker"),
                    "Type": t.get("option_type"),
                    "Strike": t.get("strike"),
                    "Expiry": t.get("expiry"),
                    "Score": t.get("predictive_score"),
                    "Verdict": t.get("verdict"),
                }
                for t in pend[-200:]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.write("### Inverse")
        inv = queue.load_queue(inverse_path)
        st.write(f"{len(inv)} items")
        st.dataframe(
            [
                {
                    "Ticker": t.get("ticker"),
                    "Type": t.get("option_type"),
                    "Strike": t.get("strike"),
                    "Expiry": t.get("expiry"),
                    "Score": t.get("predictive_score"),
                    "Verdict": t.get("verdict"),
                }
                for t in inv[-200:]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with c3:
        st.write("### Validated")
        val = queue.load_queue(validated_path)
        st.write(f"{len(val)} items")
        st.dataframe(
            [
                {
                    "Ticker": t.get("ticker"),
                    "Type": t.get("option_type"),
                    "Strike": t.get("strike"),
                    "Expiry": t.get("expiry"),
                    "Pred": t.get("predictive_score"),
                    "Val": t.get("validated_score"),
                    "Val Verdict": t.get("validated_verdict"),
                }
                for t in val[-200:]
            ],
            use_container_width=True,
            hide_index=True,
        )

# -------------------- Connections Tab --------------------

with tabs[2]:
    st.subheader("Connections / Endpoint Status")

    # Test endpoints
    uw_ok, uw_msg = uw.test_connection()
    poly_ok, poly_msg = polygon.test_connection()
    eod_ok, eod_msg = eodhd.test_connection()

    # Show like a status list
    if poly_ok:
        st.success(poly_msg + " — Intraday spot should work")
    else:
        st.error(poly_msg + " — Spot may be 0 / fallback won't work")

    if uw_ok:
        st.success(uw_msg)
    else:
        st.error(uw_msg)

    if eod_ok:
        st.success(eod_msg + " — Options chain IV should work")
    else:
        st.error(eod_msg + " — IV will be 0")

    st.divider()
    st.markdown(
        """
### Important
- **Intraday spot** comes from **Polygon**, not EODHD.
- EODHD is used for **options chain IV** only.
- If your **Spot is 0**, it usually means **Polygon key missing/invalid**.
"""
    )
