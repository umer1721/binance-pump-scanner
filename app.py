import time
import math
import os
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st
st.set_page_config(page_title="Binance Futures Early Pump Scanner", page_icon="📈", layout="wide")
# Official USDⓈ-M Futures REST API base URL. If your hosting provider is
# blocked (HTTP 451), deploy from a network/region permitted by Binance.
BASE = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com").rstrip("/")
TIMEOUT = (5, 20)  # connect timeout, read timeout
MAX_WORKERS = 4   # deliberately conservative to reduce rate-limit pressure
_thread_local = threading.local()



def get_session():
    """One requests session per worker thread; avoids sharing Session across threads."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update({"User-Agent": "EarlyPumpScanner/2.0", "Accept": "application/json"})
        _thread_local.session = session
    return _thread_local.session


def api_get(path, params=None):
    """GET public Futures data with bounded retries for transient errors only."""
    url = f"{BASE}{path}"
    last_error = None
    for attempt in range(3):
        try:
            response = get_session().get(url, params=params or {}, timeout=TIMEOUT)
            if response.status_code in (418, 429):
                # Respect Binance's Retry-After header; never hammer a limited endpoint.
                if attempt < 2:
                    wait = response.headers.get("Retry-After", "")
                    try:
                        delay = min(max(float(wait), 1.0), 30.0) if wait else (2 ** attempt) * 2
                    except ValueError:
                        delay = (2 ** attempt) * 2
                    time.sleep(delay)
                    continue
            elif response.status_code >= 500 and attempt < 2:
                time.sleep(1 + attempt)
                continue
            # 451 and other 4xx errors are not retried or bypassed.
            response.raise_for_status()
            try:
                return response.json()
            except ValueError as exc:
                raise RuntimeError(f"Binance returned invalid JSON for {path}") from exc
        except requests.RequestException as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 451 or (status is not None and 400 <= status < 500 and status not in (418, 429)):
                raise
            if attempt < 2:
                time.sleep(1 + attempt)
            else:
                raise
    if last_error:
        raise last_error
    raise RuntimeError(f"Request failed: {path}")


def api_error_message(exc):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status == 451:
        return (
            "Binance returned HTTP 451 (restricted location/network). This is a hosting/network access issue, "
            "not something a code retry can fix. Use a hosting location and data access permitted by Binance, "
            "or contact the hosting provider. The app will not attempt to bypass regional restrictions."
        )
    if status in (418, 429):
        return "Binance rate-limited this server (HTTP %s). Wait before scanning again; do not increase concurrency." % status
    if status:
        body = (getattr(response, "text", "") or "")[:400]
        return f"Binance HTTP {status}: {body}"
    return f"Network/API error: {exc}"


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    result = 100 - (100 / (1 + rs))
    result = result.where(loss.ne(0), 100.0)
    result = result.where(gain.ne(0) | loss.ne(0), 50.0)
    return result.fillna(50.0)


def atr(df, period=14):
    previous_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def fetch_oi_change(symbol):
    """Returns OI change percentage, or None when Binance has no history."""
    try:
        data = api_get("/futures/data/openInterestHist", {
            "symbol": symbol, "period": "1h", "limit": 2
        })
        if isinstance(data, list) and len(data) >= 2:
            old = float(data[-2].get("sumOpenInterest", 0))
            new = float(data[-1].get("sumOpenInterest", 0))
            if old > 0:
                return (new / old - 1) * 100
    except requests.HTTPError as exc:
        # A regional restriction must remain visible; ordinary missing OI history
        # may be skipped, but HTTP 451 is not a missing-data condition.
        if getattr(getattr(exc, "response", None), "status_code", None) == 451:
            raise
        return None
    except (requests.RequestException, RuntimeError, ValueError, TypeError, KeyError):
        return None
    return None


def fetch_symbol(symbol, funding_map, btc_bull):
    raw = api_get("/fapi/v1/klines", {"symbol": symbol, "interval": "1h", "limit": 60})
    if not isinstance(raw, list) or len(raw) < 55:
        return None

    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    frame = pd.DataFrame(raw, columns=columns)
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Use only fully closed hourly candles.
    closed = frame.iloc[:-1].dropna(subset=["high", "low", "close", "quote_volume"]).copy().reset_index(drop=True)
    if len(closed) < 50:
        return None

    closes = closed["close"]
    last = closed.iloc[-1]
    ema20 = float(ema(closes, 20).iloc[-1])
    ema50 = float(ema(closes, 50).iloc[-1])
    average_volume = float(closed["quote_volume"].iloc[-21:-1].mean())
    volume_ratio = float(last["quote_volume"] / average_volume) if average_volume > 0 else 0.0
    rsi_value = float(rsi(closes).iloc[-1])
    oi_change = fetch_oi_change(symbol)
    funding = funding_map.get(symbol)  # None means missing, not a neutral funding rate.

    resistance = float(closed["high"].iloc[-21:-1].max())
    close_price = float(last["close"])
    atr_value = float(atr(closed).iloc[-1])
    if not math.isfinite(atr_value) or atr_value <= 0:
        atr_value = close_price * 0.015

    score = 0
    reasons = []
    if volume_ratio >= 2.0:
        score += 20; reasons.append("Volume spike")
    elif volume_ratio >= 1.5:
        score += 15; reasons.append("Volume rising")
    elif volume_ratio >= 1.2:
        score += 8

    if oi_change is not None:
        if oi_change >= 5:
            score += 20; reasons.append("OI strong")
        elif oi_change >= 2:
            score += 15; reasons.append("OI rising")
        elif oi_change > 0:
            score += 8

    if close_price > ema20 and ema20 > ema50:
        score += 15; reasons.append("EMA bullish")
    elif close_price > ema20:
        score += 8

    if 50 <= rsi_value <= 65:
        score += 10; reasons.append("RSI healthy")
    elif 45 <= rsi_value < 50:
        score += 5

    if resistance > 0 and close_price >= resistance * 0.985:
        score += 10; reasons.append("Near resistance")
    if close_price > resistance and volume_ratio >= 1.5:
        score += 5; reasons.append("Breakout close")

    if funding is not None:
        if -0.0005 <= funding <= 0.0008:
            score += 10
        elif funding < -0.0005:
            score += 7

    if btc_bull is True:
        score += 10; reasons.append("BTC supportive")
    score = min(score, 100)

    entry = close_price
    stop_loss = min(close_price - 1.5 * atr_value,
                    float(closed["low"].iloc[-5:].min()) - 0.15 * atr_value)
    if stop_loss <= 0 or stop_loss >= entry:
        stop_loss = entry - 1.5 * atr_value
    risk = entry - stop_loss
    target1 = entry + 2 * risk
    target2 = entry + 3 * risk
    status = "LONG WATCH" if score >= 65 else "WATCH" if score >= 50 else "SKIP"

    return {
        "Symbol": symbol, "Score": score, "Status": status,
        "Price": close_price, "1H Vol x": round(volume_ratio, 2),
        "OI 1H %": round(oi_change, 2) if oi_change is not None else None,
        "RSI": round(rsi_value, 1),
        "Funding %": round(funding * 100, 4) if funding is not None else None,
        "EMA Trend": "Bullish" if close_price > ema20 and ema20 > ema50 else "Mixed",
        "Resistance": resistance, "Entry ref": entry,
        "Stop Loss": stop_loss, "TP1 (2R)": target1, "TP2 (3R)": target2,
        "Reasons": ", ".join(reasons) if reasons else "No strong confirmation",
    }


@st.cache_data(ttl=60, show_spinner=False)
def get_market_universe():
    info = api_get("/fapi/v1/exchangeInfo")
    if not isinstance(info, dict) or "symbols" not in info:
        raise RuntimeError("Unexpected response from exchangeInfo")
    tradable = {
        item["symbol"] for item in info["symbols"]
        if item.get("status") == "TRADING"
        and item.get("contractType") == "PERPETUAL"
        and item.get("quoteAsset") == "USDT"
    }
    tickers = api_get("/fapi/v1/ticker/24hr")
    rows = []
    for ticker in tickers:
        if ticker.get("symbol") in tradable:
            rows.append({
                "symbol": ticker["symbol"],
                "quoteVolume": float(ticker.get("quoteVolume", 0)),
                "change": float(ticker.get("priceChangePercent", 0)),
                "lastPrice": float(ticker.get("lastPrice", 0)),
            })
    return sorted(rows, key=lambda row: row["quoteVolume"], reverse=True), len(tradable)


@st.cache_data(ttl=60, show_spinner=False)
def get_funding_map():
    data = api_get("/fapi/v1/premiumIndex")
    return {item["symbol"]: float(item["lastFundingRate"])
            for item in data if item.get("lastFundingRate") is not None}


def get_btc_trend():
    candles = api_get("/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 60})
    if not isinstance(candles, list) or len(candles) < 55:
        raise RuntimeError("Not enough BTC hourly candles to calculate trend")
    frame = pd.DataFrame(candles, columns=["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "i"])
    close_series = pd.to_numeric(frame["c"], errors="coerce").iloc[:-1].dropna()
    if len(close_series) < 50:
        raise RuntimeError("Not enough closed BTC hourly candles")
    fast = float(ema(close_series, 20).iloc[-1])
    slow = float(ema(close_series, 50).iloc[-1])
    return float(close_series.iloc[-1]) > fast and fast > slow


def perform_scan(scan_limit, min_quote_volume):
    universe, total_symbols = get_market_universe()
    funding_map = get_funding_map()
    btc_bull = get_btc_trend()
    liquid = [row for row in universe if row["quoteVolume"] >= min_quote_volume][:scan_limit]
    if not liquid:
        raise RuntimeError("No symbols meet the selected minimum 24H quote volume. Lower that filter and scan again.")

    results = []
    failures = []
    progress = st.progress(0, text="Analysing closed 1H candles and open interest...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_symbol, row["symbol"], funding_map, btc_bull): row["symbol"]
            for row in liquid
        }
        for index, future in enumerate(as_completed(future_map), 1):
            symbol = future_map[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except requests.HTTPError as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                failures.append(f"{symbol}: HTTP {status_code or 'error'}")
            except Exception as exc:
                failures.append(f"{symbol}: {type(exc).__name__}")
            progress.progress(index / len(future_map), text=f"Analysed {index}/{len(future_map)}")
    progress.empty()
    # Preserve the most important API status instead of masking it as a generic
    # "no results" message when the symbol-level endpoint is the one blocked.
    if any("HTTP 451" in item for item in failures):
        raise RuntimeError(
            "Binance returned HTTP 451 for symbol market-data requests. "
            "The hosting server/network is restricted; changing retries or scan size will not fix it."
        )
    if not results:
        detail = "; ".join(failures[:5])
        raise RuntimeError("No coin data was successfully analysed. " + detail)
    return results, total_symbols, len(liquid), btc_bull, failures


st.title("📈 Binance Futures — Early Pump Scanner")
st.caption("1H Swing • USDT-margined perpetuals • Public market data • LONG watchlist candidates, not guaranteed predictions")

with st.sidebar:
    st.header("Scanner Settings")
    scan_limit = st.slider("Deep-scan most liquid symbols", 20, 120, 60, 10, key="deep_scan_1")
    min_quote_volume = st.number_input("Minimum 24H quote volume (USDT)", min_value=0, value=2_000_000, step=500_000)
    min_score = st.slider("Minimum score shown", 0, 100, 50, 5)
    st.divider()
    st.subheader("Telegram Alerts")
    bot_token = st.text_input("Telegram Bot Token", type="password", help="Use @BotFather. Do not share this token.")
    chat_id = st.text_input("Telegram Chat ID", help="Start your bot first, then enter its chat ID.")
    st.caption("No Binance API key is needed. Never enter your Binance password or trading API secret.")
    st.divider()
    st.caption(f"Market-data endpoint: {BASE}")

scan_clicked = st.button("🔄 Scan Binance Now", type="primary", use_container_width=True)
if scan_clicked:
    st.session_state.pop("scan_payload", None)
    try:
        with st.spinner("Connecting to Binance USDⓈ-M Futures..."):
            payload = perform_scan(scan_limit, min_quote_volume)
        st.session_state["scan_payload"] = payload
        st.session_state["scan_error"] = None
    except Exception as exc:
        st.session_state["scan_error"] = api_error_message(exc) if isinstance(exc, requests.RequestException) else str(exc)
        st.session_state.pop("scan_payload", None)

if st.session_state.get("scan_error"):
    st.error(st.session_state["scan_error"])
    st.info("If the message says HTTP 451, changing Python code or retrying will not remove the hosting/network restriction. Move the app to a hosting environment permitted to access Binance Futures API, then scan again.")

payload = st.session_state.get("scan_payload")
if payload:
    results, total_symbols, scanned_count, btc_bull, failures = payload
    dataframe = pd.DataFrame(results).sort_values(["Score", "1H Vol x"], ascending=False)
    filtered = dataframe[dataframe["Score"] >= min_score].copy()
    top = filtered.head(10)

    st.info(f"Universe: {total_symbols} USDT perpetual symbols • Deep-scanned: {scanned_count} • BTC 1H: {'Bullish' if btc_bull else 'Not bullish / neutral'} • Successful analyses: {len(dataframe)}")
    oi_missing = int(dataframe["OI 1H %"].isna().sum())
    if oi_missing:
        st.warning(f"Open-interest history was unavailable for {oi_missing} analysed symbols. Those symbols received zero OI points.")
    if failures:
        st.caption(f"{len(failures)} symbol requests failed. First examples: {', '.join(failures[:6])}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Symbols in universe", total_symbols)
    col2.metric("Deep-scanned", scanned_count)
    col3.metric("Candidates ≥ threshold", len(filtered))
    col4.metric("BTC 1H trend", "Bullish" if btc_bull else "Neutral / Bearish")

    display_columns = ["Symbol", "Score", "Status", "Price", "1H Vol x", "OI 1H %", "RSI", "Funding %", "EMA Trend"]
    st.subheader("🏆 Top 10 Early Pump Candidates")
    st.dataframe(top[display_columns], use_container_width=True, hide_index=True)

    st.subheader("Candidate details")
    if not top.empty:
        selected_symbol = st.selectbox("Select a coin", top["Symbol"].tolist())
        selected = top[top["Symbol"] == selected_symbol].iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Pump Score", f'{int(selected["Score"])}/100')
        b.metric("Entry reference", f'{selected["Entry ref"]:.8g}')
        c.metric("Stop Loss", f'{selected["Stop Loss"]:.8g}')
        d.metric("TP1 / TP2", f'{selected["TP1 (2R)"]:.8g} / {selected["TP2 (3R)"]:.8g}')
        st.write("**Reasons:**", selected["Reasons"])
        st.caption("Entry/SL/TP are indicative calculations from the last closed 1H candle and ATR. They are not orders or guaranteed levels. Verify the live chart, liquidity, spread and contract rules before any trade.")
    else:
        st.write("No candidates meet the selected score threshold.")

    st.subheader("Full scan results")
    st.dataframe(filtered[display_columns + ["Resistance", "Entry ref", "Stop Loss", "TP1 (2R)", "TP2 (3R)", "Reasons"]], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download scan as CSV", dataframe.to_csv(index=False).encode("utf-8"), "binance_early_pump_scan.csv", "text/csv")

    if st.button("📩 Send Top 10 to Telegram", use_container_width=True):
        if not bot_token or not chat_id:
            st.error("Enter Telegram Bot Token and Chat ID in the sidebar first.")
        elif top.empty:
            st.warning("No candidates to send.")
        else:
            lines = ["Binance Futures Early Pump Scanner — 1H", "Watchlist only; not trade instructions."]
            for _, row in top.iterrows():
                oi_text = "N/A" if pd.isna(row["OI 1H %"]) else f'{row["OI 1H %"]}%'
                lines.append(f'{row["Symbol"]} | Score {int(row["Score"])}/100 | {row["Status"]} | Vol {row["1H Vol x"]}x | OI {oi_text} | RSI {row["RSI"]}')
            message = "\n".join(lines)
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{bot_token}/sendMessage",
                    data={"chat_id": chat_id, "text": message}, timeout=15
                )
                response.raise_for_status()
                st.success("Telegram message sent.")
            except requests.RequestException as exc:
                st.error(f"Telegram send failed: {exc}")

    st.caption("Scoring weights are heuristic, not statistically validated probabilities. Backtest and paper-trade before risking money.")
else:
    st.write("Press **Scan Binance Now** to fetch the latest public Futures market data.")

st.caption("This app reads public market data only and never places orders. Binance access, API limits and product availability depend on the hosting network and applicable regional rules.")

import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://fapi.binance.com"
TIMEOUT = 12
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EarlyPumpScanner/1.0"})



def api_get(path, params=None):
    r = SESSION.get(BASE + path, params=params or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def fetch_oi_change(symbol):
    try:
        data = api_get("/futures/data/openInterestHist",
                       {"symbol": symbol, "period": "1h", "limit": 2})
        if len(data) >= 2:
            old = float(data[-2]["sumOpenInterest"])
            new = float(data[-1]["sumOpenInterest"])
            if old > 0:
                return (new / old - 1) * 100
    except Exception:
        pass
    return None

def fetch_symbol(symbol, funding_map, btc_bull):
    try:
        raw = api_get("/fapi/v1/klines", {"symbol": symbol, "interval": "1h", "limit": 60})
        if len(raw) < 30:
            return None
        cols = ["open_time","open","high","low","close","volume","close_time",
                "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
        df = pd.DataFrame(raw, columns=cols)
        for c in ["open","high","low","close","volume","quote_volume","taker_buy_quote"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Exclude the currently forming candle; use the last fully closed 1H candle.
        closed = df.iloc[:-1].copy().reset_index(drop=True)
        if len(closed) < 25:
            return None

        c = closed["close"]
        last = closed.iloc[-1]
        prev = closed.iloc[-2]
        e20 = float(ema(c, 20).iloc[-1])
        e50 = float(ema(c, 50).iloc[-1])
        rv = float(last["quote_volume"] / closed["quote_volume"].iloc[-21:-1].mean()) if closed["quote_volume"].iloc[-21:-1].mean() > 0 else 0
        r = float(rsi(c).iloc[-1])
        oi = fetch_oi_change(symbol)
        funding = funding_map.get(symbol, 0.0)

        # Recent 20-bar resistance excludes the signal candle.
        resistance = float(closed["high"].iloc[-21:-1].max())
        close = float(last["close"])
        prior_high = float(prev["high"])
        a = float(atr(closed).iloc[-1])
        if not math.isfinite(a) or a <= 0:
            a = close * 0.015

        score = 0
        notes = []
        if rv >= 2.0: score += 20; notes.append("Volume spike")
        elif rv >= 1.5: score += 15; notes.append("Volume rising")
        elif rv >= 1.2: score += 8
        if oi is not None:
            if oi >= 5: score += 20; notes.append("OI strong")
            elif oi >= 2: score += 15; notes.append("OI rising")
            elif oi > 0: score += 8
        if close > e20 and e20 > e50: score += 15; notes.append("EMA bullish")
        elif close > e20: score += 8
        if 50 <= r <= 65: score += 10; notes.append("RSI healthy")
        elif 45 <= r < 50: score += 5
        if close >= resistance * 0.985: score += 10; notes.append("Near resistance")
        if close > resistance and rv >= 1.5: score += 5; notes.append("Breakout close")
        if -0.0005 <= funding <= 0.0008: score += 10
        elif funding < -0.0005: score += 7
        if btc_bull: score += 10; notes.append("BTC supportive")
        score = min(score, 100)

        # Risk levels are estimates only, based on ATR and recent structure.
        entry = close
        sl = min(close - 1.5*a, float(closed["low"].iloc[-5:].min()) - 0.15*a)
        if sl <= 0 or sl >= entry:
            sl = entry - 1.5*a
        risk = entry - sl
        tp1 = entry + 2*risk
        tp2 = entry + 3*risk
        signal = "LONG WATCH" if score >= 65 else "WATCH" if score >= 50 else "SKIP"

        return {
            "Symbol": symbol, "Score": score, "Status": signal,
            "Price": close, "1H Vol x": round(rv, 2),
            "OI 1H %": round(oi, 2) if oi is not None else None,
            "RSI": round(r, 1), "Funding %": round(funding*100, 4),
            "EMA Trend": "Bullish" if close > e20 and e20 > e50 else "Mixed",
            "Resistance": resistance, "Entry ref": entry,
            "Stop Loss": sl, "TP1 (2R)": tp1, "TP2 (3R)": tp2,
            "Reasons": ", ".join(notes) if notes else "No strong confirmation"
        }
    except Exception:
        return None

@st.cache_data(ttl=45, show_spinner=False)
def get_market_universe():
    info = api_get("/fapi/v1/exchangeInfo")
    symbols = {
        x["symbol"] for x in info["symbols"]
        if x.get("status") == "TRADING"
        and x.get("contractType") == "PERPETUAL"
        and x.get("quoteAsset") == "USDT"
    }
    tickers = api_get("/fapi/v1/ticker/24hr")
    rows = []
    for t in tickers:
        if t["symbol"] in symbols:
            rows.append({
                "symbol": t["symbol"],
                "quoteVolume": float(t.get("quoteVolume", 0)),
                "change": float(t.get("priceChangePercent", 0)),
                "lastPrice": float(t.get("lastPrice", 0))
            })
    return sorted(rows, key=lambda x: x["quoteVolume"], reverse=True), len(symbols)

@st.cache_data(ttl=30, show_spinner=False)
def get_funding_map():
    data = api_get("/fapi/v1/premiumIndex")
    return {x["symbol"]: float(x.get("lastFundingRate", 0)) for x in data}

def get_btc_trend():
    try:
        k = api_get("/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 60})
        df = pd.DataFrame(k, columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","i"])
        c = pd.to_numeric(df["c"]).iloc[:-1]
        return float(c.iloc[-1]) > float(ema(c,20).iloc[-1]) and float(ema(c,20).iloc[-1]) > float(ema(c,50).iloc[-1])
    except Exception:
        return False

st.title("📈 Binance Futures — Early Pump Scanner")
st.caption("1H Swing • All USDT Perpetuals • Public market data • LONG candidates, not guaranteed pump predictions")

with st.sidebar:
    st.header("Scanner Settings")
    scan_limit = st.slider("Deep-scan most liquid symbols", 20, 120, 60, 10, key="deep_scan_2")
    min_quote_vol = st.number_input("Minimum 24H quote volume (USDT)", min_value=0, value=2_000_000, step=500_000)
    min_score = st.slider("Minimum score shown", 0, 100, 50, 5)
    auto_refresh = st.checkbox("Auto-refresh every 5 minutes", value=False)
    st.divider()
    st.subheader("Telegram Alerts")
    bot_token = st.text_input("Telegram Bot Token", type="password", help="Create a bot with @BotFather. Token is used only in this session.")
    chat_id = st.text_input("Telegram Chat ID", help="Send /start to your bot, then obtain your chat ID.")
    st.caption("Do not enter your Binance password, API secret, or trading key.")

if auto_refresh:
    st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

if st.button("🔄 Scan Binance Now", type="primary", use_container_width=True):
    st.cache_data.clear()

try:
    with st.spinner("Loading Binance Futures symbols and market data..."):
        universe, total_symbols = get_market_universe()
        funding_map = get_funding_map()
        btc_bull = get_btc_trend()

    liquid = [x for x in universe if x["quoteVolume"] >= min_quote_vol][:scan_limit]
    st.info(f"Universe: {total_symbols} USDT perpetual symbols • Deep scan: {len(liquid)} liquid symbols • BTC 1H: {'Bullish' if btc_bull else 'Not bullish / neutral'}")

    results = []
    progress = st.progress(0, text="Analysing 1H candles and open interest...")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch_symbol, x["symbol"], funding_map, btc_bull) for x in liquid]
        for i, f in enumerate(as_completed(futures), 1):
            item = f.result()
            if item:
                results.append(item)
            progress.progress(i / max(len(futures), 1), text=f"Analysed {i}/{len(futures)}")
    progress.empty()

    if not results:
        st.warning("No results returned. Binance may be rate-limiting requests or the endpoint may be unavailable. Try again later.")
        st.stop()

    df = pd.DataFrame(results).sort_values(["Score", "1H Vol x"], ascending=False)
    view = df[df["Score"] >= min_score].copy()
    top = view.head(10)

    a,b,c,d = st.columns(4)
    a.metric("Symbols in Binance universe", total_symbols)
    b.metric("Deep-scanned", len(liquid))
    c.metric("Candidates ≥ score threshold", len(view))
    d.metric("BTC 1H trend", "Bullish" if btc_bull else "Neutral / Bearish")

    st.subheader("🏆 Top 10 Early Pump Candidates")
    display_cols = ["Symbol","Score","Status","Price","1H Vol x","OI 1H %","RSI","Funding %","EMA Trend"]
    st.dataframe(top[display_cols], use_container_width=True, hide_index=True)

    st.subheader("Candidate details")
    if not top.empty:
        chosen = st.selectbox("Select a coin", top["Symbol"].tolist())
        row = top[top["Symbol"] == chosen].iloc[0]
        x1,x2,x3,x4 = st.columns(4)
        x1.metric("Pump Score", f'{int(row["Score"])}/100')
        x2.metric("Entry reference", f'{row["Entry ref"]:.8g}')
        x3.metric("Stop Loss", f'{row["Stop Loss"]:.8g}')
        x4.metric("TP1 / TP2", f'{row["TP1 (2R)"]:.8g} / {row["TP2 (3R)"]:.8g}')
        st.write("**Reasons:**", row["Reasons"])
        st.caption("Entry/SL/TP are indicative calculations from the last closed 1H candle and ATR. Confirm the live chart, spread, liquidity, contract filters and invalidation before trading.")
    else:
        st.write("No candidates meet the selected score threshold.")

    st.subheader("Full scan results")
    st.dataframe(view[display_cols + ["Resistance","Entry ref","Stop Loss","TP1 (2R)","TP2 (3R)","Reasons"]],
                 use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download scan as CSV", csv, "binance_early_pump_scan.csv", "text/csv")

    if st.button("📩 Send Top 10 to Telegram", use_container_width=True):
        if not bot_token or not chat_id:
            st.error("Enter Telegram Bot Token and Chat ID in the sidebar first.")
        elif top.empty:
            st.warning("No candidates to send.")
        else:
            lines = ["Binance Futures Early Pump Scanner — 1H", "Candidates are watchlist ideas, not trade instructions."]
            for _, r in top.iterrows():
                lines.append(f'{r["Symbol"]} | Score {int(r["Score"])}/100 | {r["Status"]} | Vol {r["1H Vol x"]}x | OI {r["OI 1H %"]}% | RSI {r["RSI"]}')
            msg = "\n".join(lines)
            try:
                resp = requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
                                     data={"chat_id": chat_id, "text": msg}, timeout=15)
                resp.raise_for_status()
                st.success("Telegram message sent.")
            except Exception as e:
                st.error(f"Telegram send failed: {e}")

    st.caption("Data is refreshed from Binance public Futures endpoints. OI history can be unavailable for some symbols; missing OI receives no OI points. API limits and regional access may affect results.")
except requests.HTTPError as e:
    st.error(f"Binance API HTTP error: {e}. Wait a little and retry; do not increase request frequency.")
except Exception as e:
    st.error(f"Scanner error: {e}")
