import time
import math
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://fapi.binance.com"
TIMEOUT = 12
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EarlyPumpScanner/1.0"})

st.set_page_config(page_title="Binance Early Pump Scanner", page_icon="📈", layout="wide")

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
    scan_limit = st.slider("Deep-scan most liquid symbols", 20, 120, 60, 10)
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
