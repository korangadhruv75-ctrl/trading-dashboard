"""
My Market Desk: MT5 chart + trend arrows + Yahoo facts/backdrop/news + AI chat with one-tap briefings.

Run (Windows, MT5 terminal open and logged in):
    python -m streamlit run trading_dashboard.py

Use a DEMO account while testing. Analysis only, it never places trades.
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from streamlit.errors import StreamlitSecretNotFoundError

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# label -> (MT5 constant name, yahoo interval, yahoo period)
TIMEFRAMES = {
    "M1": ("TIMEFRAME_M1", "1m", "1d"),
    "M5": ("TIMEFRAME_M5", "5m", "5d"),
    "M15": ("TIMEFRAME_M15", "15m", "5d"),
    "H1": ("TIMEFRAME_H1", "1h", "1mo"),
    "H4": ("TIMEFRAME_H4", "1h", "3mo"),  # Yahoo has no 4h, so it is built from 1h candles
    "D1": ("TIMEFRAME_D1", "1d", "1y"),
}

QUICK_SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "BTCUSD", "US30", "NAS100"]

# MT5 symbol -> Yahoo ticker (broker suffixes like .m are ignored)
YAHOO_MAP = {
    "XAUUSD": "GC=F", "XAGUSD": "SI=F",
    "USOIL": "CL=F", "XTIUSD": "CL=F", "UKOIL": "BZ=F", "XBRUSD": "BZ=F",
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD",
    "US30": "^DJI", "NAS100": "^NDX", "USTEC": "^NDX",
    "US500": "^GSPC", "SPX500": "^GSPC",
    "GER40": "^GDAXI", "UK100": "^FTSE", "JP225": "^N225",
}

BACKDROP = {
    "Dollar index": "DX-Y.NYB",
    "US 10Y yield": "^TNX",
    "S&P 500": "^GSPC",
    "VIX (fear)": "^VIX",
    "Gold": "GC=F",
    "Oil (WTI)": "CL=F",
}

# one-tap briefings: button label -> prompt sent to the AI
BRIEFINGS = {
    "⚡ Scalping": (
        "SCALPING briefing. Use the M1, M5 and M15 rows of the timeframe table. Give: bias "
        "(long, short or wait) using the M15 trend as the filter, nearest support and resistance "
        "from the data, a stop and target idea sized from ATR, whether volatility and spread look "
        "OK for scalping, and any news in the next few hours that could spike price. "
        "Max 8 short lines."
    ),
    "📈 Day trading": (
        "DAY TRADING briefing. Focus on the M15, H1 and H4 rows. Give: today's bias, key levels "
        "from the data, two possible setups with entry zone, stop and target using ATR, which "
        "macro or news items matter today (dollar, yields, headlines), and what would cancel the "
        "idea. Max 10 short lines."
    ),
    "🌊 Swing": (
        "SWING briefing (days to weeks). Focus on the H4 and D1 rows and the market backdrop. "
        "Give: the bigger trend and bias, major support and resistance from the data, where a "
        "pullback entry might be, how the dollar, yields and risk mood (VIX, S&P) affect this "
        "instrument, key risks, and what would invalidate the view. Max 10 short lines."
    ),
    "📰 News & macro": (
        "NEWS and MACRO check. Using only the headlines and the backdrop table, summarise what is "
        "moving this market, whether it is a headwind or tailwind for the chart bias, and any "
        "risk of volatility spikes. If the headlines are not relevant to this instrument, say so. "
        "Max 6 short lines."
    ),
    "🎯 Bull or bear": (
        "Quick call: bull or bear right now across all timeframes? Answer in 3 short lines: "
        "bias, strongest reason, biggest risk."
    ),
}

SYSTEM_PROMPT = (
    "You are a sharp, concise market assistant for a trader. You are given LIVE DATA from the "
    "trader's screen: chart indicators for several timeframes, Yahoo facts, a wider market "
    "backdrop and news headlines. Use ONLY that data for facts about prices, levels and news. If "
    "something is not in the data, say you don't have it, never invent numbers. Combine the chart "
    "with the backdrop and the news (for example dollar strength, yields, risk mood). Keep answers "
    "short and practical. You are not a financial advisor. If the trader shares a screenshot of "
    "their trades or a chart, read it carefully: say what you can see (instrument, entries, exits, "
    "stops, targets, profit or loss), judge the decisions and the risk management, point out "
    "mistakes and what to improve. Only state what is visible and say if something is unreadable."
)

DEFAULT_IMAGE_PROMPT = (
    "Review this screenshot I am sharing (my trades or my chart). Tell me what you see, what was "
    "done well, what went wrong, and what to improve. Be specific and concise."
)
# models that can read images, tried in order (the free list on NVIDIA changes often)
VISION_MODELS = [
    "meta/muse-glimmer-30b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
]
# models are tried in this order until one answers (the free list on NVIDIA changes often)
MODELS = [
    "deepseek-ai/deepseek-v4.1-flash",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3-ultra-550b-a55b",
]
KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvidia_key.txt")
# no key is stored in the code: put it in a file named nvidia_key.txt next to this script
DEFAULT_KEY = ""


# ---------- helpers ----------
def guess_yahoo(symbol):
    s = "".join(ch for ch in symbol.upper() if ch.isalnum())
    for key, ticker in YAHOO_MAP.items():
        if s.startswith(key):
            return ticker
    if len(s) >= 6 and s[:6].isalpha():
        return s[:6] + "=X"  # forex pair, e.g. EURUSD -> EURUSD=X
    return symbol  # stocks, e.g. AAPL


def load_key():
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if key:
        return key
    try:
        key = st.secrets.get("NVIDIA_API_KEY", "").strip()
        if key:
            return key
    except StreamlitSecretNotFoundError:
        pass
    try:
        with open(KEY_FILE, encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key
    except OSError:
        pass
    return DEFAULT_KEY


def arrow_md(call):
    if call == "BULLISH":
        return ":green[▲ BULL]"
    if call == "BEARISH":
        return ":red[▼ BEAR]"
    return ":gray[► MIXED]"


# ---------- data ----------
def get_mt5_candles(symbol, tf_label, count):
    if mt5 is None or not mt5.initialize():
        return None, None
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, TIMEFRAMES[tf_label][0]), 0, count)
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if rates is None or len(rates) == 0:
        return None, None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    live = {}
    if tick:
        live = {"bid": tick.bid, "ask": tick.ask}
        if info:
            live["spread_points"] = info.spread
    return df, live


@st.cache_data(ttl=120, show_spinner=False)
def get_yahoo_candles(ticker, tf_label):
    _, interval, period = TIMEFRAMES[tf_label]
    df = yf.Ticker(ticker).history(period=period, interval=interval).reset_index()
    if df.empty:
        return None
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "time"})
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df[["time", "open", "high", "low", "close"]]
    if tf_label == "H4":
        df = (
            df.set_index("time")
            .resample("4h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
            .reset_index()
        )
    return df


def add_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
    prev = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def verdict(df):
    last = df.iloc[-1]
    score = 0
    score += 1 if last.close > last.ema20 else -1
    score += 1 if last.ema20 > last.ema50 else -1
    score += 1 if last.rsi > 50 else -1
    if score >= 2:
        return "BULLISH", score
    if score <= -2:
        return "BEARISH", score
    return "SIDEWAYS / MIXED", score


def trend_by_timeframe(symbol, yahoo_ticker):
    rows = {}
    for label in TIMEFRAMES:
        d, _ = get_mt5_candles(symbol, label, 200)
        if d is None:
            d = get_yahoo_candles(yahoo_ticker, label)
        if d is None or len(d) < 60:
            rows[label] = None
            continue
        d = add_indicators(d)
        call, score = verdict(d)
        rows[label] = {
            "call": call,
            "score": score,
            "rsi": float(d["rsi"].iloc[-1]),
            "atr": float(d["atr"].iloc[-1]),
            "high": float(d["high"].tail(50).max()),
            "low": float(d["low"].tail(50).min()),
        }
    return rows


@st.cache_data(ttl=300, show_spinner=False)
def yahoo_info(ticker):
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def market_backdrop():
    rows = []
    for name, tk in BACKDROP.items():
        try:
            h = yf.Ticker(tk).history(period="5d")["Close"].dropna()
            if len(h) >= 2:
                rows.append({
                    "Market": name,
                    "Last": round(float(h.iloc[-1]), 3),
                    "Change %": round(float(h.iloc[-1] / h.iloc[-2] - 1) * 100, 2),
                })
        except Exception:
            pass
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def get_news(tickers):
    seen, out = set(), []
    for tk in tickers:
        try:
            items = yf.Ticker(tk).news or []
        except Exception:
            items = []
        for item in items[:6]:
            c = item.get("content", item)
            title = c.get("title")
            if not title or title in seen:
                continue
            seen.add(title)
            pub = (c.get("provider") or {}).get("displayName") or c.get("publisher") or ""
            out.append(f"{title} ({pub})" if pub else title)
    return out[:14]


# ---------- images ----------
def prepare_image(uploaded):
    """Shrink a screenshot so it fits NVIDIA's inline image size limit."""
    import io

    from PIL import Image

    img = Image.open(uploaded).convert("RGB")
    side, quality = 1280, 80
    while True:
        im = img.copy()
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) * 4 / 3 < 170_000 or side <= 400:
            return data
        side = int(side * 0.8)
        quality = max(50, quality - 5)


def split_input(v):
    """Works whether chat_input returned text only, or text plus attached files."""
    if v is None:
        return "", []
    if isinstance(v, str):
        return v, []
    text = getattr(v, "text", None)
    files = getattr(v, "files", None)
    if text is None and isinstance(v, dict):
        text = v.get("text")
    if files is None and isinstance(v, dict):
        files = v.get("files")
    return text or "", list(files or [])


# ---------- AI (NVIDIA, OpenAI-compatible) ----------
def ask_ai(api_key, model, context, history, images=None):
    from openai import OpenAI

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=40,
        max_retries=0,
    )
    clean = [{"role": m["role"], "content": m["content"]} for m in history if m.get("content")][-10:]
    if images and clean:
        import base64

        parts = [{"type": "text", "text": clean[-1]["content"]}]
        for raw in images:
            b64 = base64.b64encode(raw).decode()
            parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}})
        clean[-1] = {"role": clean[-1]["role"], "content": parts}
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nLIVE DATA:\n" + context}] + clean
    kwargs = dict(model=model, messages=messages, temperature=0.3, max_tokens=1500)
    try:
        resp = client.chat.completions.create(
            **kwargs, extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
    except Exception as e:
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            raise
        resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    text = msg.content
    if not text:
        text = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if not text:
        raise RuntimeError("empty answer")
    return text


def ask_with_fallback(api_key, models, context, history, images=None):
    errors = []
    for m in models:
        try:
            return ask_ai(api_key, m, context, history, images), m
        except Exception as e:
            errors.append(f"{m}: {e}")
    raise RuntimeError(" | ".join(errors))


# ---------- callbacks ----------
def pick_symbol(s):
    st.session_state.symbol = s


def queue_briefing(label, text):
    st.session_state.pending = (label, text)


def main():
    st.set_page_config(page_title="My Market Desk", layout="wide")

    # ---------- sidebar ----------
    st.session_state.setdefault("symbol", "XAUUSD")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("My Market Desk")

    with st.sidebar:
        st.header("Settings")
        mt5_symbol = st.text_input("MT5 symbol", key="symbol")
        st.caption("Quick pick (the name must match your broker's symbol):")
        qcols = st.columns(2)
        for i, s in enumerate(QUICK_SYMBOLS):
            qcols[i % 2].button(s, key=f"q_{s}", on_click=pick_symbol, args=(s,))
        yahoo_override = st.text_input(
            "Yahoo ticker (optional)",
            "",
            help="Leave empty to auto-detect from the MT5 symbol. Examples: GC=F, EURUSD=X, ^GSPC, AAPL",
        )
        yahoo_ticker = yahoo_override.strip() or guess_yahoo(mt5_symbol)
        st.caption(f"Using Yahoo ticker: {yahoo_ticker}")
        tf = st.selectbox("Chart timeframe", list(TIMEFRAMES), index=1)
        count = st.slider("Candles", 100, 1000, 300, 50)
        if st.button("Refresh"):
            st.cache_data.clear()
        st.divider()
        st.header("AI chat")
        with st.expander("Advanced (optional)"):
            key_override = st.text_input("Use a different NVIDIA key", type="password")
            model_override = st.text_input(
                "Use a specific model", "", help="Leave empty and the app picks a working model by itself."
            )
        api_key = key_override.strip() or load_key()
        models = [model_override.strip()] if model_override.strip() else MODELS
        st.caption("NVIDIA key loaded" if api_key else "No NVIDIA key found")
        if st.button("Clear chat"):
            st.session_state.messages = []

    # ---------- data ----------
    df, live = get_mt5_candles(mt5_symbol, tf, count)
    source = "MT5"
    if df is None:
        df = get_yahoo_candles(yahoo_ticker, tf)
        live = {}
        source = "Yahoo (MT5 not connected)"

    if df is None or len(df) < 60:
        st.error("No chart data. Check that MT5 is open and logged in, and that the symbol names are right.")
        st.stop()

    df = add_indicators(df)
    last = df.iloc[-1]
    call, score = verdict(df)
    mtf = trend_by_timeframe(mt5_symbol, yahoo_ticker)

    info = yahoo_info(yahoo_ticker)
    backdrop = market_backdrop()
    news_tickers = tuple(dict.fromkeys([yahoo_ticker, "DX-Y.NYB", "^GSPC"]))
    headlines = get_news(news_tickers)

    facts = {
        "Name": info.get("shortName") or info.get("longName"),
        "Previous close": info.get("previousClose"),
        "Day low": info.get("dayLow"),
        "Day high": info.get("dayHigh"),
        "Volume": info.get("volume"),
        "52w low": info.get("fiftyTwoWeekLow"),
        "52w high": info.get("fiftyTwoWeekHigh"),
        "Market cap": info.get("marketCap"),
        "P/E": info.get("trailingPE"),
    }
    facts = {k: v for k, v in facts.items() if v is not None}
    if live:
        facts["MT5 bid"] = live.get("bid")
        facts["MT5 ask"] = live.get("ask")
        facts["Spread (points)"] = live.get("spread_points")

    tf_lines = []
    for label, r in mtf.items():
        if r:
            tf_lines.append(
                f"{label}: {r['call']}, RSI {r['rsi']:.0f}, ATR {r['atr']:.5g}, "
                f"50-bar high {r['high']:.5g}, low {r['low']:.5g}"
            )
        else:
            tf_lines.append(f"{label}: no data")
    backdrop_text = backdrop.to_string(index=False) if not backdrop.empty else "not available"
    context = (
        f"Symbol {mt5_symbol} (Yahoo {yahoo_ticker}), chart timeframe {tf}, chart source {source}\n"
        f"Last close {last.close}, EMA20 {last.ema20:.5g}, EMA50 {last.ema50:.5g}, "
        f"RSI {last.rsi:.1f}, ATR {last.atr:.5g}\n"
        f"Rule-based call on {tf}: {call}\n"
        f"Trend by timeframe:\n" + "\n".join(tf_lines) + "\n"
        f"Live quote: {live}\nYahoo facts for {yahoo_ticker}: {facts}\n"
        f"Market backdrop (last close, daily change %):\n{backdrop_text}\n"
        f"News headlines:\n" + "\n".join("- " + h for h in headlines) + "\n"
        f"Last 15 candles on {tf}:\n{df[['time','open','high','low','close']].tail(15).to_string(index=False)}"
    )

    # ---------- top row: arrows ----------
    ref = df["close"].iloc[-13] if len(df) > 13 else df["close"].iloc[0]
    chg = float(last.close - ref)
    pct = chg / float(ref) * 100
    icon = {"BULLISH": "▲", "BEARISH": "▼"}.get(call, "►")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend call", f"{icon} {call}")
    c2.metric("Price", f"{last.close:.5g}", delta=f"{chg:+.5g} ({pct:+.2f}%)", help="Change over the last 12 candles")
    c3.metric("RSI (14)", f"{last.rsi:.1f}")
    c4.metric("ATR (14)", f"{last.atr:.5g}")
    st.caption(f"{mt5_symbol} | chart source: {source} | rule-based call from EMA20/EMA50 and RSI, score {score} of 3")

    st.caption("Trend by timeframe")
    tcols = st.columns(len(TIMEFRAMES))
    for col, label in zip(tcols, TIMEFRAMES):
        r = mtf.get(label)
        col.markdown(f"**{label}**  \n{arrow_md(r['call']) if r else 'n/a'}")

    # ---------- chart (left) + AI (right) ----------
    chart_col, chat_col = st.columns([2.2, 1.2])

    with chart_col:
        fig = go.Figure()
        fig.add_candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price")
        fig.add_scatter(x=df["time"], y=df["ema20"], name="EMA20", line=dict(width=1.2))
        fig.add_scatter(x=df["time"], y=df["ema50"], name="EMA50", line=dict(width=1.2))
        fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
        lo, hi = df["low"].quantile(0.01), df["high"].quantile(0.99)
        pad = (hi - lo) * 0.1
        fig.update_yaxes(range=[lo - pad, hi + pad])
        st.plotly_chart(fig, width="stretch")

    with chat_col:
        st.subheader("AI assistant")
        st.caption("One-tap briefings:")
        bcols = st.columns(2)
        for i, (label, text) in enumerate(BRIEFINGS.items()):
            bcols[i % 2].button(label, key=f"b_{i}", on_click=queue_briefing, args=(label, text))
        box = st.container(height=400)
        try:
            typed = st.chat_input(
                "Ask anything, or attach / paste a screenshot of your trades",
                accept_file=True,
                file_type=["png", "jpg", "jpeg", "webp"],
            )
        except TypeError:
            typed = st.chat_input("Ask anything, e.g. where are the key levels?")

        pending = st.session_state.pop("pending", None)
        typed_text, typed_files = split_input(typed)
        q_images = []
        if typed_text or typed_files:
            q_label, q_text = None, typed_text or DEFAULT_IMAGE_PROMPT
            q_images = [prepare_image(f) for f in typed_files]
        elif pending:
            q_label, q_text = pending
        else:
            q_label, q_text = None, None

        with box:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    for raw in m.get("images", []):
                        st.image(raw, width=280)
                    st.write(m.get("label") or m["content"])
            if q_text:
                st.session_state.messages.append(
                    {"role": "user", "content": q_text, "label": q_label, "images": q_images}
                )
                use_models = ([model_override.strip()] if model_override.strip() else VISION_MODELS) if q_images else models
                with st.chat_message("user"):
                    for raw in q_images:
                        st.image(raw, width=280)
                    st.write(q_label or q_text)
                with st.chat_message("assistant"):
                    if not api_key:
                        reply = (
                            "No NVIDIA key found. Create a file called nvidia_key.txt next to this script, "
                            "paste your key inside, save, then press F5. (Or paste it under Advanced in the sidebar.)"
                        )
                    else:
                        try:
                            with st.spinner("Thinking..."):
                                reply, used = ask_with_fallback(
                                    api_key, use_models, context, st.session_state.messages, q_images
                                )
                            reply = f"{reply}\n\n*Answered by {used}*"
                        except Exception as e:
                            reply = f"The AI call failed on every model: {e}"
                    st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

    # ---------- facts, backdrop, news ----------
    left, right = st.columns(2)
    with left:
        st.subheader(f"Market facts ({yahoo_ticker})")
        st.table(pd.DataFrame({"Value": facts}).astype(str))
        st.subheader("Market backdrop")
        if backdrop.empty:
            st.write("Backdrop data not available right now.")
        else:
            st.dataframe(backdrop, hide_index=True)
    with right:
        st.subheader("Latest news")
        if headlines:
            for h in headlines:
                st.write("- " + h)
        else:
            st.write("No headlines found.")


if __name__ == "__main__":
    main()
