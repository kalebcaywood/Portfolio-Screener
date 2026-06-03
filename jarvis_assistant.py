"""Jarvis — in-app AI assistant for the Quantitative Portfolio Analytics app.

Renders a floating circular button (bottom-right) that opens a chat panel.
Any authenticated user can ask questions; Jarvis can read the live trading-bot
data and search the web. Shares the same "brain" (Claude) as the desktop Jarvis
orb, but runs in-process here.

Config: needs ANTHROPIC_API_KEY in .streamlit/secrets.toml or the environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

MODEL = "claude-sonnet-4-6"
BASE = Path(__file__).resolve().parent
BOT = BASE / "trading_bot"

SYSTEM = (
    "You are Jarvis, the built-in assistant for this Quantitative Portfolio Analytics "
    "workbench (a University of Tennessee equity/portfolio research app). Be concise, "
    "precise, and helpful. Plain text, no markdown headers.\n"
    "- You can read the user's live trading-bot portfolio and signals with the tools. "
    "Use them for any question about positions, P&L, buying power, recent trades, or a ticker.\n"
    "- You can search the web for current market data, prices, and news.\n"
    "- You are NOT a licensed financial advisor: share data and observations, explain the "
    "app's analytics, but avoid hard buy/sell directives.\n"
    "- If asked how to use a feature, explain the relevant page (Screener, Performance, Risk "
    "Metrics, Factor Models, Optimization, Monte Carlo, Stress Tests, Pacing & Reup, etc.)."
)


# ─── Key ─────────────────────────────────────────────────────────────────────
def _get_key() -> str:
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return str(st.secrets["ANTHROPIC_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


# ─── Equity data (read-only) ─────────────────────────────────────────────────
def _read(name: str):
    try:
        return json.loads((BOT / name).read_text(encoding="utf-8"))
    except Exception:
        return None


def _round(x, d=2):
    try:
        return round(float(x), d)
    except (TypeError, ValueError):
        return x


def portfolio_status():
    state = _read("state.json")
    snap = _read("account_snapshot.json")
    if not state and not snap:
        return {"ok": False, "error": "No trading data found."}
    quotes = (snap or {}).get("quotes", {})
    positions = (state or {}).get("positions", {})
    rows, total_cost, total_val, have_val = [], 0.0, 0.0, False
    for sym, p in positions.items():
        price = quotes.get(sym, {}).get("price")
        cost = (p.get("shares") or 0) * (p.get("entry_price") or 0)
        value = (p.get("shares") or 0) * price if price is not None else None
        total_cost += cost
        if value is not None:
            total_val += value
            have_val = True
        rows.append({
            "symbol": sym, "shares": _round(p.get("shares"), 4), "entry": p.get("entry_price"),
            "last": price, "stop": p.get("stop_price"), "target": p.get("target_price"),
            "unrealized": _round(value - cost) if value is not None else None,
            "pct_change": _round((price / p["entry_price"] - 1) * 100, 2) if price and p.get("entry_price") else None,
            "buy_date": p.get("buy_date"),
        })
    return {
        "ok": True, "buying_power": (snap or {}).get("buying_power"),
        "position_count": len(rows), "positions": rows,
        "invested_cost": _round(total_cost), "market_value": _round(total_val) if have_val else None,
        "total_unrealized": _round(total_val - total_cost) if have_val else None,
        "daily": (state or {}).get("daily"), "lifetime": (state or {}).get("lifetime"),
    }


def recent_trades(count=5):
    try:
        lines = (BOT / "trades.jsonl").read_text(encoding="utf-8").strip().splitlines()
        last = [json.loads(l) for l in lines[-max(1, count):] if l.strip()]
        return {"ok": True, "count": len(last), "trades": [
            {"time": t.get("ts"), "action": t.get("action"), "symbol": t.get("symbol"),
             "price": t.get("price"), "shares": _round(t.get("shares"), 4), "notional": t.get("notional")}
            for t in last]}
    except Exception:
        return {"ok": False, "error": "No trade log found."}


def signals():
    ns = _read("news_signals.json") or {}
    ao = _read("approved_orders.json") or {}
    evals = ao.get("evaluations", [])
    actionable = [{"symbol": e.get("symbol"), "action": e.get("action"), "reason": e.get("reason")}
                  for e in evals if e.get("action") and e.get("action") != "NONE"]
    return {"ok": True, "as_of": ao.get("timestamp") or ns.get("as_of"),
            "buying_power": ao.get("buying_power"), "symbols_evaluated": len(evals),
            "actionable": actionable,
            "news": {"enabled": ns.get("enabled"), "positive": ns.get("positive_symbols", []),
                     "blocked": ns.get("blocked_symbols", []), "watchlist": ns.get("watchlist", []),
                     "buy_candidates": ns.get("autonomous_buy_candidates", [])}}


def analyze_ticker(symbol: str):
    sym = str(symbol or "").upper().strip()
    if not sym:
        return {"ok": False, "error": "No symbol given."}
    ao = _read("approved_orders.json") or {}
    snap = _read("account_snapshot.json") or {}
    state = _read("state.json") or {}
    ev = next((e for e in ao.get("evaluations", []) if str(e.get("symbol", "")).upper() == sym), None)
    held = (state.get("positions") or {}).get(sym)
    price = (snap.get("quotes") or {}).get(sym, {}).get("price")
    if not ev and not held and price is None:
        return {"ok": False, "error": f"No data for {sym} in the screener."}
    return {"ok": True, "symbol": sym, "last": price,
            "held": {"entry": held["entry_price"], "stop": held["stop_price"],
                     "target": held["target_price"], "shares": _round(held["shares"], 4)} if held else None,
            "evaluation": {"action": ev["action"], "reason": ev["reason"], "indicators": ev.get("indicators")} if ev else None}


TOOLS = [
    {"name": "get_portfolio_status", "description": "Current trading portfolio: open positions with entry/stop/target/last, unrealized P&L, buying power, daily/lifetime stats.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_recent_trades", "description": "Most recent trades the bot executed.",
     "input_schema": {"type": "object", "properties": {"count": {"type": "integer"}}}},
    {"name": "get_signals", "description": "Current bot signals: actionable symbols with reasons, plus news positive/blocked/watchlist.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "analyze_ticker", "description": "Screener view of one ticker: last price, holding info, latest evaluation/indicators.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 4},
]


def _run_tool(name, inp):
    if name == "get_portfolio_status":
        return portfolio_status()
    if name == "get_recent_trades":
        return recent_trades(inp.get("count", 5))
    if name == "get_signals":
        return signals()
    if name == "analyze_ticker":
        return analyze_ticker(inp.get("symbol", ""))
    return {"ok": False, "error": f"Unknown tool {name}"}


def ask_jarvis(history):
    """history: list of {role, content} (text). Returns assistant reply text."""
    import anthropic
    key = _get_key()
    if not key:
        return "I'm not configured yet — add ANTHROPIC_API_KEY to .streamlit/secrets.toml."
    client = anthropic.Anthropic(api_key=key)
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    try:
        for _ in range(8):
            r = client.messages.create(model=MODEL, max_tokens=700, system=SYSTEM, tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": r.content})
            if r.stop_reason == "pause_turn":
                continue
            if r.stop_reason == "tool_use":
                results = []
                for b in r.content:
                    if getattr(b, "type", None) == "tool_use":
                        out = _run_tool(b.name, dict(b.input or {}))
                        results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out)})
                if results:
                    messages.append({"role": "user", "content": results})
                    continue
            text = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", None) == "text").strip()
            return text or "…"
        return "Sorry, that got complicated — try rephrasing?"
    except Exception as e:
        return f"I hit an error reaching my brain: {e}"


# ─── UI ──────────────────────────────────────────────────────────────────────
def _inject_css():
    st.markdown(
        """
        <style>
        .st-key-jarvis_fab { position: fixed; bottom: 26px; right: 26px; z-index: 1000; width: 64px; }
        .st-key-jarvis_fab button {
            width: 64px; height: 64px; border-radius: 50% !important; padding: 0 !important;
            font-size: 26px; font-weight: 700; color: #fff !important;
            background: radial-gradient(circle at 35% 30%, #ff6b6b, #c81e1e) !important;
            border: 2px solid rgba(255,90,90,0.65) !important;
            box-shadow: 0 0 22px rgba(255,40,40,0.55), 0 4px 14px rgba(0,0,0,0.4) !important;
            transition: transform .15s, box-shadow .15s;
        }
        .st-key-jarvis_fab button:hover {
            transform: scale(1.07);
            box-shadow: 0 0 30px rgba(255,40,40,0.8), 0 4px 16px rgba(0,0,0,0.5) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _submit():
    q = (st.session_state.get("jarvis_input") or "").strip()
    if not q:
        return
    st.session_state.jarvis_msgs.append({"role": "user", "content": q})
    reply = ask_jarvis(st.session_state.jarvis_msgs)
    st.session_state.jarvis_msgs.append({"role": "assistant", "content": reply})


def render_jarvis():
    """Call once per page (e.g. at the end of app.py)."""
    _inject_css()
    st.session_state.setdefault("jarvis_msgs", [])

    with st.container(key="jarvis_fab"):
        with st.popover("◉", use_container_width=False):
            st.markdown("##### ◉ Jarvis")
            st.caption("Ask about your portfolio, a ticker, the app, or the markets.")

            if not st.session_state.jarvis_msgs:
                st.info("Try: \"How's my portfolio?\", \"Analyze NKE\", or \"What moved the market today?\"")
            for m in st.session_state.jarvis_msgs[-12:]:
                with st.chat_message("user" if m["role"] == "user" else "assistant"):
                    st.write(m["content"])

            with st.form("jarvis_form", clear_on_submit=True):
                st.text_input("Ask Jarvis", key="jarvis_input",
                              label_visibility="collapsed", placeholder="Ask Jarvis anything…")
                c1, c2 = st.columns([3, 1])
                c1.form_submit_button("Send", use_container_width=True, type="primary", on_click=_submit)
                if c2.form_submit_button("Clear", use_container_width=True):
                    st.session_state.jarvis_msgs = []
                    st.rerun()
