"""Planning entrypoint — turns market data + account state into vetted orders.

Run by the scheduled agent each cycle:

    python -m trading_bot.plan --snapshot account_snapshot.json --out approved_orders.json

Pipeline:
  1. load config + persistent state
  2. download daily OHLC for the universe (yfinance)
  3. for each symbol: strategy.evaluate(holding=?, entry, stop)  -> BUY/SELL/HOLD/NONE
  4. feed candidates + live account snapshot through risk_guard.vet()
  5. write approved_orders.json (the ONLY thing the agent is allowed to execute)

The account snapshot is supplied by the agent from the Robinhood MCP (live
buying power, real positions, real-time quotes). If --snapshot is omitted, a
synthetic snapshot is built from state.json + yfinance closes so the planner can
be exercised offline (dry-run / testing) without the broker.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from . import journal, news, risk_guard, strategy

ET = ZoneInfo("America/New_York")
HERE = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(HERE, "config.json")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_ohlc(symbols: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Return {symbol: DataFrame[High, Low, Close]} of daily bars."""
    period = f"{max(lookback_days, 120) + 10}d"
    raw = yf.download(
        symbols, period=period, interval="1d", auto_adjust=True,
        progress=False, group_by="ticker", threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    multi = isinstance(raw.columns, pd.MultiIndex)
    for sym in symbols:
        try:
            df = raw[sym] if multi else raw
            sub = df[["High", "Low", "Close"]].dropna()
            if not sub.empty:
                out[sym] = sub
        except (KeyError, Exception):
            continue
    return out


def build_synthetic_snapshot(config: dict, state: dict, ohlc: dict, buying_power: float) -> dict:
    """Offline snapshot: positions from state, quotes = last close."""
    positions = {}
    for sym, meta in state.get("positions", {}).items():
        shares = float(meta.get("shares", 0.0))
        price = float(ohlc[sym]["Close"].iloc[-1]) if sym in ohlc else float(meta.get("entry_price", 0.0))
        positions[sym] = {
            "shares": shares,
            "shares_available_for_sells": shares,
            "market_value": round(shares * price, 2),
            "average_buy_price": float(meta.get("entry_price", price)),
        }
    quotes = {sym: {"price": float(df["Close"].iloc[-1])} for sym, df in ohlc.items()}
    return {"buying_power": buying_power, "positions": positions, "quotes": quotes}


def make_plan(config: dict, state: dict, snapshot: dict, now_et: datetime | None = None) -> dict:
    now_et = now_et or datetime.now(ET)
    params = config["strategy"]
    universe = config["universe"]
    held_syms = set(snapshot.get("positions", {}).keys())

    ohlc = fetch_ohlc(universe + list(held_syms - set(universe)), params["lookback_days"])

    evaluations = []
    buys, sells = [], []
    for sym in sorted(set(universe) | held_syms):
        df = ohlc.get(sym)
        holding = sym in held_syms
        meta = state.get("positions", {}).get(sym, {})
        res = strategy.evaluate(
            df, params, holding=holding,
            entry_price=meta.get("entry_price"),
            stop_price=meta.get("stop_price"),
            target_price=meta.get("target_price"),
        )
        res["symbol"] = sym
        res["holding"] = holding
        evaluations.append(res)
        if res["action"] == "BUY":
            buys.append(res)
        elif res["action"] == "SELL":
            sells.append({**res, "shares_available_for_sells":
                          snapshot["positions"].get(sym, {}).get("shares_available_for_sells", 0.0)})

    # rank entry candidates best-first by reward:risk so the daily buy cap and
    # capital headroom go to the highest-quality setups
    buys.sort(key=lambda b: b.get("reward_risk", 0.0), reverse=True)

    news_signals = news.load_signals(config)

    # In autonomous news mode, positive non-held in-universe names that produced
    # no technical BUY can still become candidates (still fully risk-gated below).
    if news_signals and config.get("news", {}).get("mode") == "autonomous":
        existing = {b["symbol"] for b in buys}
        for sym in news_signals.get("autonomous_buy_candidates", []):
            if sym not in existing and sym not in held_syms:
                buys.append({"symbol": sym, "suggested_stop": None,
                             "reason": "news-driven (autonomous mode)", "indicators": {}})

    vetted = risk_guard.vet(
        buys=buys, sells=sells, snapshot=snapshot, state=state,
        config=config, now_et=now_et, news=news_signals,
    )

    return {
        "timestamp": now_et.isoformat(),
        "account_number": config["account_number"],
        "dry_run": config["dry_run"],
        "enabled": config["enabled"],
        "buying_power": snapshot.get("buying_power"),
        "evaluations": evaluations,
        "approved_buys": vetted["approved_buys"],
        "approved_sells": vetted["approved_sells"],
        "blocked": vetted["blocked"],
        "audit": vetted["audit"],
        "news": None if not news_signals else {
            "mode": news_signals.get("mode"),
            "blocked_symbols": news_signals.get("blocked_symbols", []),
            "positive_symbols": news_signals.get("positive_symbols", []),
            "watchlist": news_signals.get("watchlist", []),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="trading_bot planner")
    p.add_argument("--snapshot", default=None, help="account snapshot JSON from MCP")
    p.add_argument("--out", default=os.path.join(HERE, "approved_orders.json"))
    p.add_argument("--buying-power", type=float, default=None,
                   help="override buying power for synthetic snapshot")
    args = p.parse_args()

    config = load_config()
    state = journal.load_state()

    if args.snapshot:
        with open(args.snapshot, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
    else:
        bp = args.buying_power if args.buying_power is not None else 113.06
        ohlc = fetch_ohlc(config["universe"] + list(state.get("positions", {}).keys()),
                          config["strategy"]["lookback_days"])
        snapshot = build_synthetic_snapshot(config, state, ohlc, bp)

    plan = make_plan(config, state, snapshot)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    # human summary to stdout
    print(f"=== trading_bot plan @ {plan['timestamp']} ===")
    print(f"account {plan['account_number']}  dry_run={plan['dry_run']}  "
          f"enabled={plan['enabled']}  buying_power={plan['buying_power']}")
    if plan["blocked"]:
        print("!! WRITES BLOCKED this cycle")
    print(f"approved buys : {len(plan['approved_buys'])}")
    for b in plan["approved_buys"]:
        print(f"   BUY  {b['symbol']:5} ${b['notional']:.2f}  stop {b['stop_price']}  "
              f"target {b.get('target_price')}  R:R {b.get('reward_risk')}  — {b['reason']}")
    print(f"approved sells: {len(plan['approved_sells'])}")
    for s in plan["approved_sells"]:
        print(f"   SELL {s['symbol']:5}  — {s['reason']}")
    if plan.get("news"):
        n = plan["news"]
        print(f"--- news (mode={n['mode']}) ---")
        print(f"  blocked (negative): {n['blocked_symbols'] or '-'}")
        print(f"  positive          : {n['positive_symbols'] or '-'}")
        if n["watchlist"]:
            print("  watchlist (mentioned, NOT in universe — alert only):")
            for w in n["watchlist"]:
                tag = " [social-only]" if w.get("social_only") else ""
                print(f"     {w['symbol']:6} score {w['score']:+d}{tag}  — {w.get('sample','')[:60]}")
    print("--- audit ---")
    for line in plan["audit"]:
        print("  " + line)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
