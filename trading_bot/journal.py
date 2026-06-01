"""Persistent bot state + append-only trade journal.

state.json   -> the bot's memory: per-symbol entry price / trailing stop / buy
                date, and daily counters used by the risk guard.
trades.jsonl -> immutable audit log, one JSON object per executed (or simulated)
                fill. Never rewritten, only appended.

The broker (via MCP) is the source of truth for *what we hold*. This file holds
the *metadata* the broker doesn't track for us: our stop levels, entry basis for
take-profit math, and the daily circuit-breaker counters.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HERE = os.path.dirname(__file__)
STATE_PATH = os.path.join(HERE, "state.json")
TRADES_PATH = os.path.join(HERE, "trades.jsonl")


def _today() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(ET).isoformat()


def default_state() -> dict:
    return {
        "positions": {},  # SYM -> {entry_price, stop_price, buy_date, shares, notional}
        "daily": {"date": _today(), "buys": 0, "realized_pnl": 0.0, "sold_symbols": []},
        "lifetime": {"buys": 0, "sells": 0, "realized_pnl": 0.0},
    }


def load_state(path: str = STATE_PATH) -> dict:
    if not os.path.exists(path):
        return default_state()
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    # normalize missing keys
    base = default_state()
    for k, v in base.items():
        state.setdefault(k, v)
    return roll_daily(state)


def save_state(state: dict, path: str = STATE_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def roll_daily(state: dict) -> dict:
    """Reset the daily counters when the calendar day (ET) changes."""
    today = _today()
    if state.get("daily", {}).get("date") != today:
        state["daily"] = {"date": today, "buys": 0, "realized_pnl": 0.0, "sold_symbols": []}
    return state


def _append_trade(record: dict) -> None:
    record = {"ts": _now_iso(), **record}
    with open(TRADES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def record_buy(
    state: dict,
    *,
    symbol: str,
    price: float,
    shares: float,
    notional: float,
    stop_price: float | None,
    dry_run: bool,
    target_price: float | None = None,
    order_id: str | None = None,
) -> dict:
    state = roll_daily(state)
    state["positions"][symbol] = {
        "entry_price": round(price, 4),
        "stop_price": round(stop_price, 2) if stop_price is not None else None,
        "target_price": round(target_price, 2) if target_price is not None else None,
        "buy_date": _today(),
        "shares": round(shares, 6),
        "notional": round(notional, 2),
    }
    state["daily"]["buys"] += 1
    state["lifetime"]["buys"] += 1
    _append_trade(
        {
            "action": "BUY",
            "symbol": symbol,
            "price": price,
            "shares": shares,
            "notional": notional,
            "stop_price": stop_price,
            "target_price": target_price,
            "dry_run": dry_run,
            "order_id": order_id,
        }
    )
    return state


def record_sell(
    state: dict,
    *,
    symbol: str,
    price: float,
    shares: float,
    dry_run: bool,
    reason: str = "",
    order_id: str | None = None,
) -> dict:
    state = roll_daily(state)
    pos = state["positions"].get(symbol, {})
    entry = float(pos.get("entry_price", price))
    realized = round((price - entry) * shares, 2)
    state["daily"]["realized_pnl"] = round(state["daily"]["realized_pnl"] + realized, 2)
    if symbol not in state["daily"]["sold_symbols"]:
        state["daily"]["sold_symbols"].append(symbol)
    state["lifetime"]["sells"] += 1
    state["lifetime"]["realized_pnl"] = round(state["lifetime"]["realized_pnl"] + realized, 2)
    state["positions"].pop(symbol, None)
    _append_trade(
        {
            "action": "SELL",
            "symbol": symbol,
            "price": price,
            "shares": shares,
            "entry_price": entry,
            "realized_pnl": realized,
            "reason": reason,
            "dry_run": dry_run,
            "order_id": order_id,
        }
    )
    return state


def update_stop(state: dict, symbol: str, stop_price: float | None) -> dict:
    if symbol in state["positions"] and stop_price is not None:
        state["positions"][symbol]["stop_price"] = round(stop_price, 2)
    return state


# --------------------------------------------------------------------------- #
# CLI — used by the scheduled agent to persist fills after placing MCP orders
# --------------------------------------------------------------------------- #
def _cli() -> None:
    p = argparse.ArgumentParser(description="trading_bot journal")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("record-buy")
    b.add_argument("--symbol", required=True)
    b.add_argument("--price", type=float, required=True)
    b.add_argument("--shares", type=float, required=True)
    b.add_argument("--notional", type=float, required=True)
    b.add_argument("--stop", type=float, default=None)
    b.add_argument("--target", type=float, default=None)
    b.add_argument("--order-id", default=None)
    b.add_argument("--live", action="store_true", help="real order (not dry-run)")

    s = sub.add_parser("record-sell")
    s.add_argument("--symbol", required=True)
    s.add_argument("--price", type=float, required=True)
    s.add_argument("--shares", type=float, required=True)
    s.add_argument("--reason", default="")
    s.add_argument("--order-id", default=None)
    s.add_argument("--live", action="store_true")

    u = sub.add_parser("update-stop")
    u.add_argument("--symbol", required=True)
    u.add_argument("--stop", type=float, required=True)

    sub.add_parser("show")

    args = p.parse_args()
    state = load_state()

    if args.cmd == "record-buy":
        state = record_buy(
            state, symbol=args.symbol, price=args.price, shares=args.shares,
            notional=args.notional, stop_price=args.stop, target_price=args.target,
            dry_run=not args.live, order_id=args.order_id,
        )
    elif args.cmd == "record-sell":
        state = record_sell(
            state, symbol=args.symbol, price=args.price, shares=args.shares,
            dry_run=not args.live, reason=args.reason, order_id=args.order_id,
        )
    elif args.cmd == "update-stop":
        state = update_stop(state, args.symbol, args.stop)
    elif args.cmd == "show":
        print(json.dumps(state, indent=2))
        return

    save_state(state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    _cli()
