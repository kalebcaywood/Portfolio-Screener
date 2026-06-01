"""Hard risk limits — the safety layer.

Every order, buy or sell, passes through here before it can reach the broker.
These checks are intentionally boring, deterministic, and conservative. The
autonomous agent CANNOT override them: it only ever places orders this module
returns as approved.

Checks enforced for BUYS:
  - kill switch (config.enabled) + HALT file
  - market is open (regular hours, weekday) if regular_hours_only
  - daily realized loss has not breached daily_loss_limit
  - max_new_buys_per_day not exceeded
  - max_positions not exceeded
  - symbol not already held (no pyramiding)
  - symbol not sold earlier today (cash-account good-faith / wash avoidance)
  - per-order notional clamped to max_position_notional
  - total deployed (existing mkt value + new) <= max_total_deployed
  - notional <= buying_power - cash_buffer
SELLS are always allowed (risk-reducing), provided shares are available.
"""
from __future__ import annotations

import os
from datetime import datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HALT_FILE = os.path.join(os.path.dirname(__file__), "HALT")


def market_is_open(now_et: datetime) -> tuple[bool, str]:
    if now_et.weekday() >= 5:
        return False, "weekend"
    t = now_et.time()
    if t < time(9, 30):
        return False, "pre-market"
    if t >= time(16, 0):
        return False, "after-hours"
    return True, "regular hours"


def kill_switch_engaged(config: dict) -> tuple[bool, str]:
    if not config.get("enabled", False):
        return True, "config.enabled is false"
    if os.path.exists(HALT_FILE):
        return True, f"HALT file present ({HALT_FILE})"
    return False, ""


def vet(
    *,
    buys: list[dict],
    sells: list[dict],
    snapshot: dict,
    state: dict,
    config: dict,
    now_et: datetime | None = None,
    news: dict | None = None,
) -> dict:
    """Return final approved orders + an audit trail of every decision.

    buys/sells: candidate orders from the strategy layer. Each buy dict has at
    least {symbol, suggested_stop, reason, indicators}. Each sell dict has
    {symbol, reason, ...}.

    snapshot: live account truth from the MCP layer:
      {
        "buying_power": float,
        "positions": {SYM: {"shares": float, "shares_available_for_sells": float,
                             "market_value": float, "average_buy_price": float}},
        "quotes": {SYM: {"price": float}},
      }
    """
    now_et = now_et or datetime.now(ET)
    risk = config["risk"]
    audit: list[str] = []
    approved_buys: list[dict] = []
    approved_sells: list[dict] = []

    positions = snapshot.get("positions", {})
    quotes = snapshot.get("quotes", {})
    buying_power = float(snapshot.get("buying_power", 0.0))
    daily = state.get("daily", {})
    sold_today = set(daily.get("sold_symbols", []))
    buys_today = int(daily.get("buys", 0))
    realized_pnl_today = float(daily.get("realized_pnl", 0.0))

    # ---- account-wide gates ----
    killed, why = kill_switch_engaged(config)
    if killed:
        audit.append(f"BLOCK ALL: kill switch — {why}")
        return _result([], [], audit, blocked=True)

    if risk.get("regular_hours_only", True):
        is_open, mkt = market_is_open(now_et)
        if not is_open:
            audit.append(f"BLOCK BUYS: market {mkt} ({now_et:%Y-%m-%d %H:%M %Z})")
            # sells of fractional/dollar orders also need regular hours; block all writes
            return _result([], [], audit, blocked=True)

    # ---- sells first (free up risk, always allowed) ----
    for s in sells:
        sym = s["symbol"]
        pos = positions.get(sym)
        if not pos:
            audit.append(f"SKIP SELL {sym}: not held in account")
            continue
        avail = float(pos.get("shares_available_for_sells", pos.get("shares", 0.0)))
        if avail <= 0:
            audit.append(f"SKIP SELL {sym}: no sellable shares (unsettled?)")
            continue
        approved_sells.append({**s, "shares_available_for_sells": avail})
        audit.append(f"APPROVE SELL {sym}: {s.get('reason', '')} ({avail} sh)")

    # ---- daily loss circuit breaker (blocks new buys only) ----
    loss_limit = float(risk["daily_loss_limit"])
    if realized_pnl_today <= -abs(loss_limit):
        audit.append(
            f"BLOCK BUYS: daily loss limit hit (realized {realized_pnl_today:.2f} "
            f"<= -{loss_limit:.2f})"
        )
        return _result([], approved_sells, audit)

    # ---- buys ----
    open_count = len(positions)
    deployed = sum(float(p.get("market_value", 0.0)) for p in positions.values())
    max_notional = float(risk["max_position_notional"])
    max_total = float(risk["max_total_deployed"])
    max_positions = int(risk["max_positions"])
    max_buys = int(risk["max_new_buys_per_day"])
    cash_buffer = float(risk["cash_buffer"])
    spendable = buying_power - cash_buffer

    # news filter sets (empty/None when news disabled or in off/alert mode)
    news_cfg = config.get("news", {})
    news_active = bool(news) and news_cfg.get("enabled", False) and \
        news_cfg.get("mode", "filter") in ("filter", "autonomous")
    blocked_news = set(news.get("blocked_symbols", [])) if news_active else set()
    positive_news = set(news.get("positive_symbols", [])) if news_active else set()
    require_positive = bool(news_cfg.get("require_positive_for_buy", False)) and news_active

    max_trades = int(risk.get("max_trades_per_day", 999))
    for b in buys:
        sym = b["symbol"]
        trades_committed = (buys_today + len(sold_today)
                            + len(approved_buys) + len(approved_sells))
        if trades_committed >= max_trades:
            audit.append(f"SKIP BUY {sym}: daily trade cap reached ({max_trades})")
            break
        if buys_today + len(approved_buys) >= max_buys:
            audit.append(f"SKIP BUY {sym}: daily buy cap reached ({max_buys})")
            continue
        if sym in blocked_news:
            audit.append(f"SKIP BUY {sym}: negative news sentiment (news filter)")
            continue
        if require_positive and sym not in positive_news:
            audit.append(f"SKIP BUY {sym}: require_positive_for_buy and no positive news")
            continue
        if sym in positions:
            audit.append(f"SKIP BUY {sym}: already held (no pyramiding)")
            continue
        if sym in sold_today:
            audit.append(f"SKIP BUY {sym}: sold earlier today (good-faith/wash guard)")
            continue
        if open_count + len(approved_buys) >= max_positions:
            audit.append(f"SKIP BUY {sym}: max positions reached ({max_positions})")
            continue

        notional = max_notional
        # clamp to remaining total-deployed headroom
        pending_notional = sum(float(x["notional"]) for x in approved_buys)
        total_headroom = max_total - deployed - pending_notional
        if total_headroom <= 0:
            audit.append(f"SKIP BUY {sym}: total-deployed cap reached (${max_total:.0f})")
            continue
        notional = min(notional, total_headroom)
        # clamp to spendable cash
        cash_headroom = spendable - pending_notional
        if cash_headroom < 1.0:
            audit.append(
                f"SKIP BUY {sym}: insufficient buying power "
                f"(spendable ${spendable:.2f} after buffer)"
            )
            continue
        notional = min(notional, cash_headroom)
        notional = round(notional, 2)
        min_notional = float(risk.get("min_order_notional", 1.0))
        if notional < min_notional:
            audit.append(
                f"SKIP BUY {sym}: notional ${notional:.2f} < min ${min_notional:.2f} after clamps"
            )
            continue

        q = quotes.get(sym, {})
        price = float(q.get("price", 0.0)) or None
        approved_buys.append(
            {
                "symbol": sym,
                "side": "buy",
                "notional": notional,
                "ref_price": price,
                "stop_price": b.get("suggested_stop"),
                "target_price": b.get("suggested_target"),
                "reward_risk": b.get("reward_risk"),
                "reason": b.get("reason", ""),
                "indicators": b.get("indicators", {}),
            }
        )
        audit.append(
            f"APPROVE BUY {sym}: ${notional:.2f} notional — {b.get('reason', '')}"
        )

    return _result(approved_buys, approved_sells, audit)


def _result(buys, sells, audit, blocked: bool = False) -> dict:
    return {
        "approved_buys": buys,
        "approved_sells": sells,
        "audit": audit,
        "blocked": blocked,
    }
