"""Quick historical sanity-check of the strategy.

This is a per-symbol, single-position, fills-at-close walk-forward — deliberately
simple. It exists to confirm the rules behave sanely (reasonable trade count, not
catastrophic), NOT to promise profit. Real fills, slippage, settlement, and the
portfolio-level risk caps are not modeled here.

    python -m trading_bot.backtest
    python -m trading_bot.backtest --period 3y --symbols SPY QQQ AAPL
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yfinance as yf

from . import strategy

HERE = os.path.dirname(__file__)


def backtest_symbol(df, params) -> dict:
    need = max(params["slow_ma"], params["trend_ma"], params["rsi_period"], params["atr_period"]) + 2
    if len(df) < need + 5:
        return {"trades": 0, "skipped": True}

    holding = False
    entry_price = stop_price = target_price = None
    trade_returns = []
    equity = 1.0
    equity_curve = [1.0]

    for i in range(need, len(df)):
        window = df.iloc[: i + 1]
        res = strategy.evaluate(
            window, params, holding=holding,
            entry_price=entry_price, stop_price=stop_price,
            target_price=target_price,
        )
        close = float(window["Close"].iloc[-1])

        if not holding and res["action"] == "BUY":
            holding = True
            entry_price = close
            stop_price = res["suggested_stop"]
            target_price = res.get("suggested_target")
        elif holding:
            if res["action"] == "SELL":
                ret = (close - entry_price) / entry_price
                trade_returns.append(ret)
                equity *= (1 + ret)
                holding = entry_price = stop_price = target_price = None
            else:
                stop_price = res.get("suggested_stop", stop_price)
        equity_curve.append(equity if not holding else equity * (close / entry_price))

    # close any open position at the last price (mark-to-market)
    if holding:
        close = float(df["Close"].iloc[-1])
        ret = (close - entry_price) / entry_price
        trade_returns.append(ret)
        equity *= (1 + ret)

    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak).min()) if len(eq) else 0.0
    wins = [r for r in trade_returns if r > 0]
    bh = float(df["Close"].iloc[-1] / df["Close"].iloc[need] - 1.0)

    return {
        "trades": len(trade_returns),
        "win_rate": round(len(wins) / len(trade_returns), 3) if trade_returns else None,
        "avg_trade": round(float(np.mean(trade_returns)), 4) if trade_returns else None,
        "total_return": round(equity - 1.0, 4),
        "buy_hold_return": round(bh, 4),
        "max_drawdown": round(max_dd, 4),
        "skipped": False,
    }


def main() -> None:
    with open(os.path.join(HERE, "config.json")) as f:
        config = json.load(f)
    p = argparse.ArgumentParser()
    p.add_argument("--period", default="3y")
    p.add_argument("--symbols", nargs="*", default=config["universe"])
    args = p.parse_args()
    params = config["strategy"]

    raw = yf.download(args.symbols, period=args.period, interval="1d",
                      auto_adjust=True, progress=False, group_by="ticker")
    multi = hasattr(raw.columns, "levels")

    rows = []
    agg_trades = 0
    strat_rets, bh_rets = [], []
    for sym in args.symbols:
        try:
            df = (raw[sym] if multi else raw)[["High", "Low", "Close"]].dropna()
        except Exception:
            continue
        r = backtest_symbol(df, params)
        if r.get("skipped"):
            continue
        rows.append((sym, r))
        agg_trades += r["trades"]
        strat_rets.append(r["total_return"])
        bh_rets.append(r["buy_hold_return"])

    print(f"=== backtest ({args.period}) — strategy vs buy&hold, per symbol ===")
    print(f"{'sym':6}{'trades':>7}{'win%':>7}{'avgT':>8}{'strat':>9}{'B&H':>9}{'maxDD':>9}")
    for sym, r in rows:
        wr = f"{r['win_rate']*100:.0f}" if r["win_rate"] is not None else "-"
        at = f"{r['avg_trade']*100:.1f}" if r["avg_trade"] is not None else "-"
        print(f"{sym:6}{r['trades']:>7}{wr:>7}{at:>8}"
              f"{r['total_return']*100:>8.1f}%{r['buy_hold_return']*100:>8.1f}%"
              f"{r['max_drawdown']*100:>8.1f}%")
    if strat_rets:
        print("-" * 55)
        print(f"{'MEAN':6}{agg_trades:>7}{'':>7}{'':>8}"
              f"{np.mean(strat_rets)*100:>8.1f}%{np.mean(bh_rets)*100:>8.1f}%")
        print("\nNote: per-symbol, no costs/slippage/settlement, no portfolio caps. "
              "Sanity check only — not a profit forecast.")


if __name__ == "__main__":
    main()
