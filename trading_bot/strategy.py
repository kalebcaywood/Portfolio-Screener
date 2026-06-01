"""Deterministic technical-analysis strategy.

All trade *decisions* live here as pure functions of price history. No I/O,
no network, no broker calls — so the logic is fully testable and the LLM agent
never improvises a signal.

Strategy (daily bars):
  ENTRY (when flat):
    - fast SMA crosses ABOVE slow SMA (golden cross)        -> momentum turning up
    - AND close > trend SMA                                  -> longer-term uptrend filter
    - AND RSI < rsi_overbought                               -> not buying into froth
  EXIT (when holding), first matching rule wins:
    - close <= trailing stop  (ATR-based, ratchets up)       -> hard risk stop
    - close >= entry * (1 + take_profit_pct)                 -> take profit
    - fast SMA crosses BELOW slow SMA (death cross)          -> momentum turning down
    - RSI >= rsi_exit                                        -> overbought, ring the register
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 -> no down moves -> maximally strong
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's Average True Range. Expects High/Low/Close columns."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()


# --------------------------------------------------------------------------- #
# Decision engine
# --------------------------------------------------------------------------- #
def _f(x) -> float | None:
    """Coerce to plain float, NaN/None -> None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def evaluate(
    df: pd.DataFrame,
    params: dict,
    *,
    holding: bool,
    entry_price: float | None = None,
    stop_price: float | None = None,
    target_price: float | None = None,
) -> dict:
    """Evaluate one symbol.

    Returns a dict:
      action: "BUY" | "SELL" | "HOLD" | "NONE"
      reason: human-readable explanation
      indicators: latest indicator snapshot
      suggested_stop: ATR stop to record on a BUY, or ratcheted stop on HOLD
    """
    fast_n = params["fast_ma"]
    slow_n = params["slow_ma"]
    trend_n = params["trend_ma"]
    rsi_n = params["rsi_period"]
    atr_n = params["atr_period"]
    rsi_ob = params["rsi_overbought"]
    rsi_exit = params["rsi_exit"]
    stop_mult = params["atr_stop_mult"]
    tp_pct = params["take_profit_pct"]
    tp_atr_mult = params.get("tp_atr_mult", 4.0)
    resist_lb = int(params.get("resistance_lookback", 60))
    min_rr = float(params.get("min_reward_risk", 1.5))

    need = max(slow_n, trend_n, rsi_n, atr_n) + 2
    if df is None or len(df) < need:
        return {
            "action": "NONE",
            "reason": f"insufficient history ({0 if df is None else len(df)} < {need} bars)",
            "indicators": {},
            "suggested_stop": None,
        }

    fast = sma(df["Close"], fast_n)
    slow = sma(df["Close"], slow_n)
    trend = sma(df["Close"], trend_n)
    rsi_s = rsi(df["Close"], rsi_n)
    atr_s = atr(df, atr_n)

    close = _f(df["Close"].iloc[-1])
    fast_now, fast_prev = _f(fast.iloc[-1]), _f(fast.iloc[-2])
    slow_now, slow_prev = _f(slow.iloc[-1]), _f(slow.iloc[-2])
    trend_now = _f(trend.iloc[-1])
    rsi_now = _f(rsi_s.iloc[-1])
    atr_now = _f(atr_s.iloc[-1])

    ind = {
        "close": close,
        "fast_ma": fast_now,
        "slow_ma": slow_now,
        "trend_ma": trend_now,
        "rsi": rsi_now,
        "atr": atr_now,
    }

    if None in (close, fast_now, fast_prev, slow_now, slow_prev, trend_now, rsi_now, atr_now):
        return {
            "action": "NONE",
            "reason": "indicator NaN (stale/missing data)",
            "indicators": ind,
            "suggested_stop": None,
        }

    cross_up = fast_prev <= slow_prev and fast_now > slow_now
    cross_down = fast_prev >= slow_prev and fast_now < slow_now

    if not holding:
        if cross_up and close > trend_now and rsi_now < rsi_ob:
            stop = round(close - stop_mult * atr_now, 2)
            risk = stop_mult * atr_now  # = close - stop, per-share downside

            # overhead resistance = highest high over lookback, excluding today
            highs = df["High"].iloc[-(resist_lb + 1):-1]
            swing_high = float(highs.max()) if len(highs) else close

            if close >= swing_high:
                # breakout / at highs: clear overhead -> project an ATR target
                target = close + tp_atr_mult * atr_now
                reward = tp_atr_mult * atr_now
                tgt_kind = "breakout"
            else:
                # room to run up to the prior ceiling
                target = swing_high
                reward = swing_high - close
                tgt_kind = "resistance"

            rr = round(reward / risk, 2) if risk > 0 else 0.0
            # NOTE: reward:risk is used to RANK/SELECT among same-day candidates
            # in plan.py (and shown as the price target), NOT as a hard reject —
            # backtesting showed a hard min-R:R filter removed net-winning trades.
            # min_rr is retained only as an optional, default-off floor.
            if min_rr > 0 and rr < min_rr:
                return {
                    "action": "NONE",
                    "reason": (f"signal but reward:risk {rr:.2f} < {min_rr:.2f} "
                               f"(target {target:.2f}, stop {stop:.2f})"),
                    "indicators": ind,
                    "suggested_stop": None,
                }
            return {
                "action": "BUY",
                "reason": (
                    f"golden cross (fast {fast_now:.2f} > slow {slow_now:.2f}), "
                    f"close {close:.2f} > trend {trend_now:.2f}, RSI {rsi_now:.1f} < {rsi_ob}; "
                    f"R:R {rr:.2f} (target {target:.2f} [{tgt_kind}], stop {stop:.2f})"
                ),
                "indicators": ind,
                "suggested_stop": stop,
                "suggested_target": round(target, 2),
                "reward_risk": rr,
            }
        # explain the closest miss for transparency
        reasons = []
        if not cross_up:
            reasons.append("no golden cross")
        if not close > trend_now:
            reasons.append(f"below trend ({close:.2f}<={trend_now:.2f})")
        if not rsi_now < rsi_ob:
            reasons.append(f"RSI {rsi_now:.1f} >= {rsi_ob}")
        return {
            "action": "NONE",
            "reason": "no entry: " + ", ".join(reasons),
            "indicators": ind,
            "suggested_stop": None,
        }

    # ---- holding: exit logic ----
    eff_stop = stop_price
    if eff_stop is not None and close <= eff_stop:
        return _sell(ind, f"stop hit (close {close:.2f} <= stop {eff_stop:.2f})")

    # Profit target — PROVEN flat take-profit (backtested best). The dynamic
    # resistance/ATR target (suggested_target) is computed at entry and used only
    # to RANK candidates and to show the user a price target; backtesting showed
    # using it as the exit gave gains back, so the exit stays flat.
    if entry_price is not None:
        tp = entry_price * (1.0 + tp_pct)
        if close >= tp:
            return _sell(ind, f"take profit (close {close:.2f} >= +{tp_pct*100:.0f}% {tp:.2f})")

    if cross_down:
        return _sell(ind, f"death cross (fast {fast_now:.2f} < slow {slow_now:.2f})")

    if rsi_now >= rsi_exit:
        return _sell(ind, f"overbought exit (RSI {rsi_now:.1f} >= {rsi_exit})")

    # hold and ratchet the trailing stop upward only
    new_stop = round(close - stop_mult * atr_now, 2)
    ratcheted = new_stop if eff_stop is None else round(max(eff_stop, new_stop), 2)
    return {
        "action": "HOLD",
        "reason": f"holding; trailing stop {ratcheted:.2f}, RSI {rsi_now:.1f}",
        "indicators": ind,
        "suggested_stop": ratcheted,
    }


def _sell(ind: dict, reason: str) -> dict:
    return {"action": "SELL", "reason": reason, "indicators": ind, "suggested_stop": None}
